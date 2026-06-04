"""
Model loader and scoring logic.
Loaded once at API startup — not per request.
"""
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from src.api.utils import prob_to_score, get_risk_band, get_risk_band_color
from src.features.synthetic_data import generate_upi_features   # for feature engineering
import sys

# Allow running from project root
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PROC_DIR   = ROOT / "data" / "processed"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as the notebook.
    Must stay in sync with Cell 10 of 01_eda_and_features.ipynb.
    """
    bill_cols = [f'bill_amt{i}' for i in range(1, 7)]
    pay_cols  = [f'pay_amt{i}'  for i in range(1, 7)]

    df['avg_bill_6m']       = df[bill_cols].mean(axis=1)
    df['avg_payment_6m']    = df[pay_cols].mean(axis=1)
    df['bill_trend']        = df['bill_amt1'] - df['bill_amt6']
    df['payment_trend']     = df['pay_amt1']  - df['pay_amt6']

    total_bill = df[bill_cols].sum(axis=1).replace(0, np.nan)
    total_paid = df[pay_cols].sum(axis=1)
    df['payment_ratio'] = (total_paid / total_bill).clip(0, 5).fillna(0)

    pay_status_cols = [f'pay_{i}' for i in [0, 2, 3, 4, 5, 6]]
    df['repayment_consistency'] = (df[pay_status_cols] <= 0).mean(axis=1)
    df['max_delay_status']      = df[pay_status_cols].max(axis=1)
    df['credit_utilisation']    = (
        df['avg_bill_6m'] / df['limit_bal'].replace(0, np.nan)
    ).clip(0, 2).fillna(0)

    df['age_band'] = pd.cut(
        df['age'], bins=[0, 25, 35, 45, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    df['spend_momentum']   = df['avg_txn_amount'] * df['monthly_txn_count']
    df['payment_discipline'] = (
        df['utility_payment_ratio']   * 0.40 +
        df['recharge_consistency']    * 0.30 +
        (1 - df['late_payment_flag']) * 0.30
    )
    df['financial_stability'] = (
        df['salary_credit_regular'] * 0.50 +
        df['txn_regularity_score']  * 0.30 +
        df['payment_discipline']    * 0.20
    )
    df['spend_volatility'] = (
        df['txn_amount_std'] / df['avg_txn_amount'].replace(0, np.nan)
    ).clip(0, 10).fillna(0)

    df['util_x_discipline']    = df['credit_utilisation'] * df['payment_discipline']
    df['repayment_x_stability']= df['repayment_consistency'] * df['financial_stability']

    return df


class CreditScorer:
    """Singleton scorer — loaded once at startup."""

    MODEL_VERSION = "v1.0"

    def __init__(self):
        self.model       = joblib.load(MODELS_DIR / "xgb_tuned.pkl")
        self.explainer   = joblib.load(MODELS_DIR / "shap_explainer.pkl")
        self.feature_cols= joblib.load(MODELS_DIR / "feature_cols.pkl")
        print(f"[Scorer] Loaded model with {len(self.feature_cols)} features.")

    def score(self, applicant_id: str, features: dict) -> dict:
        """Score a single applicant. Returns full ScoreResponse dict."""

        # Build raw dataframe
        df_raw = pd.DataFrame([features])

        # Apply feature engineering
        df_eng = _engineer_features(df_raw)

        # Align to exact training columns
        X = df_eng.reindex(columns=self.feature_cols, fill_value=0)

        # Predict
        prob        = float(self.model.predict_proba(X)[:, 1][0])
        score       = prob_to_score(prob)
        band        = get_risk_band(score)
        band_color  = get_risk_band_color(band)

        # SHAP explanation
        shap_vals    = self.explainer.shap_values(X)[0]
        feature_impact = list(zip(self.feature_cols, shap_vals))
        sorted_impact  = sorted(feature_impact, key=lambda x: x[1])

        def make_factor(name, val):
            return {
                "feature":    name,
                "shap_value": round(float(val), 4),
                "direction":  "positive" if val > 0 else "negative"
            }

        # Bottom 3 = most negative (hurt the score)
        # Top 3    = most positive (helped the score)
        top_negative = [make_factor(n, v) for n, v in sorted_impact[:3]]
        top_positive = [make_factor(n, v) for n, v in sorted_impact[-3:][::-1]]

        return {
            "applicant_id":         applicant_id,
            "score":                score,
            "risk_band":            band,
            "risk_band_color":      band_color,
            "default_probability":  round(prob, 4),
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "model_version":        self.MODEL_VERSION,
        }

    def score_batch(self, items: list[dict]) -> list[dict]:
        """Score a list of applicants."""
        return [self.score(item["applicant_id"], item["features"]) for item in items]


# Module-level singleton — imported by main.py
scorer = CreditScorer()
import numpy as np
import pandas as pd


def generate_upi_features(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    """
    Returns a DataFrame of synthetic UPI/mobile transaction features.
    Each row represents one borrower's aggregated 12-month profile.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        # Volume signals
        "monthly_txn_count":      rng.poisson(45, n).clip(1, 200),
        "avg_txn_amount":         rng.lognormal(7.0, 1.2, n).clip(50, 100000),
        "txn_amount_std":         rng.lognormal(6.0, 1.0, n).clip(10, 50000),

        # Regularity signals
        "txn_regularity_score":   rng.beta(5, 2, n),       # 0–1
        "salary_credit_regular":  rng.binomial(1, 0.65, n),
        "recharge_consistency":   rng.beta(6, 2, n),

        # Payment discipline
        "utility_payment_ratio":  rng.beta(8, 3, n),
        "late_payment_flag":      rng.binomial(1, 0.18, n),
        "emi_to_income_proxy":    rng.beta(2, 5, n),        # lower = healthier

        # Spend behaviour
        "merchant_diversity":     rng.integers(3, 40, n),
        "weekend_spend_ratio":    rng.uniform(0.1, 0.5, n),
        "essential_spend_ratio":  rng.beta(6, 3, n),        # groceries, utilities, etc.
        "p2p_to_merchant_ratio":  rng.beta(3, 5, n),

        # Stability signals
        "months_active":          rng.integers(3, 24, n),
        "balance_volatility":     rng.beta(3, 5, n),        # lower = more stable
    })

    return df


def merge_with_labels(
    upi_df: pd.DataFrame,
    label_series: pd.Series,
    seed: int = 42
) -> pd.DataFrame:
    """
    Attach ground-truth default labels with WEAK correlation to synthetic features.
    Nudge magnitude kept small to simulate realistic (noisy) alternative data signals.
    """
    rng = np.random.default_rng(seed)
    n = len(label_series)
    upi_df = upi_df.iloc[:n].copy().reset_index(drop=True)
    labels = label_series.reset_index(drop=True)

    defaulters = labels == 1
    d = defaulters.sum()

    # Weak nudges — realistic correlation, not near-perfect separation
    upi_df.loc[defaulters, "txn_regularity_score"]  *= rng.uniform(0.88, 0.97, d)
    upi_df.loc[defaulters, "utility_payment_ratio"]  *= rng.uniform(0.85, 0.95, d)
    upi_df.loc[defaulters, "late_payment_flag"]       = rng.binomial(1, 0.28, d)
    upi_df.loc[defaulters, "salary_credit_regular"]   = rng.binomial(1, 0.52, d)
    upi_df.loc[defaulters, "balance_volatility"]      *= rng.uniform(1.02, 1.08, d)
    upi_df.loc[defaulters, "recharge_consistency"]    *= rng.uniform(0.92, 0.98, d)

    upi_df["default_flag"] = labels
    return upi_df

if __name__ == "__main__":
    df = generate_upi_features()
    df.to_csv("data/synthetic/upi_features_raw.csv", index=False)
    print(f"Generated {len(df):,} rows → data/synthetic/upi_features_raw.csv")
    print(df.describe().round(3))
# Credit Score Engine

An end-to-end alternative credit scoring system that combines traditional bureau features with UPI transaction signals to score borrowers on a 300–900 scale — with full SHAP explainability and a live underwriter dashboard.

Built as a portfolio project demonstrating ML engineering for AI-driven fintech and digital lending use cases.

---

## Live demo

| Component | Link |
|---|---|
| Streamlit dashboard | [credit-score-engine.hf.space](https://huggingface.co/spaces/Tasty-kadai-paneer/credit-score-engine) |


---

## Results

| Model | AUC | Gini | KS Statistic |
|---|---|---|---|
| Logistic Regression (baseline) | 0.8045 | 0.6090 | — |
| Decision Tree (baseline) | 0.8159 | 0.6317 | — |
| XGBoost (tuned) | **0.8466** | **0.6932** | **0.5186** |
| LightGBM (tuned) | 0.8456 | 0.6912 | 0.5265 |
| Stacking Ensemble | 0.8468 | 0.6936 | 0.5227 |

XGBoost selected as production model — highest single-model AUC with cleanest SHAP compatibility.

---

## Architecture

```
Raw data (UCI + synthetic UPI)
        ↓
Feature engineering (36 features)
        ↓
SMOTE oversampling → XGBoost (Optuna-tuned)
        ↓
PDO-calibrated score (300–900)
        ↓
┌──────────────────┬─────────────────────┐
│  FastAPI service │  Streamlit dashboard│
│  /score          │  Page 1: Performance│
│  /score/batch    │  Page 2: Scorer     │
│  /health         │  Page 3: Portfolio  │
└──────────────────┴─────────────────────┘
```

---

## Features

**Bureau signals (UCI dataset)**
- Credit limit, age, education, marital status
- 6-month repayment status history
- 6-month bill and payment amounts
- Engineered: payment ratio, credit utilisation, repayment consistency, bill trend

**Alternative data signals (synthetic UPI)**
- Monthly transaction count, average amount, spend volatility
- Transaction regularity score, merchant diversity
- Utility payment ratio, recharge consistency
- Salary credit regularity, late payment flag
- Engineered: payment discipline index, financial stability score, spend momentum

---

## Project structure

```
credit-score-engine/
├── data/
│   ├── raw/              # UCI dataset (gitignored)
│   ├── processed/        # engineered features, test set
│   └── synthetic/        # generated UPI features
├── notebooks/
│   ├── 01_eda_and_features.ipynb
│   └── 02_modelling.ipynb
├── src/
│   ├── features/
│   │   └── synthetic_data.py
│   ├── models/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── scorer.py
│   │   └── utils.py
│   └── dashboard/
│       └── app.py
├── tests/
│   ├── test_synthetic_data.py
│   └── test_api.py
├── models/               # saved artifacts (gitignored)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and create environment
git clone https://github.com/YOUR_USERNAME/credit-score-engine.git
cd credit-score-engine
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data and generate synthetic features
python src/data_download.py
python src/features/synthetic_data.py

# 4. Run notebooks in order
jupyter notebook
# → notebooks/01_eda_and_features.ipynb
# → notebooks/02_modelling.ipynb

# 5. Start API
uvicorn src.api.main:app --reload --port 8000

# 6. Start dashboard
streamlit run src/dashboard/app.py
```

---

## API usage

```bash
# Health check
curl http://localhost:8000/health

# Score a single applicant
curl -X POST http://localhost:8000/score \
     -H "Content-Type: application/json" \
     -d '{"applicant_id": "TEST_001", "features": {...}}'

# Get example payload
curl http://localhost:8000/score/example
```

**Response:**
```json
{
  "applicant_id": "TEST_001",
  "score": 728,
  "risk_band": "LOW",
  "risk_band_color": "#82E0AA",
  "default_probability": 0.1821,
  "top_positive_factors": [
    {"feature": "repayment_consistency", "shap_value": 0.312, "direction": "positive"}
  ],
  "top_negative_factors": [
    {"feature": "max_delay_status", "shap_value": -0.041, "direction": "negative"}
  ],
  "model_version": "v1.0"
}
```

---

## Score bands

| Score range | Risk band | Typical recommendation |
|---|---|---|
| 750 – 900 | VERY LOW | Auto-approve |
| 650 – 749 | LOW | Approve |
| 550 – 649 | MEDIUM | Manual review |
| 450 – 549 | HIGH | Decline or secured product |
| 300 – 449 | VERY HIGH | Decline |

---

## Key concepts demonstrated

- **PDO score calibration** — Points to Double Odds scaling anchored to dataset default rate
- **SHAP explainability** — Per-applicant waterfall charts for regulatory transparency
- **Class imbalance handling** — SMOTE oversampling + scale_pos_weight
- **Fairness auditing** — AUC stability across age and credit limit segments
- **Alternative data signals** — UPI transaction patterns as credit proxies
- **Production API design** — Pydantic validation, batch scoring, health checks

---

## Limitations

- UPI features are synthetically generated with weak label correlation to simulate real alternative data. In production these would be sourced from an Account Aggregator (AA) consent feed and retrained on actual transaction data.
- The UCI dataset is Taiwan credit card data from 2005 — not Indian lending data. Feature distributions would differ on a real Indian NBFC portfolio.
- No model monitoring or drift detection implemented (PSI tracking would be the next addition).

---

## Tech stack

`Python 3.11` · `XGBoost` · `LightGBM` · `scikit-learn` · `SHAP` · `Optuna` · `FastAPI` · `Streamlit` · `Plotly` · `pandas` · `imbalanced-learn`

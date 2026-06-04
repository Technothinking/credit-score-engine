import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

SAMPLE_FEATURES = {
    "limit_bal": 50000, "sex": 2, "education": 2,
    "marriage": 1, "age": 35,
    "pay_0": 0, "pay_2": 0, "pay_3": 0,
    "pay_4": 0, "pay_5": 0, "pay_6": 0,
    "bill_amt1": 12000, "bill_amt2": 11000, "bill_amt3": 10500,
    "bill_amt4": 9800,  "bill_amt5": 9200,  "bill_amt6": 8900,
    "pay_amt1": 2000,  "pay_amt2": 1800,   "pay_amt3": 1700,
    "pay_amt4": 1600,  "pay_amt5": 1500,   "pay_amt6": 1400,
    "monthly_txn_count": 52,    "avg_txn_amount": 1800,
    "txn_amount_std": 600,      "txn_regularity_score": 0.82,
    "salary_credit_regular": 1, "recharge_consistency": 0.88,
    "utility_payment_ratio": 0.91, "late_payment_flag": 0,
    "emi_to_income_proxy": 0.28,   "merchant_diversity": 18,
    "weekend_spend_ratio": 0.31,   "essential_spend_ratio": 0.64,
    "p2p_to_merchant_ratio": 0.22, "months_active": 18,
    "balance_volatility": 0.21
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_score_returns_200():
    payload = {"applicant_id": "TEST_001", "features": SAMPLE_FEATURES}
    r = client.post("/score", json=payload)
    assert r.status_code == 200

def test_score_range():
    payload = {"applicant_id": "TEST_001", "features": SAMPLE_FEATURES}
    r = client.post("/score", json=payload)
    assert 300 <= r.json()["score"] <= 900

def test_score_has_shap_factors():
    payload = {"applicant_id": "TEST_001", "features": SAMPLE_FEATURES}
    r = client.post("/score", json=payload)
    body = r.json()
    assert len(body["top_positive_factors"]) == 3
    assert len(body["top_negative_factors"]) == 3

def test_risk_band_valid():
    payload = {"applicant_id": "TEST_001", "features": SAMPLE_FEATURES}
    r = client.post("/score", json=payload)
    assert r.json()["risk_band"] in \
           ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
def test_batch_scoring():
    payload = [
        {"applicant_id": f"APP_{i:03d}", "features": SAMPLE_FEATURES}
        for i in range(5)
    ]
    r = client.post("/score/batch", json=payload)
    assert r.status_code == 200
    assert r.json()["total"] == 5

def test_batch_limit():
    payload = [
        {"applicant_id": f"APP_{i}", "features": SAMPLE_FEATURES}
        for i in range(101)
    ]
    r = client.post("/score/batch", json=payload)
    assert r.status_code == 400

from src.api.schemas import (
    ApplicantFeatures, ScoreResponse, ScoreRequest,
    BatchItem, BatchResponse, HealthResponse, BaseModel
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from src.api.schemas import (
    ApplicantFeatures, ScoreResponse,
    BatchItem, BatchResponse, HealthResponse
)
from src.api.scorer import scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    print("[API] Credit Score Engine starting up...")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Credit Score Engine",
    description="Alternative data credit scoring API with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Service health check."""
    return {
        "status":        "ok",
        "model_version": scorer.MODEL_VERSION,
        "feature_count": len(scorer.feature_cols),
    }


class ScoreRequest(BaseModel):
    applicant_id: str = "APPLICANT_001"
    features: ApplicantFeatures


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_applicant(request: ScoreRequest):
    """
    Score a single applicant.

    Returns a 300–900 credit score, risk band,
    default probability, and top SHAP factors.
    """
    try:
        result = scorer.score(request.applicant_id, request.features.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchResponse, tags=["Scoring"])
def score_batch(items: list[BatchItem]):
    """
    Score up to 100 applicants in one request.
    """
    if len(items) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 100 applicants per request."
        )
    try:
        payload = [{"applicant_id": i.applicant_id,
                    "features": i.features.model_dump()} for i in items]
        results = scorer.score_batch(payload)
        return {"total": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/score/example", tags=["Scoring"])
def example_payload():
    """
    Returns a sample request payload you can paste into /docs.
    Useful for testing without building a full client.
    """
    return {
        "applicant_id": "TEST_001",
        "features": {
            "limit_bal": 50000, "sex": 2, "education": 2,
            "marriage": 1, "age": 35,
            "pay_0": 0,  "pay_2": 0,  "pay_3": 0,
            "pay_4": 0,  "pay_5": 0,  "pay_6": 0,
            "bill_amt1": 12000, "bill_amt2": 11000, "bill_amt3": 10500,
            "bill_amt4": 9800,  "bill_amt5": 9200,  "bill_amt6": 8900,
            "pay_amt1": 2000,   "pay_amt2": 1800,   "pay_amt3": 1700,
            "pay_amt4": 1600,   "pay_amt5": 1500,   "pay_amt6": 1400,
            "monthly_txn_count": 52,    "avg_txn_amount": 1800,
            "txn_amount_std": 600,      "txn_regularity_score": 0.82,
            "salary_credit_regular": 1, "recharge_consistency": 0.88,
            "utility_payment_ratio": 0.91, "late_payment_flag": 0,
            "emi_to_income_proxy": 0.28,   "merchant_diversity": 18,
            "weekend_spend_ratio": 0.31,   "essential_spend_ratio": 0.64,
            "p2p_to_merchant_ratio": 0.22, "months_active": 18,
            "balance_volatility": 0.21
        }
    }
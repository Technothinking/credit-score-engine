"""
Pydantic models for request validation and response serialization.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ApplicantFeatures(BaseModel):
    """
    All features the model expects.
    Mirrors the columns in features_engineered.csv.
    """

    # --- UCI features ---
    limit_bal:   float = Field(..., gt=0,       description="Credit limit in NT dollars")
    sex:         int   = Field(..., ge=1, le=2, description="1=male, 2=female")
    education:   int   = Field(..., ge=0, le=6, description="1=grad, 2=university, 3=high school, 4=others")
    marriage:    int   = Field(..., ge=0, le=3, description="1=married, 2=single, 3=others")
    age:         int   = Field(..., ge=18, le=100)

    pay_0:       int   = Field(..., ge=-2, le=8, description="Repayment status Sep")
    pay_2:       int   = Field(..., ge=-2, le=8, description="Repayment status Aug")
    pay_3:       int   = Field(..., ge=-2, le=8, description="Repayment status Jul")
    pay_4:       int   = Field(..., ge=-2, le=8, description="Repayment status Jun")
    pay_5:       int   = Field(..., ge=-2, le=8, description="Repayment status May")
    pay_6:       int   = Field(..., ge=-2, le=8, description="Repayment status Apr")

    bill_amt1:   float = Field(..., description="Bill amount Sep")
    bill_amt2:   float = Field(..., description="Bill amount Aug")
    bill_amt3:   float = Field(..., description="Bill amount Jul")
    bill_amt4:   float = Field(..., description="Bill amount Jun")
    bill_amt5:   float = Field(..., description="Bill amount May")
    bill_amt6:   float = Field(..., description="Bill amount Apr")

    pay_amt1:    float = Field(..., ge=0, description="Payment amount Sep")
    pay_amt2:    float = Field(..., ge=0, description="Payment amount Aug")
    pay_amt3:    float = Field(..., ge=0, description="Payment amount Jul")
    pay_amt4:    float = Field(..., ge=0, description="Payment amount Jun")
    pay_amt5:    float = Field(..., ge=0, description="Payment amount May")
    pay_amt6:    float = Field(..., ge=0, description="Payment amount Apr")

    # --- UPI / alternative data features ---
    monthly_txn_count:      float = Field(..., ge=0)
    avg_txn_amount:         float = Field(..., ge=0)
    txn_amount_std:         float = Field(..., ge=0)
    txn_regularity_score:   float = Field(..., ge=0, le=1)
    salary_credit_regular:  int   = Field(..., ge=0, le=1)
    recharge_consistency:   float = Field(..., ge=0, le=1)
    utility_payment_ratio:  float = Field(..., ge=0, le=1)
    late_payment_flag:      int   = Field(..., ge=0, le=1)
    emi_to_income_proxy:    float = Field(..., ge=0, le=1)
    merchant_diversity:     int   = Field(..., ge=0)
    weekend_spend_ratio:    float = Field(..., ge=0, le=1)
    essential_spend_ratio:  float = Field(..., ge=0, le=1)
    p2p_to_merchant_ratio:  float = Field(..., ge=0, le=1)
    months_active:          int   = Field(..., ge=0)
    balance_volatility:     float = Field(..., ge=0, le=1)


class FactorDetail(BaseModel):
    feature:    str
    shap_value: float
    direction:  str    # "positive" | "negative"


class ScoreResponse(BaseModel):
    applicant_id:         str
    score:                int
    risk_band:            str
    risk_band_color:      str
    default_probability:  float
    top_positive_factors: list[FactorDetail]
    top_negative_factors: list[FactorDetail]
    model_version:        str


class BatchItem(BaseModel):
    applicant_id: str
    features:     ApplicantFeatures


class BatchResponse(BaseModel):
    total:   int
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    feature_count: int

class ScoreRequest(BaseModel):
    applicant_id: str = "APPLICANT_001"
    features: ApplicantFeatures
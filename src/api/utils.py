"""
PDO score calibration and helper utilities.
"""
import numpy as np


def prob_to_score(
    probability: float,
    pdo: int = 20,
    base_score: int = 600,
    base_odds: float = 3.53,   # calibrated to 22% default rate dataset
) -> int:
    """
    Convert default probability to a 300–900 credit score.

    base_odds = (1 - default_rate) / default_rate
    At base_score=600, good:bad ratio equals base_odds.
    PDO=20 means score drops 20 points each time odds double.
    """
    odds  = (1 - probability) / (probability + 1e-9)
    score = base_score + pdo * np.log2(odds / base_odds)
    return int(np.clip(score, 300, 900))


def get_risk_band(score: int) -> str:
    """Map score to human-readable risk band."""
    if score >= 750:
        return "VERY_LOW"
    elif score >= 650:
        return "LOW"
    elif score >= 550:
        return "MEDIUM"
    elif score >= 450:
        return "HIGH"
    else:
        return "VERY_HIGH"


def get_risk_band_color(band: str) -> str:
    """For dashboard use."""
    return {
        "VERY_LOW":  "#2ECC71",
        "LOW":       "#82E0AA",
        "MEDIUM":    "#F39C12",
        "HIGH":      "#E74C3C",
        "VERY_HIGH": "#922B21",
    }.get(band, "#888888")
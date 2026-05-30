import pandas as pd
import sys
sys.path.insert(0, '.')
from src.features.synthetic_data import generate_upi_features, merge_with_labels

def test_shape():
    df = generate_upi_features(n=1000)
    assert df.shape == (1000, 15)

def test_no_nulls():
    df = generate_upi_features(n=1000)
    assert df.isnull().sum().sum() == 0

def test_score_bounds():
    df = generate_upi_features(n=1000)
    assert df["txn_regularity_score"].between(0, 1).all()
    assert df["utility_payment_ratio"].between(0, 1).all()

def test_label_merge():
    import numpy as np
    upi = generate_upi_features(n=500)
    labels = pd.Series(np.random.binomial(1, 0.22, 500))
    merged = merge_with_labels(upi, labels)
    assert "default_flag" in merged.columns
    assert len(merged) == 500
"""Unit tests for FraudFeaturePipeline in src/features.py."""

import pytest
import numpy as np
import pandas as pd
from src.features import FraudFeaturePipeline


@pytest.fixture
def sample_data():
    """Create a minimal synthetic transaction batch for testing."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [1000, 7200, 86400, 150000],
        "TransactionAmt": [100.0, 250.50, 49.99, 1200.0],
        "ProductCD": ["W", "H", "C", "W"],
        "card1": [1000, 2000, 1000, 3000],
        "card2": [100, 200, 100, 300],
        "card3": [150, 150, 150, 150],
        "card5": [226, 226, 226, 226],
        "addr1": [315, 315, 204, 315],
        "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", "anonymous.com"],
        "R_emaildomain": ["gmail.com", "hotmail.com", "gmail.com", "missing"],
        "isFraud": [0, 1, 0, 0],
    })


def test_pipeline_fit_transform(sample_data):
    """Test that pipeline fits and produces correct engineered feature columns."""
    pipeline = FraudFeaturePipeline()
    pipeline.fit(sample_data)
    assert pipeline.is_fitted_

    X, y = pipeline.transform(sample_data)
    assert len(X) == len(sample_data)
    assert y is not None
    assert len(y) == len(sample_data)

    # Check key engineered feature columns exist
    expected_cols = [
        "hour_of_day",
        "day_of_week",
        "TransactionAmt_log",
        "TransactionAmt_decimal",
        "TransactionAmt_is_integer",
        "email_domain_match",
        "card1_freq",
        "card_id_freq",
        "amt_diff_card1_mean",
        "amt_ratio_card1_mean",
    ]
    for col in expected_cols:
        assert col in X.columns, f"Missing feature: {col}"


def test_unseen_categories_handling(sample_data):
    """Verify that unseen categories in new transactions are mapped gracefully to -1."""
    pipeline = FraudFeaturePipeline()
    pipeline.fit(sample_data)

    unseen_record = {
        "TransactionID": 99,
        "TransactionDT": 200000,
        "TransactionAmt": 80.0,
        "ProductCD": "NEW_PRODUCT_UNSEEN",
        "card1": 9999,
        "P_emaildomain": "unseen-domain.xyz",
    }
    X_unseen = pipeline.transform_single(unseen_record)
    assert len(X_unseen) == 1
    assert X_unseen.shape[1] == len(pipeline.feature_columns_)
    # Check that unseen ProductCD was mapped to -1
    assert X_unseen["ProductCD"].iloc[0] == -1


def test_pipeline_save_and_load(tmp_path, sample_data):
    """Verify serialization and deserialization of the fitted pipeline."""
    pipeline = FraudFeaturePipeline()
    pipeline.fit(sample_data)

    save_path = tmp_path / "test_pipeline.joblib"
    pipeline.save(save_path)
    assert save_path.is_file()

    reloaded = FraudFeaturePipeline.load(save_path)
    assert reloaded.is_fitted_
    assert reloaded.feature_columns_ == pipeline.feature_columns_

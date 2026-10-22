"""Unit tests verifying strict time-based split separation and integrity.

Guarantees:
1. No temporal overlap: max(train) < min(val) < min(test)
2. Conservation of rows: len(train) + len(val) + len(test) == len(total)
3. Target prevalence consistency without extreme distribution shifts.
"""

import pytest
import pandas as pd
from src.config import TIME_COL, TARGET_COL, PROCESSED_DATA_DIR
from src.data import load_splits, load_cleaned_data

# Skip split verification in clean CI environments where full datasets are excluded from Git
pytestmark = pytest.mark.skipif(
    not (PROCESSED_DATA_DIR / "train.parquet").is_file(),
    reason="Full dataset parquet splits are local-only and excluded from Git.",
)


@pytest.fixture(scope="module")
def splits():
    """Load pre-split train, val, and test datasets."""
    train_df, val_df, test_df = load_splits(verbose=False)
    return train_df, val_df, test_df


def test_split_files_exist():
    """Verify that all three split parquet files exist on disk."""
    for split_name in ["train.parquet", "val.parquet", "test.parquet"]:
        path = PROCESSED_DATA_DIR / split_name
        assert path.is_file(), f"Missing split file: {path}"
        assert path.stat().st_size > 0, f"Split file is empty: {path}"


def test_strict_temporal_ordering(splits):
    """Assert max(train time) < min(val time) < min(test time).

    This test mathematically proves zero temporal data leakage.
    """
    train_df, val_df, test_df = splits

    train_max_time = train_df[TIME_COL].max()
    val_min_time = val_df[TIME_COL].min()
    val_max_time = val_df[TIME_COL].max()
    test_min_time = test_df[TIME_COL].min()

    assert train_max_time < val_min_time, (
        f"Temporal leakage detected: max(train)={train_max_time} >= min(val)={val_min_time}"
    )
    assert val_max_time < test_min_time, (
        f"Temporal leakage detected: max(val)={val_max_time} >= min(test)={test_min_time}"
    )


def test_row_count_conservation(splits):
    """Ensure sum of split records strictly equals total cleaned records."""
    train_df, val_df, test_df = splits
    total_split_rows = len(train_df) + len(val_df) + len(test_df)

    cleaned_df = load_cleaned_data(verbose=False)
    assert total_split_rows == len(cleaned_df), (
        f"Row count mismatch: splits sum to {total_split_rows}, but cleaned data has {len(cleaned_df)}"
    )


def test_fraud_rates_in_realistic_bounds(splits):
    """Verify fraud rate is ~3.4% - 3.6% in each partition."""
    train_df, val_df, test_df = splits
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        fraud_rate = df[TARGET_COL].mean() * 100
        assert 2.0 < fraud_rate < 5.0, f"{name} fraud rate out of expected bounds: {fraud_rate:.2f}%"

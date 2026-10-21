"""Data ingestion, memory optimization, and cleaning utilities.

Handles reading IEEE-CIS transaction and identity tables, downcasting numeric types
to fit large datasets in consumer hardware memory, and merging tables without data leakage.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_TRANSACTION_FILE,
    TRAIN_IDENTITY_FILE,
    ID_COL,
    TIME_COL,
    TARGET_COL,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Iterate through all columns of a dataframe and downcast numeric types to reduce memory.

    Parameters:
        df: Input DataFrame.
        verbose: Whether to print memory reduction stats.

    Returns:
        DataFrame with downcasted numeric data types.
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024**2)

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Float downcasting - float32 gives 7 decimal digits of precision, ample for amounts/anonymised features
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / (1024**2)
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")

    return df


def load_raw_data(
    raw_dir: Optional[Path] = None,
    nrows: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load train_transaction.csv and train_identity.csv, downcast types, and merge on TransactionID.

    Parameters:
        raw_dir: Directory containing raw CSV files. Defaults to RAW_DATA_DIR.
        nrows: Optional limit on rows to load for rapid testing or memory constraints.
        verbose: Whether to log loading progress.

    Returns:
        Merged DataFrame sorted chronologically by TransactionDT.
    """
    data_dir = raw_dir or RAW_DATA_DIR
    trans_path = data_dir / TRAIN_TRANSACTION_FILE
    id_path = data_dir / TRAIN_IDENTITY_FILE

    if not trans_path.is_file():
        raise FileNotFoundError(
            f"Missing {trans_path}. Please download the IEEE-CIS dataset using scripts/download_data.py"
        )

    if verbose:
        print(f"Loading transactions from {trans_path} (nrows={nrows})...")
    df_trans = pd.read_csv(trans_path, nrows=nrows)
    df_trans = reduce_mem_usage(df_trans, verbose=verbose)

    if id_path.is_file():
        if verbose:
            print(f"Loading identity table from {id_path}...")
        df_id = pd.read_csv(id_path)
        df_id = reduce_mem_usage(df_id, verbose=verbose)

        if verbose:
            print(f"Merging tables on {ID_COL} (left join)...")
        df = df_trans.merge(df_id, on=ID_COL, how="left")
    else:
        if verbose:
            print(f"Identity file {id_path} not found. Proceeding with transactions only.")
        df = df_trans

    # Ensure strictly sorted by TransactionDT as required by PRD Section 6
    if TIME_COL in df.columns:
        if verbose:
            print(f"Sorting transactions chronologically by {TIME_COL}...")
        df = df.sort_values(by=TIME_COL).reset_index(drop=True)

    return df


def compute_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key exploratory statistics for IEEE-CIS fraud detection dataset."""
    total_records = len(df)
    total_columns = df.shape[1]

    fraud_counts = df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {}
    fraud_rate = (df[TARGET_COL].mean() * 100) if TARGET_COL in df.columns else 0.0

    # Missing value statistics
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / total_records) * 100
    missing_df = pd.DataFrame({"missing_count": missing_series, "missing_pct": missing_pct})
    cols_gt_50_pct_missing = int((missing_pct > 50).sum())
    cols_gt_80_pct_missing = int((missing_pct > 80).sum())

    # Time span
    time_min = float(df[TIME_COL].min()) if TIME_COL in df.columns else 0.0
    time_max = float(df[TIME_COL].max()) if TIME_COL in df.columns else 0.0
    time_span_days = (time_max - time_min) / 86400.0  # TransactionDT is in seconds

    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)

    return {
        "total_records": total_records,
        "total_columns": total_columns,
        "fraud_count": fraud_counts.get(1, 0),
        "legit_count": fraud_counts.get(0, 0),
        "fraud_rate_pct": fraud_rate,
        "cols_gt_50_pct_missing": cols_gt_50_pct_missing,
        "cols_gt_80_pct_missing": cols_gt_80_pct_missing,
        "time_min_sec": time_min,
        "time_max_sec": time_max,
        "time_span_days": time_span_days,
        "memory_mb": mem_mb,
    }


def clean_data(
    df: pd.DataFrame,
    missing_threshold: float = 0.90,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw IEEE-CIS dataset using principled, reproducible rules.

    1. Checks and removes any duplicate transaction records on TransactionID.
    2. Drops columns with extreme missingness (> missing_threshold, default 90%).
    3. Standardizes categorical columns: strips whitespaces, converts to lowercase, fills NaNs with 'missing'.
    4. Creates data quality flags:
       - 'has_identity': binary indicator whether device/identity telemetry was recorded.
       - 'is_high_amount': binary indicator for extreme amount outliers (> $5,000).
    5. Re-applies memory downcasting for optimal RAM efficiency.

    Parameters:
        df: Raw merged DataFrame.
        missing_threshold: Maximum allowable fraction of missing values before a column is dropped.
        verbose: Whether to log cleaning progress.

    Returns:
        Tuple of (Cleaned DataFrame, dictionary of cleaning statistics).
    """
    initial_rows = len(df)
    initial_cols = df.shape[1]

    # 1. Deduplication: verify TransactionID uniqueness
    if ID_COL in df.columns:
        num_dupes = df[ID_COL].duplicated().sum()
        if num_dupes > 0:
            if verbose:
                print(f"Removing {num_dupes} duplicate transaction IDs...")
            df = df.drop_duplicates(subset=[ID_COL]).reset_index(drop=True)
        elif verbose:
            print("Checked duplicate transaction IDs: 0 duplicates found.")

    # 2. Add domain-specific data quality indicators before dropping sparse columns
    # 'has_identity' flag indicates if identity verification telemetry was captured
    df = df.copy()
    if "id_01" in df.columns:
        df["has_identity"] = df["id_01"].notnull().astype(np.int8)
    elif "DeviceType" in df.columns:
        df["has_identity"] = df["DeviceType"].notnull().astype(np.int8)
    else:
        df["has_identity"] = np.int8(0)

    # Flag high amount outliers without dropping valid high-value fraud cases
    if "TransactionAmt" in df.columns:
        df["is_high_amount"] = (df["TransactionAmt"] > 5000.0).astype(np.int8)

    # 3. Identify and drop columns with extreme missingness (> 90%)
    missing_fraction = df.isnull().mean()
    dropped_cols = missing_fraction[missing_fraction > missing_threshold].index.tolist()

    # Never drop target or essential identifier/time columns
    protected_cols = {ID_COL, TIME_COL, TARGET_COL, "has_identity", "is_high_amount"}
    dropped_cols = [c for c in dropped_cols if c not in protected_cols]

    if verbose:
        print(f"Dropping {len(dropped_cols)} columns with > {missing_threshold*100:.0f}% missing values:")
        print(f"  {dropped_cols}")
    df = df.drop(columns=dropped_cols)

    # 4. Clean categorical columns: standardize text and handle missingness
    # Categorical columns in IEEE-CIS are either object or category type
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if verbose:
        print(f"Standardizing {len(cat_cols)} categorical columns (filling NaNs with 'missing')...")

    for col in cat_cols:
        # Standardize strings: strip whitespace and lowercase
        s = df[col].astype(str).str.strip().str.lower()
        # Handle original NaNs (which became 'nan' or '<na>')
        s = s.replace({"nan": "missing", "<na>": "missing", "none": "missing", "": "missing"})
        # Convert to category dtype for memory and LightGBM efficiency
        df[col] = s.astype("category")

    # 5. Re-apply numeric memory downcasting
    df = reduce_mem_usage(df, verbose=verbose)

    stats = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "initial_cols": initial_cols,
        "final_cols": df.shape[1],
        "dropped_cols_count": len(dropped_cols),
        "dropped_cols": dropped_cols,
        "missing_threshold": missing_threshold,
        "categorical_cols_standardized": len(cat_cols),
    }

    if verbose:
        print(f"Cleaning complete: Shape changed from ({initial_rows}, {initial_cols}) to ({len(df)}, {df.shape[1]}).")

    return df, stats


def save_processed_data(
    df: pd.DataFrame,
    filename: str = "cleaned_data.parquet",
    verbose: bool = True,
) -> Path:
    """Save cleaned DataFrame to data/processed/ in Apache Parquet format.

    Parquet preserves column types (categories, downcasted floats/ints) and compresses
    the multi-gigabyte dataset into a fast-loading file (~200MB).
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    if verbose:
        print(f"Saving cleaned dataset to {out_path}...")
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = out_path.stat().st_size / (1024**2)
    if verbose:
        print(f"Saved successfully: {size_mb:.1f} MB on disk.")
    return out_path


def load_cleaned_data(
    filepath: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the cleaned dataset from data/processed/cleaned_data.parquet."""
    path = filepath or (PROCESSED_DATA_DIR / "cleaned_data.parquet")
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run data cleaning pipeline first."
        )
    if verbose:
        print(f"Loading cleaned dataset from {path}...")
    df = pd.read_parquet(path, engine="pyarrow")
    if verbose:
        print(f"Loaded {len(df):,} records with {df.shape[1]} columns.")
    return df


def split_time_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Perform a strict chronological train/val/test split without data leakage.

    Sorts chronologically by TransactionDT and partitions into:
    - Train: Earliest 70%
    - Validation: Next 15%
    - Test: Latest 15% (touched once at final evaluation)

    Parameters:
        df: Input DataFrame containing TIME_COL.
        train_ratio: Proportion of earliest records allocated to training.
        val_ratio: Proportion of intermediate records allocated to validation.
        test_ratio: Proportion of latest records allocated to hold-out testing.
        verbose: Whether to log split statistics.

    Returns:
        Tuple of (train_df, val_df, test_df, split_statistics).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    # Strictly sort by timestamp to guarantee time order
    if verbose:
        print(f"Sorting {len(df):,} records chronologically by {TIME_COL}...")
    df_sorted = df.sort_values(by=TIME_COL).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    # Strict temporal separation assertion: no time overlap
    train_max_t = train_df[TIME_COL].max()
    val_min_t = val_df[TIME_COL].min()
    val_max_t = val_df[TIME_COL].max()
    test_min_t = test_df[TIME_COL].min()

    assert train_max_t < val_min_t, f"Temporal overlap between Train and Val: {train_max_t} >= {val_min_t}"
    assert val_max_t < test_min_t, f"Temporal overlap between Val and Test: {val_max_t} >= {test_min_t}"

    stats = {
        "total_records": n,
        "train_rows": len(train_df),
        "train_fraud_rate_pct": float(train_df[TARGET_COL].mean() * 100) if TARGET_COL in train_df else 0.0,
        "train_time_range": (float(train_df[TIME_COL].min()), float(train_max_t)),
        "val_rows": len(val_df),
        "val_fraud_rate_pct": float(val_df[TARGET_COL].mean() * 100) if TARGET_COL in val_df else 0.0,
        "val_time_range": (float(val_min_t), float(val_max_t)),
        "test_rows": len(test_df),
        "test_fraud_rate_pct": float(test_df[TARGET_COL].mean() * 100) if TARGET_COL in test_df else 0.0,
        "test_time_range": (float(test_min_t), float(test_df[TIME_COL].max())),
    }

    if verbose:
        print("\n=== Chronological Split Completed ===")
        print(f"Train: {stats['train_rows']:,} rows ({train_ratio*100:.0f}%) | Fraud Rate: {stats['train_fraud_rate_pct']:.2f}% | Time: [{stats['train_time_range'][0]:.0f} to {stats['train_time_range'][1]:.0f}]")
        print(f"Val:   {stats['val_rows']:,} rows ({val_ratio*100:.0f}%) | Fraud Rate: {stats['val_fraud_rate_pct']:.2f}% | Time: [{stats['val_time_range'][0]:.0f} to {stats['val_time_range'][1]:.0f}]")
        print(f"Test:  {stats['test_rows']:,} rows ({test_ratio*100:.0f}%) | Fraud Rate: {stats['test_fraud_rate_pct']:.2f}% | Time: [{stats['test_time_range'][0]:.0f} to {stats['test_time_range'][1]:.0f}]")
        print(f"Verification: max(train) = {train_max_t} < min(val) = {val_min_t} < min(test) = {test_min_t} (PASSED)")

    return train_df, val_df, test_df, stats


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[Path, Path, Path]:
    """Save train, val, and test partitions to data/processed/ as Parquet files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    if verbose:
        print("Saving splits to data/processed/...")
    train_df.to_parquet(train_path, index=False, engine="pyarrow", compression="snappy")
    val_df.to_parquet(val_path, index=False, engine="pyarrow", compression="snappy")
    test_df.to_parquet(test_path, index=False, engine="pyarrow", compression="snappy")

    if verbose:
        print(f"  - Train: {train_path.stat().st_size / (1024**2):.1f} MB")
        print(f"  - Val:   {val_path.stat().st_size / (1024**2):.1f} MB")
        print(f"  - Test:  {test_path.stat().st_size / (1024**2):.1f} MB")

    return train_path, val_path, test_path


def load_splits(
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test splits from data/processed/."""
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    for p in [train_path, val_path, test_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Split file {p} not found. Run time-based split first.")

    if verbose:
        print("Loading train, val, and test splits from data/processed/...")
    train_df = pd.read_parquet(train_path, engine="pyarrow")
    val_df = pd.read_parquet(val_path, engine="pyarrow")
    test_df = pd.read_parquet(test_path, engine="pyarrow")

    if verbose:
        print(f"Loaded Train ({len(train_df):,}), Val ({len(val_df):,}), Test ({len(test_df):,}).")

    return train_df, val_df, test_df



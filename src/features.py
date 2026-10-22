"""Feature engineering pipeline for transaction fraud detection.

Generates temporal, monetary, card aggregation, and frequency features.
Crucially enforces zero data leakage: all encoders, aggregations, and frequency maps
are strictly fitted on the training split only, then applied to validation, test,
and live API inference requests.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    ID_COL,
    TIME_COL,
    TARGET_COL,
)


class FraudFeaturePipeline:
    """Stateful feature engineering pipeline.

    Fits aggregations, frequency tables, and categorical encoders exclusively on train data.
    Provides reproducible transformation for Train, Validation, Test, and single API records.
    """

    def __init__(self):
        # Learned statistics from train
        self.card1_counts_: Dict[Any, int] = {}
        self.card_id_counts_: Dict[str, int] = {}
        self.addr1_counts_: Dict[Any, int] = {}
        self.p_email_counts_: Dict[str, int] = {}

        self.card1_amt_mean_: Dict[Any, float] = {}
        self.card1_amt_std_: Dict[Any, float] = {}
        self.global_amt_mean_: float = 0.0
        self.global_amt_std_: float = 1.0

        self.categorical_maps_: Dict[str, Dict[str, int]] = {}
        self.categorical_cols_: List[str] = []
        self.feature_columns_: List[str] = []
        self.is_fitted_: bool = False

    def _create_card_id(self, df: pd.DataFrame) -> pd.Series:
        """Construct synthetic card entity identifier combining bank, issuer, and country codes."""
        c1 = df["card1"].fillna(-1).astype(str) if "card1" in df.columns else ""
        c2 = df["card2"].fillna(-1).astype(str) if "card2" in df.columns else ""
        c3 = df["card3"].fillna(-1).astype(str) if "card3" in df.columns else ""
        c5 = df["card5"].fillna(-1).astype(str) if "card5" in df.columns else ""
        return c1 + "_" + c2 + "_" + c3 + "_" + c5

    def fit(self, train_df: pd.DataFrame) -> "FraudFeaturePipeline":
        """Fit feature encoders, frequency counts, and aggregations on train_df ONLY.

        Parameters:
            train_df: Chronological training partition.

        Returns:
            Fitted pipeline instance.
        """
        # 1. Global amount statistics
        self.global_amt_mean_ = float(train_df["TransactionAmt"].mean())
        self.global_amt_std_ = float(train_df["TransactionAmt"].std() or 1.0)

        # 2. Card entity identifier for train
        card_id_series = self._create_card_id(train_df)

        # 3. Frequency distributions (fit on train only)
        if "card1" in train_df.columns:
            self.card1_counts_ = train_df["card1"].value_counts().to_dict()
        self.card_id_counts_ = card_id_series.value_counts().to_dict()

        if "addr1" in train_df.columns:
            self.addr1_counts_ = train_df["addr1"].value_counts().to_dict()

        if "P_emaildomain" in train_df.columns:
            self.p_email_counts_ = train_df["P_emaildomain"].astype(str).value_counts().to_dict()

        # 4. Amount group statistics by card1
        if "card1" in train_df.columns and "TransactionAmt" in train_df.columns:
            card1_grouped = train_df.groupby("card1")["TransactionAmt"]
            self.card1_amt_mean_ = card1_grouped.mean().to_dict()
            self.card1_amt_std_ = card1_grouped.std().fillna(1.0).to_dict()

        # 5. Categorical mappings (Label / Ordinal encoding, unseen categories -> -1)
        cat_cols = train_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        # Exclude IDs or target if present
        cat_cols = [c for c in cat_cols if c not in {ID_COL, TARGET_COL}]
        self.categorical_cols_ = cat_cols

        for col in cat_cols:
            unique_vals = train_df[col].dropna().unique().tolist()
            # Map each category to an integer (0, 1, 2, ...)
            self.categorical_maps_[col] = {str(val).lower().strip(): idx for idx, val in enumerate(unique_vals)}

        self.is_fitted_ = True

        # Generate sample transformation to lock feature_columns_
        sample_transformed = self._transform_internal(train_df.head(10))
        # Drop ID, time, and target from model feature set
        exclude_cols = {ID_COL, TIME_COL, TARGET_COL}
        self.feature_columns_ = [c for c in sample_transformed.columns if c not in exclude_cols]

        return self

    def _transform_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal transformation applying feature engineering rules."""
        out = df.copy()

        # A. Temporal Features (from TransactionDT in seconds)
        if TIME_COL in out.columns:
            out["hour_of_day"] = ((out[TIME_COL] // 3600) % 24).astype(np.int8)
            out["day_of_week"] = ((out[TIME_COL] // 86400) % 7).astype(np.int8)
        else:
            out["hour_of_day"] = np.int8(12)
            out["day_of_week"] = np.int8(3)

        # B. Monetary Features (from TransactionAmt)
        if "TransactionAmt" in out.columns:
            amt = out["TransactionAmt"].fillna(self.global_amt_mean_)
            out["TransactionAmt_log"] = np.log1p(amt.clip(lower=0)).astype(np.float32)
            out["TransactionAmt_decimal"] = (amt - np.floor(amt)).astype(np.float32)
            out["TransactionAmt_is_integer"] = (out["TransactionAmt_decimal"] == 0).astype(np.int8)

        # C. Entity & Domain Interaction Features
        card_id_series = self._create_card_id(out)
        if "P_emaildomain" in out.columns and "R_emaildomain" in out.columns:
            p_email = out["P_emaildomain"].astype(str).str.lower().str.strip()
            r_email = out["R_emaildomain"].astype(str).str.lower().str.strip()
            out["email_domain_match"] = (
                (p_email == r_email) & (p_email != "missing")
            ).astype(np.int8)
        else:
            out["email_domain_match"] = np.int8(0)

        # D. Frequency Features (Mapped strictly from training statistics)
        if "card1" in out.columns:
            out["card1_freq"] = out["card1"].map(self.card1_counts_).fillna(1).astype(np.int32)
        out["card_id_freq"] = card_id_series.map(self.card_id_counts_).fillna(1).astype(np.int32)

        if "addr1" in out.columns:
            out["addr1_freq"] = out["addr1"].map(self.addr1_counts_).fillna(1).astype(np.int32)

        if "P_emaildomain" in out.columns:
            p_str = out["P_emaildomain"].astype(str).str.lower().str.strip()
            out["P_emaildomain_freq"] = p_str.map(self.p_email_counts_).fillna(1).astype(np.int32)

        # E. Monetary Relative Aggregations (Deviation from card historical behavior)
        if "card1" in out.columns and "TransactionAmt" in out.columns:
            card1_mean = out["card1"].map(self.card1_amt_mean_).fillna(self.global_amt_mean_)
            card1_std = out["card1"].map(self.card1_amt_std_).fillna(self.global_amt_std_).replace(0, 1.0)

            out["amt_diff_card1_mean"] = (out["TransactionAmt"] - card1_mean).astype(np.float32)
            out["amt_ratio_card1_mean"] = (out["TransactionAmt"] / (card1_mean + 1e-4)).astype(np.float32)
            out["amt_zscore_card1"] = ((out["TransactionAmt"] - card1_mean) / card1_std).astype(np.float32)

        # F. Categorical Encoding (Label / Ordinal encoding; unseen -> -1)
        for col in self.categorical_cols_:
            if col in out.columns:
                cat_map = self.categorical_maps_.get(col, {})
                s = out[col].astype(str).str.lower().str.strip()
                out[col] = s.map(cat_map).fillna(-1).astype(np.int16)

        return out

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform a dataset into engineered feature matrix X and target y (if present).

        Parameters:
            df: Input DataFrame (Train, Val, Test, or batch inference).

        Returns:
            Tuple of (X DataFrame with selected feature columns, y Series or None).
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted yet. Call fit() with training data first.")

        transformed = self._transform_internal(df)

        # Efficiently reindex to expected feature columns (pads missing with 0 for single inference)
        X = transformed.reindex(columns=self.feature_columns_, fill_value=0).copy()
        y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None

        return X, y

    def transform_single(self, record: Dict[str, Any]) -> pd.DataFrame:
        """Transform a single transaction dictionary for production API serving."""
        single_df = pd.DataFrame([record])
        X, _ = self.transform(single_df)
        return X

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist fitted feature pipeline to disk using joblib."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = filepath or (MODELS_DIR / "feature_pipeline.joblib")
        joblib.dump(self, path)
        print(f"Feature pipeline saved to {path} ({path.stat().st_size / 1024:.1f} KB)")
        return path

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "FraudFeaturePipeline":
        """Load fitted pipeline from disk."""
        path = Path(filepath) if filepath else (MODELS_DIR / "feature_pipeline.joblib")
        if not path.is_file():
            raise FileNotFoundError(f"Fitted feature pipeline not found at {path}")
        pipeline = joblib.load(path)
        return pipeline


def run_feature_pipeline(verbose: bool = True) -> Tuple[Path, Path, Path, Path]:
    """Execute feature engineering pipeline on train, val, and test splits and save engineered matrices.

    Returns:
        Tuple of (train_feat_path, val_feat_path, test_feat_path, pipeline_path).
    """
    from src.data import load_splits

    if verbose:
        print("Loading train, val, and test splits...")
    train_df, val_df, test_df = load_splits(verbose=verbose)

    pipeline = FraudFeaturePipeline()
    if verbose:
        print(f"Fitting feature pipeline on train split ({len(train_df):,} rows)...")
    pipeline.fit(train_df)

    # Save pipeline artifact for API reuse
    pipeline_path = pipeline.save()

    if verbose:
        print(f"Transforming train split ({len(train_df):,} rows)...")
    X_train, y_train = pipeline.transform(train_df)
    train_feat = X_train.copy()
    train_feat[TARGET_COL] = y_train.values

    if verbose:
        print(f"Transforming validation split ({len(val_df):,} rows)...")
    X_val, y_val = pipeline.transform(val_df)
    val_feat = X_val.copy()
    val_feat[TARGET_COL] = y_val.values

    if verbose:
        print(f"Transforming test split ({len(test_df):,} rows)...")
    X_test, y_test = pipeline.transform(test_df)
    test_feat = X_test.copy()
    test_feat[TARGET_COL] = y_test.values

    # Save engineered feature parquets
    train_feat_path = PROCESSED_DATA_DIR / "train_features.parquet"
    val_feat_path = PROCESSED_DATA_DIR / "val_features.parquet"
    test_feat_path = PROCESSED_DATA_DIR / "test_features.parquet"

    if verbose:
        print(f"Saving engineered feature sets to {PROCESSED_DATA_DIR}...")
    train_feat.to_parquet(train_feat_path, index=False, engine="pyarrow", compression="snappy")
    val_feat.to_parquet(val_feat_path, index=False, engine="pyarrow", compression="snappy")
    test_feat.to_parquet(test_feat_path, index=False, engine="pyarrow", compression="snappy")

    if verbose:
        print(f"  - Train Features: {train_feat_path.stat().st_size / (1024**2):.1f} MB | {train_feat.shape}")
        print(f"  - Val Features:   {val_feat_path.stat().st_size / (1024**2):.1f} MB | {val_feat.shape}")
        print(f"  - Test Features:  {test_feat_path.stat().st_size / (1024**2):.1f} MB | {test_feat.shape}")
        print(f"Total features engineered: {len(pipeline.feature_columns_)}")

    return train_feat_path, val_feat_path, test_feat_path, pipeline_path

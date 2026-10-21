"""Model training pipeline.

Implements baseline Logistic Regression model with class weights and scaled features,
evaluating strictly on the chronological validation partition.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COL,
)
from src.evaluate import (
    evaluate_predictions,
    save_evaluation_report,
    plot_confusion_matrix_heatmap,
)

# Core simple features for interpretable baseline model
BASELINE_FEATURES: List[str] = [
    "TransactionAmt",
    "TransactionAmt_log",
    "TransactionAmt_decimal",
    "TransactionAmt_is_integer",
    "hour_of_day",
    "day_of_week",
    "has_identity",
    "is_high_amount",
    "card1_freq",
    "card_id_freq",
    "amt_diff_card1_mean",
    "amt_ratio_card1_mean",
    "C1",
    "C2",
    "C5",
    "C6",
    "C9",
    "C11",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
]


def train_baseline_model(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    features: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train Logistic Regression baseline with balanced class weights on simple scaled features.

    Parameters:
        train_path: Path to train_features.parquet.
        val_path: Path to val_features.parquet.
        features: Subset of features to train on. Defaults to BASELINE_FEATURES.
        verbose: Whether to log training progress.

    Returns:
        Tuple of (fitted sklearn Pipeline, evaluation metrics dictionary).
    """
    t_path = train_path or (PROCESSED_DATA_DIR / "train_features.parquet")
    v_path = val_path or (PROCESSED_DATA_DIR / "val_features.parquet")

    if not t_path.is_file() or not v_path.is_file():
        raise FileNotFoundError(
            f"Feature sets not found in {PROCESSED_DATA_DIR}. Run src/features.py first."
        )

    if verbose:
        print(f"Loading training features from {t_path}...")
    train_df = pd.read_parquet(t_path)
    if verbose:
        print(f"Loading validation features from {v_path}...")
    val_df = pd.read_parquet(v_path)

    # Filter features to those available in dataset
    candidate_features = features or BASELINE_FEATURES
    selected_features = [f for f in candidate_features if f in train_df.columns]

    if verbose:
        print(f"Selected {len(selected_features)} simple baseline features:")
        print(f"  {selected_features}")

    X_train = train_df[selected_features]
    y_train = train_df[TARGET_COL]

    X_val = val_df[selected_features]
    y_val = val_df[TARGET_COL]

    # Build Pipeline: Median Imputation -> Standard Scaling -> Logistic Regression
    baseline_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])

    if verbose:
        print("Fitting Baseline Logistic Regression (class_weight='balanced')...")
    baseline_pipeline.fit(X_train, y_train)

    if verbose:
        print("Generating predictions on validation set...")
    val_pred_proba = baseline_pipeline.predict_proba(X_val)[:, 1]

    # Evaluate predictions
    metrics = evaluate_predictions(y_val, val_pred_proba)

    # Persist metrics and visual artifacts
    save_evaluation_report(metrics, filename="baseline_metrics.json", verbose=verbose)
    plot_confusion_matrix_heatmap(
        metrics["confusion_matrix"],
        title="Baseline (Logistic Regression) Confusion Matrix",
        filename="baseline_confusion_matrix.png",
    )

    # Save trained baseline model pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "baseline_model.joblib"
    joblib.dump(baseline_pipeline, model_path)
    if verbose:
        print(f"Saved baseline model to {model_path}")
        print("\n=== Baseline Validation Results ===")
        print(f"PR-AUC (Average Precision): {metrics['pr_auc']:.4f} (Benchmark to beat)")
        print(f"ROC-AUC:                    {metrics['roc_auc']:.4f}")
        print(f"F1-Score (threshold=0.5):   {metrics['f1_score']:.4f}")
        print(f"Recall:                     {metrics['recall']:.4f}")
        print(f"Precision:                  {metrics['precision']:.4f}")
        print(f"Total Model Cost:           ${metrics['business_costs']['total_model_cost']:,.2f}")
        print(f"Cost Saved vs. Do-Nothing:  ${metrics['business_costs']['cost_saved']:,.2f} ({metrics['business_costs']['cost_reduction_pct']:.1f}% reduction)")

    return baseline_pipeline, metrics


if __name__ == "__main__":
    train_baseline_model()

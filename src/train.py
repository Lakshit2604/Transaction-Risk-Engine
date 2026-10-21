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


def train_and_compare_models(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    """Train and compare LightGBM variants against baseline across imbalance handling techniques.

    Compares:
    1. Baseline Logistic Regression (simple scaled features, class_weight='balanced')
    2. LightGBM with balanced class weights
    3. LightGBM with scale_pos_weight=5.0
    4. LightGBM with random undersampling (1:10 ratio on train only)
    5. LightGBM Tuned (Unweighted with optimized capacity: 250 trees, 95 leaves, 0.8 feature fraction)

    Outputs:
    - reports/model_comparison.csv
    - reports/model_comparison_prauc.png
    - models/best_lightgbm.joblib (winning model)
    """
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    from src.config import REPORTS_DIR

    t_path = train_path or (PROCESSED_DATA_DIR / "train_features.parquet")
    v_path = val_path or (PROCESSED_DATA_DIR / "val_features.parquet")

    if verbose:
        print("Loading train and validation feature sets...")
    train_df = pd.read_parquet(t_path)
    val_df = pd.read_parquet(v_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]

    # Ensure baseline metrics exist or train baseline
    baseline_model, baseline_metrics = train_baseline_model(
        train_path=t_path, val_path=v_path, verbose=False
    )

    results: List[Dict[str, Any]] = []

    # Record Baseline
    results.append({
        "model": "Logistic Regression",
        "imbalance_strategy": "class_weight='balanced' (26 core features)",
        "pr_auc": baseline_metrics["pr_auc"],
        "roc_auc": baseline_metrics["roc_auc"],
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
        "f1_score": baseline_metrics["f1_score"],
        "total_cost": baseline_metrics["business_costs"]["total_model_cost"],
        "cost_reduction_pct": baseline_metrics["business_costs"]["cost_reduction_pct"],
    })

    # Imbalance experiments for LightGBM
    lgb_experiments = [
        {
            "name": "LightGBM (Balanced Weights)",
            "strategy": "class_weight='balanced'",
            "params": {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 63, "class_weight": "balanced"},
            "undersample": False,
        },
        {
            "name": "LightGBM (scale_pos_weight=5.0)",
            "strategy": "scale_pos_weight=5.0",
            "params": {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 63, "scale_pos_weight": 5.0},
            "undersample": False,
        },
        {
            "name": "LightGBM (Undersampling 1:10)",
            "strategy": "Random majority undersampling (1:10 ratio)",
            "params": {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 63},
            "undersample": True,
        },
        {
            "name": "LightGBM (Tuned - Unweighted)",
            "strategy": "Natural prevalence (250 trees, 95 leaves, colsample 0.8)",
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "num_leaves": 95,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
            },
            "undersample": False,
        },
    ]

    trained_models = {}

    for exp in lgb_experiments:
        name = exp["name"]
        if verbose:
            print(f"\nTraining {name} [{exp['strategy']}]...")

        if exp["undersample"]:
            # Undersample majority class on train set ONLY (never touch validation!)
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            rng = np.random.RandomState(RANDOM_SEED)
            sampled_neg = rng.choice(neg_idx, size=len(pos_idx) * 10, replace=False)
            sub_idx = np.concatenate([pos_idx, sampled_neg])
            rng.shuffle(sub_idx)

            X_tr_exp = X_train.iloc[sub_idx]
            y_tr_exp = y_train.iloc[sub_idx]
        else:
            X_tr_exp = X_train
            y_tr_exp = y_train

        clf = lgb.LGBMClassifier(
            **exp["params"],
            random_state=RANDOM_SEED,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_tr_exp, y_tr_exp)
        probs = clf.predict_proba(X_val)[:, 1]

        m = evaluate_predictions(y_val, probs)
        trained_models[name] = (clf, m, probs)

        results.append({
            "model": name,
            "imbalance_strategy": exp["strategy"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "total_cost": m["business_costs"]["total_model_cost"],
            "cost_reduction_pct": m["business_costs"]["cost_reduction_pct"],
        })

        if verbose:
            print(f"  -> PR-AUC: {m['pr_auc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | Cost Reduction: {m['business_costs']['cost_reduction_pct']:.1f}%")

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by="pr_auc", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nSaved model comparison table to {csv_path}")

    # Identify best model by validation PR-AUC
    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model"]
    best_model, best_m, best_probs = trained_models[best_model_name]

    # Save winning model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_lightgbm.joblib"
    joblib.dump(best_model, best_model_path)
    if verbose:
        print(f"Saved winning model ({best_model_name}) to {best_model_path}")

    # Generate Model Comparison Visualization Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#2ecc71" if m == best_model_name else "#3498db" if "LightGBM" in m else "#95a5a6" for m in comparison_df["model"]]
    bars = ax.barh(comparison_df["model"], comparison_df["pr_auc"], color=bar_colors, edgecolor="black", height=0.6)
    ax.set_xlabel("Validation PR-AUC (Average Precision)", fontweight="bold")
    ax.set_title("Model Comparison on Validation Set (Baseline vs. LightGBM Variants)", fontweight="bold", pad=12)
    ax.set_xlim(0, max(comparison_df["pr_auc"]) * 1.18)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2.0, f"{w:.4f}", va="center", fontweight="bold")

    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = REPORTS_DIR / "model_comparison_prauc.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    if verbose:
        print(f"Saved comparison plot to {plot_path}")
        print("\n" + "=" * 75)
        print("MODEL COMPARISON SUMMARY (Ranked by Validation PR-AUC):")
        print("=" * 75)
        print(comparison_df.to_string(index=False))
        print("=" * 75)

    return best_model, comparison_df


if __name__ == "__main__":
    train_and_compare_models()


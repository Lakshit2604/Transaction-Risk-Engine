"""Unit tests for baseline model training and evaluation."""

import json
from pathlib import Path
import pytest
from src.config import MODELS_DIR, REPORTS_DIR


def test_baseline_artifacts_exist():
    """Verify that baseline model binary and evaluation JSON exist on disk."""
    model_path = MODELS_DIR / "baseline_model.joblib"
    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    plot_path = REPORTS_DIR / "baseline_confusion_matrix.png"

    assert model_path.is_file(), f"Missing baseline model: {model_path}"
    assert metrics_path.is_file(), f"Missing metrics report: {metrics_path}"
    assert plot_path.is_file(), f"Missing confusion matrix plot: {plot_path}"


def test_baseline_metrics_content():
    """Ensure baseline metrics report contains all required PRD metrics."""
    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Check required metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert "recall_at_80pct_precision" in metrics
    assert "confusion_matrix" in metrics
    assert "business_costs" in metrics

    # Sanity bounds
    assert 0.10 <= metrics["pr_auc"] <= 0.40, f"Unexpected baseline PR-AUC: {metrics['pr_auc']}"
    assert 0.70 <= metrics["roc_auc"] <= 0.90, f"Unexpected baseline ROC-AUC: {metrics['roc_auc']}"
    assert metrics["confusion_matrix"]["true_positives"] > 1000

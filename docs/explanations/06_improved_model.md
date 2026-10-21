# Step 06: Improved Model and Imbalance Handling

## What was done (plain English, 5-8 lines)
Trained LightGBM gradient boosted decision trees across all 434 engineered features on the 70% chronological training split.
Benchmarked three distinct class imbalance mitigation strategies: balanced class weighting (`class_weight='balanced'`), positive gradient scaling (`scale_pos_weight=5.0`), and random majority undersampling (1:10 ratio on train only).
Conducted time-aware hyperparameter tuning over tree complexity, learning rate, and feature subsampling evaluated exclusively on the validation set.
Generated a comprehensive comparative report saved to `reports/model_comparison.csv` and visualization chart `reports/model_comparison_prauc.png`.
Identified the top-performing configuration: Tuned LightGBM (250 estimators, 95 leaves, 0.8 feature fraction, 0.8 bagging fraction).
Serialized the winning model to `models/best_lightgbm.joblib` achieving a **PR-AUC of 0.6134**—an increase of **+0.4147 absolute (+208.7% relative improvement)** over the baseline.

## Why this step matters
While linear models are fast, transaction fraud involves intricate non-linear decision surfaces, feature interactions (e.g. card frequency conditional on late-night hours and specific dollar amounts), and high-dimensional categorical structures that linear classifiers cannot separate. LightGBM excels on tabular data with histogram-based splitting and native NaN routing. Furthermore, empirical comparison of class imbalance strategies reveals a crucial production insight: heavily penalizing majority instances (like `class_weight='balanced'`) artificially degrades precision at high thresholds, whereas training on natural prevalence with feature subsampling yields superior probability ranking (PR-AUC).

## Files created / changed (path + one-line purpose each)
- `src/train.py`: Added `train_and_compare_models()` executing comparative training across imbalance handling variants.
- `reports/model_comparison.csv`: Tabular benchmark comparing Baseline vs. all 4 LightGBM variants on PR-AUC, ROC-AUC, F1, and cost.
- `reports/model_comparison_prauc.png`: Horizontal bar chart ranking all model architectures by validation PR-AUC.
- `models/best_lightgbm.joblib`: Serialized winning LightGBM model artifact (gitignored).
- `tests/test_model.py`: Added unit test asserting LightGBM outperforms baseline PR-AUC by >0.30.
- `docs/explanations/06_improved_model.md`: Step documentation and interview prep guide.

## Key decisions and alternatives rejected
- **Picking Winner by PR-AUC vs. Raw F1 at Default 0.5 Threshold:** At the default 0.5 cutoff, heavily weighted models can show a deceptively balanced F1, but their overall ranking ability is compromised. PR-AUC evaluates the entire precision-recall curve independently of any arbitrary threshold. The tuned unweighted LightGBM achieved the highest PR-AUC (**0.6134** vs. 0.5385 for balanced weights), providing the cleanest posterior probability ranking for threshold optimization in Step 07.
- **Natural Prevalence + Threshold Tuning vs. Aggressive Resampling:** Synthetic oversampling (like SMOTE) in high-dimensional tabular data with mixed categories often generates unrealistic synthetic records and causes severe calibration drift. Training LightGBM on true natural prevalence preserves calibrated probability ordering, allowing business costs to dictate the optimal operating cutoff downstream.
- **Undersampling on Train Set Only vs. Resampling Validation:** When testing undersampling, we downsampled genuine transactions strictly on `X_train`. The validation set was preserved untouched at its true real-world 3.43% fraud prevalence, ensuring evaluation was strictly unbiased.

## Results / outputs (numbers, plots, observations)
Validation Set Performance Comparison (`reports/model_comparison.csv`):

| Model Architecture | Imbalance Handling Strategy | Validation PR-AUC | Validation ROC-AUC | Precision (@0.5) | Recall (@0.5) | F1-Score (@0.5) | Cost Reduction % |
|---|---|---|---|---|---|---|---|
| **LightGBM (Tuned - Unweighted)** | **Natural prevalence (250 trees, 95 leaves, colsample 0.8)** | **0.6134** | **0.9305** | **87.07%** | **37.64%** | **0.5256** | 37.1% |
| LightGBM (`scale_pos_weight=5.0`) | Positive penalty scaling | 0.5703 | 0.9239 | 62.26% | 48.09% | 0.5427 | 45.2% |
| LightGBM (Undersampling 1:10) | Random majority downsampling on train | 0.5650 | 0.9212 | 71.15% | 42.41% | 0.5314 | 40.7% |
| LightGBM (Balanced Weights) | `class_weight='balanced'` | 0.5385 | 0.9196 | 24.28% | 71.30% | 0.3622 | 49.1% |
| Logistic Regression (Baseline) | `class_weight='balanced'` (26 core features) | 0.1987 | 0.7919 | 9.56% | 69.53% | 0.1682 | 3.8% |

**Key Takeaways:**
1. **LightGBM Tuned delivers a +208.7% relative improvement** in PR-AUC over the Logistic Regression baseline (0.1987 -> 0.6134).
2. ROC-AUC surged from **0.7919 to 0.9305**.
3. At the standard 0.5 threshold, LightGBM attains **87.07% precision** (vs. 9.56% for baseline).

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: Why did unweighted LightGBM achieve a higher PR-AUC (0.6134) than `class_weight='balanced'` (0.5385)?**
  *A:* In gradient boosting, setting `class_weight='balanced'` multiplies the positive gradient by ~27x. This forces the tree-building algorithm to aggressively branch on noisy patterns just to isolate individual fraud instances, causing overfitting on false alarms and degrading precision at higher thresholds. Unweighted LightGBM with tree regularization (feature subsampling and leaf limits) preserves natural probability calibration and maximizes ranking capability across all thresholds.
- **Q: Why did you test random undersampling, and what was the trade-off?**
  *A:* Undersampling speeds up training significantly (training on ~160k samples instead of ~413k) while improving recall. However, discarding 90% of genuine transactions deprives the model of learning the diverse distribution of normal customer behavior, leading to a slight drop in PR-AUC (0.5650 vs 0.6134).
- **Q: Why did you tune hyperparameters on the chronological validation set instead of 5-fold cross-validation?**
  *A:* Cross-validation across randomly shuffled folds would leak future transaction patterns into past folds. Using the out-of-time validation partition guarantees that our hyperparameter tuning (selecting 95 leaves, 0.05 learning rate, and 0.8 feature subsampling) simulates real-world temporal generalization.
- **Q: How does LightGBM handle missing values compared to Logistic Regression?**
  *A:* Logistic Regression requires explicit imputation (e.g., median filling), which can introduce bias. LightGBM handles missing values natively during split finding: for each feature, it evaluates placing all NaNs in the left child vs. the right child and selects whichever child yields the greater reduction in loss.

## Limitations / what I would improve
- We could evaluate an ensemble / stacking blend combining LightGBM, CatBoost, and XGBoost to capture complementary tree split heuristics.
- Bayesian optimization (via Optuna) could be used for a deeper multi-parameter search if training resources and compute time permitted.

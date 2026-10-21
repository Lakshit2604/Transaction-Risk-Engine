# Step 05: Baseline Model 🧠

## What was done (plain English, 5-8 lines)
Constructed an interpretable baseline modeling pipeline in `src/train.py` using `LogisticRegression` with `class_weight='balanced'`.
Trained on 26 curated, simple transactional features (monetary patterns, time features, card frequencies, and key C/D counters).
Applied median imputation and standard scaling fit strictly on the training partition within a unified scikit-learn pipeline.
Evaluated predictions on the chronological validation set (88,581 transactions) without touching the holdout test set.
Recorded a baseline benchmark PR-AUC of **0.1987** (Average Precision) and ROC-AUC of **0.7919**.
Generated confusion matrix heatmap and financial cost metrics, saving all results to `reports/baseline_metrics.json` and `reports/baseline_confusion_matrix.png`.
Created automated tests in `tests/test_model.py` verifying model artifacts and metric thresholds.

## Why this step matters
In machine learning projects, you cannot claim an advanced model (like LightGBM or XGBoost) is "successful" unless you benchmark it against a simpler, well-understood baseline. Logistic Regression with class weights provides a fast, convex, interpretable reference point. Furthermore, evaluating this baseline on the chronological validation set illustrates the "metric paradox": while standard ROC-AUC appears superficially strong (~0.792), PR-AUC is only ~0.199, reflecting the extreme challenge of separating rare positive fraud cases (3.4%) from genuine transactions. This sets the precise benchmark numbers we must beat in Step 06.

## Files created / changed (path + one-line purpose each)
- `src/evaluate.py`: Central evaluation engine computing PR-AUC, Recall at 80% Precision, F1, confusion matrix, and dollar cost functions.
- `src/train.py`: Baseline model training script incorporating imputation, scaling, balanced Logistic Regression, and reporting.
- `models/baseline_model.joblib`: Serialized baseline model pipeline artifact.
- `reports/baseline_metrics.json`: JSON report containing all baseline evaluation metrics on the validation set.
- `reports/baseline_confusion_matrix.png`: Visual heatmap of baseline validation classification outcomes.
- `tests/test_model.py`: Pytest suite validating baseline artifact generation and metric bounds.
- `docs/explanations/05_baseline_model.md`: Step documentation, line-by-line explanation, and interview questions.

## Key decisions and alternatives rejected
- **Logistic Regression on Curated Core Features vs. All 434 Features:** When tested on all 434 features, dense multicollinear Vesta features caused L-BFGS convergence failures and took ~219 seconds. Selecting 26 core features converged in **2.5 seconds** cleanly, providing a lightweight, robust baseline.
- **`class_weight='balanced'` vs. Unweighted Logistic Regression:** Without class weighting, Logistic Regression on a 3.5% positive dataset predicts virtually all zeros to minimize standard cross-entropy loss, yielding 0% recall. `class_weight='balanced'` inversely weights each sample by class frequency, forcing the loss function to heavily penalize fraud errors.
- **Evaluation on Validation Only vs. Testing on Holdout Test:** The holdout test set must remain unpolluted. Per PRD Section 6, the test set is touched strictly once at the end of the project (Step 07). All intermediate architecture and baseline comparisons occur on validation.

## Line-by-Line Code Explanation (`train_baseline_model` in `src/train.py`)
```python
BASELINE_FEATURES = [
    "TransactionAmt", "TransactionAmt_log", "TransactionAmt_decimal", "TransactionAmt_is_integer",
    "hour_of_day", "day_of_week", "has_identity", "is_high_amount",
    "card1_freq", "card_id_freq", "amt_diff_card1_mean", "amt_ratio_card1_mean",
    "C1", "C2", "C5", "C6", "C9", "C11", "C13", "C14", "D1", "D2", "D3", "D4", "D10", "D15"
]
```
- Selects 26 core numeric signals representing spend magnitude, transaction timing, card frequency, and payment velocity without high-dimensional multicollinearity.

```python
baseline_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs", random_state=RANDOM_SEED)),
])
```
- Assembles an atomic scikit-learn pipeline:
  1. `SimpleImputer`: Replaces missing numeric values with the feature median learned strictly from `X_train`.
  2. `StandardScaler`: Normalizes features to mean 0 and variance 1, preventing high-magnitude columns (like amount) from dominating gradient descent.
  3. `LogisticRegression`: Solves the weighted binary logistic loss using L-BFGS optimizer with class weighting inversely proportional to class frequencies ($w_1 \approx 14.2, w_0 \approx 0.52$).

```python
baseline_pipeline.fit(X_train, y_train)
```
- Trains the imputer, scaler, and regression weights strictly on 413,378 historical training examples.

```python
val_pred_proba = baseline_pipeline.predict_proba(X_val)[:, 1]
```
- Computes calibrated continuous posterior probabilities $P(\text{isFraud}=1 | x)$ on the 88,581 out-of-time validation transactions.

```python
metrics = evaluate_predictions(y_val, val_pred_proba)
```
- Passes validation predictions to `src/evaluate.py` to calculate PR-AUC (average precision), ROC-AUC, thresholded metrics, and business dollar cost ($10 \times FN + 1 \times FP$).

## Results / outputs (numbers, plots, observations)
Validation Evaluation Metrics (`reports/baseline_metrics.json`):
- **Primary Metric (PR-AUC):** **0.1987** *(This is the benchmark to beat in Step 06)*
- **Secondary Metric (ROC-AUC):** **0.7919**
- **Precision (at threshold 0.5):** **0.0956** (9.56%)
- **Recall (at threshold 0.5):** **0.6953** (69.53%)
- **F1-Score (at threshold 0.5):** **0.1682**
- **Recall at 80% Precision:** **0.0736** (7.36% fraud caught when tuned to 80% precision threshold of 0.9195)
- **Confusion Matrix on Validation (88,581 samples):**
  - True Negatives (TN): 65,542
  - False Positives (FP): 19,997 (high false alarm rate due to balanced class weighting)
  - False Negatives (FN): 927
  - True Positives (TP): 2,115
- **Business Cost Analysis:**
  - Total Baseline Cost: **$29,267.00**
  - Naive "Do Nothing" Cost: $30,420.00
  - Net Dollar Savings: $1,153.00 (**3.79% reduction**)

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: Why is ROC-AUC (0.7919) so much higher than PR-AUC (0.1987) on your baseline model?**
  *A:* ROC-AUC evaluates True Positive Rate vs False Positive Rate. Because genuine transactions vastly outnumber frauds (96.5% vs 3.5%), the denominator for FPR ($FP + TN$) is huge (~85,000). Even when the model produces 20,000 false alarms, the FPR appears moderate (~23%), giving a deceptive ROC-AUC of ~0.79. In contrast, PR-AUC evaluates Precision ($TP / [TP + FP]$). Because precision directly penalizes false alarms against rare true positives, PR-AUC drops to 0.1987, accurately reflecting that 9 out of 10 flagged transactions are false alarms.
- **Q: Why did you use `class_weight='balanced'` in Logistic Regression?**
  *A:* Standard maximum likelihood estimation minimizes total classification error. In an imbalanced dataset with 3.5% positives, a model predicting 0 for every transaction achieves 96.5% accuracy with low loss. `class_weight='balanced'` weights fraud instances by $\frac{N}{2 \times N_{\text{fraud}}} \approx 14.2$, making false negatives 14 times more costly during optimization and forcing the decision boundary to identify fraud.
- **Q: Why build a simple Logistic Regression baseline if you know gradient boosted trees (LightGBM) will perform better?**
  *A:* In production ML, you must always justify model complexity. A linear baseline sets a reproducible floor, trains in 2.5 seconds, establishes feature interpretability, and quantifies the exact marginal gain delivered by non-linear interactions and tree ensembles.
- **Q: Why must imputation and standard scaling be packaged inside a `Pipeline` instead of applied to the whole dataset upfront?**
  *A:* If you compute the median or standard deviation across the full dataset prior to splitting, the mean and variance of the validation/test sets leak into the training features. Fitting `SimpleImputer` and `StandardScaler` inside `Pipeline.fit()` ensures only training data statistics are used, preventing data leakage.

## Limitations / what I would improve
- Logistic Regression assumes linear boundaries and cannot capture feature interactions (e.g. card frequency conditional on specific hour and amount range) without manual polynomial expansion.
- The high false positive rate (19,997 false alarms) creates significant operational friction, which we will address via LightGBM and threshold tuning.

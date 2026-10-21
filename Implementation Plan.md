# Implementation Plan: Transaction Fraud Detection Pipeline

Source of truth for requirements: `PRD.md`. Follow the steps **in order**.

## Working rules (apply to every step)
1. Do **one step at a time**. When a step is finished, **stop and wait for my approval** before starting the next.
2. At the end of each step, create its explanation file in `docs/explanations/` (name given in each step). Use the template below.
3. Steps marked 🧠 are interview-critical. For these, explain the code line by line in the explanation file and ask me **3 quiz questions** before moving on.
4. Do not change decisions marked **[CONFIRM]** in the PRD. If something is unclear, ask.
5. Keep code simple and readable. No unnecessary libraries.

## Explanation file template
Every `docs/explanations/NN_name.md` must contain:
```
# Step NN: <title>
## What was done (plain English, 5-8 lines)
## Why this step matters
## Files created / changed (path + one-line purpose each)
## Key decisions and alternatives rejected
## Results / outputs (numbers, plots, observations)
## How to explain this in an interview (3-5 likely questions with short answers)
## Limitations / what I would improve
```

## Target repo structure
```
fraud-detection/
  data/ (raw, processed; gitignored)
  notebooks/
  src/ (config.py, data.py, features.py, train.py, evaluate.py)
  app/ (api.py, dashboard.py)
  models/
  reports/
  docs/explanations/
  tests/
  Dockerfile  requirements.txt  README.md
```

---

## Step 00: Project setup
**Tasks:** create repo structure, virtual env, `requirements.txt`, `.gitignore` (ignore data and models), `src/config.py` (seed, paths, cost values, split ratios).
**Done when:** `python -c "import src.config"` runs and the folder structure exists.
**Explanation file:** `docs/explanations/00_project_setup.md`

## Step 01: Data acquisition and understanding
**Tasks:** script/instructions to download IEEE-CIS data; load and join transaction + identity tables; reduce memory (dtype downcasting); EDA notebook covering size, fraud rate, missing values per column, fraud rate over time, key columns.
**Output:** `notebooks/01_eda.ipynb`, `reports/eda_summary.md`
**Done when:** I can state dataset size, fraud percentage, top missing columns and any time pattern.
**Explanation file:** `docs/explanations/01_data_understanding.md`

## Step 02: Data cleaning 🧠
**Tasks:** handle missing values (decide per column group, with reasons), fix dtypes, remove or flag duplicates/outliers, drop columns with extreme missingness (justify the threshold). Cleaning logic must be written as functions in `src/data.py`.
**Done when:** a cleaned dataset is saved and cleaning is reproducible by one function call.
**Explanation file:** `docs/explanations/02_data_cleaning.md`

## Step 03: Time-based split 🧠
**Tasks:** sort by `TransactionDT`; split 70/15/15 into train/val/test; verify no time overlap; print fraud rate in each split; explain why a random split would leak and inflate scores.
**Done when:** three saved splits and an assertion test that max(train time) < min(val time) < min(test time).
**Explanation file:** `docs/explanations/03_time_based_split.md`

## Step 04: Feature engineering
**Tasks:** encode categoricals (fit on train only); time features (hour, day of week from `TransactionDT`); amount features (log amount, decimal part); frequency/aggregate features per card or email domain computed **using only past or train data**; document every feature group.
**Done when:** a feature pipeline function exists in `src/features.py` and the same function is usable by the API later.
**Explanation file:** `docs/explanations/04_feature_engineering.md`

## Step 05: Baseline model 🧠
**Tasks:** Logistic Regression with class weights and simple scaled features; report PR-AUC, recall at fixed precision, F1, plus confusion matrix on **validation**; save metrics to `reports/baseline_metrics.json`.
**Done when:** baseline numbers exist. These are the numbers to beat.
**Explanation file:** `docs/explanations/05_baseline_model.md`

## Step 06: Improved model and imbalance handling
**Tasks:** train LightGBM; compare class weights vs. `scale_pos_weight` vs. undersampling; light hyperparameter tuning using validation only (time-aware); pick the best by validation PR-AUC; save the model to `models/`.
**Output:** `reports/model_comparison.csv` (all variants, one row each)
**Done when:** improvement over baseline is quantified with the same metrics.
**Explanation file:** `docs/explanations/06_improved_model.md`

## Step 07: Threshold selection and cost analysis 🧠
**Tasks:** plot precision-recall curve; compute total cost for thresholds using cost_FN and cost_FP from `config.py`; choose the threshold minimising validation cost; then evaluate **once** on the test set; report final metrics and cost saved vs. baseline.
**Output:** `reports/final_metrics.json`, `reports/pr_curve.png`, `reports/cost_vs_threshold.png`
**Explanation file:** `docs/explanations/07_threshold_and_cost.md`

## Step 08: Error analysis and explainability
**Tasks:** SHAP summary plot; inspect top false positives and false negatives; find patterns (amount range, product type, time); list 3 concrete insights and 3 possible improvements.
**Output:** `reports/shap_summary.png`, `reports/error_analysis.md`
**Explanation file:** `docs/explanations/08_error_analysis.md`

## Step 09: API (FastAPI)
**Tasks:** `/health` and `/predict` endpoints; request validation with Pydantic; reuse `src/features.py` and the saved model; return fraud probability, decision at chosen threshold, and top 3 contributing features; add 3-5 pytest tests.
**Done when:** `uvicorn app.api:app` works locally and tests pass.
**Explanation file:** `docs/explanations/09_api.md`

## Step 10: Dashboard (Streamlit)
**Tasks:** simple UI to upload a CSV or enter a transaction; call the model or API; show fraud score, decision and reasons; add a page with metrics and plots from `reports/`.
**Explanation file:** `docs/explanations/10_dashboard.md`

## Step 11: Docker and deployment
**Tasks:** `Dockerfile` for the API (and dashboard if feasible); run locally in Docker; deploy to a free host (Render / Hugging Face Spaces); verify the live URL.
**Explanation file:** `docs/explanations/11_docker_and_deployment.md`

## Step 12: README and resume packaging
**Tasks:** README with problem, dataset, approach, **results table (baseline vs final)**, charts, architecture diagram (Mermaid), how to run, live link; fill the resume bullet from `PRD.md` with real numbers; write a 1-page "project interview cheat sheet" (metric choice, split choice, imbalance handling, threshold logic, limitations).
**Output:** `README.md`, `docs/interview_cheatsheet.md`
**Explanation file:** `docs/explanations/12_readme_and_resume.md`

---

## Optional stretch (only after Step 12 is approved)
- Add a small LLM layer that explains a flagged transaction in plain English from the SHAP reasons.
- Add data-drift check on the test period.

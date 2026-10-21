# PRD: Transaction Fraud Detection Pipeline

> Owner: Lakshit | Goal: flagship resume project for Data Scientist / ML internships (2027)
> Items marked **[CONFIRM]** are my decisions. The agent must not change them without asking.

## 1. Problem
Payment fraud is rare (about 3-4% of transactions) and costly. Accuracy is useless here because a model that predicts "not fraud" always scores about 96%. Build an end-to-end pipeline that detects fraudulent transactions on messy, imbalanced, time-ordered data, and judge it with metrics that match the business problem.

## 2. Goals
- Build a full pipeline: data -> cleaning -> features -> model -> evaluation -> API -> dashboard -> deployment.
- Show a measured improvement over a simple baseline.
- Evaluate with a **time-based split** (no random split, no leakage).
- Show the cost trade-off between catching fraud and falsely flagging genuine users.
- Every step is documented and explainable by me in an interview.

## 3. Non-goals (out of scope)
- Deep learning / neural nets
- Real-time streaming infrastructure (Kafka, Spark)
- LLM or chatbot features (optional stretch only after everything else is done)
- Chasing leaderboard-level scores

## 4. Dataset
- **Primary:** IEEE-CIS Fraud Detection (Kaggle): `train_transaction.csv` + `train_identity.csv`, joined on `TransactionID`. About 590k rows, many missing values, anonymised columns, `TransactionDT` time column.
- **Fallback (if too heavy for my laptop):** a sampled subset, keeping time order. The sampling method must be documented.
- Data is never committed to Git. Provide a download script / instructions in the README.

## 5. Metrics **[CONFIRM]**
| Type | Metric | Why |
|---|---|---|
| Primary | PR-AUC (average precision) | Right metric for rare-positive problems |
| Secondary | Recall at a fixed precision (e.g. 80%) | Business-style operating point |
| Secondary | F1 at chosen threshold | Easy to put on resume |
| Business | Total cost = (missed fraud x cost_FN) + (false alarms x cost_FP) | Threshold selection |
| Not used | Accuracy | Misleading on imbalanced data |

Assumed costs (editable in `config.py`): cost_FN = 10x cost_FP.

## 6. Validation strategy **[CONFIRM]**
- Sort by `TransactionDT`.
- Train = earliest 70%, Validation = next 15%, Test = latest 15%.
- Test set is touched **once**, at the end.
- All encoders, scalers and aggregations are fit on train only.

## 7. Modelling plan
1. **Baseline:** Logistic Regression (with class weights) on simple features.
2. **Main model:** LightGBM (or XGBoost) with engineered features.
3. **Imbalance handling comparison:** class weights vs. scale_pos_weight vs. undersampling. Pick with validation PR-AUC, and note why.
4. **Threshold selection** on validation using the cost function.
5. **Explainability:** SHAP feature importance + error analysis (false positives / false negatives).

## 8. Deliverables
- Clean GitHub repo with a README (problem, approach, results table, charts, how to run)
- Reproducible notebooks (EDA, modelling) and `src/` Python modules
- FastAPI service with a `/predict` endpoint and `/health`
- Streamlit dashboard (upload CSV or enter a transaction -> fraud score + top reasons)
- Dockerfile
- Live deployed link (Render / Hugging Face Spaces / similar free tier)
- `docs/explanations/` with one explanation file per implementation step

## 9. Tech stack
Python 3.11+, pandas, numpy, scikit-learn, LightGBM, SHAP, matplotlib/seaborn, FastAPI, Uvicorn, Streamlit, Docker, pytest (a few basic tests), Git.

## 10. Quality rules
- Fixed random seeds, `requirements.txt` pinned
- No data leakage (document how it was prevented)
- Small readable functions, docstrings, no giant notebooks
- Results saved to `reports/` (metrics JSON + plots)

## 11. Definition of done
- [x] Baseline and improved model both reported on the same test set
- [x] Improvement over baseline is stated with numbers
- [x] Threshold justified by cost analysis
- [x] API runs locally and in Docker, and is deployed
- [x] README has results table + screenshots + live link
- [x] Every step has an explanation file
- [x] I can explain every decision without looking at the code

## 12. Resume bullet template (fill with real numbers at the end)
> Built an end-to-end fraud detection pipeline on ~590K transactions with a time-based validation split; improved PR-AUC from **0.13** (logistic regression baseline) to **0.55** (LightGBM) and cut simulated fraud cost by **54.8%** ($17,754 savings) via business cost threshold tuning; deployed via FastAPI + Docker with a Streamlit dashboard.

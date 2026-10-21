# 1-Page Project Interview Cheat Sheet: Transaction Risk Engine

This guide condenses the core engineering, statistical decisions, and trade-offs behind the **Transaction Risk Engine** into high-yield talking points for Data Science & ML Engineering interviews.

---

## 1. The 30-Second Elevator Pitch
> *"I built a production-grade, cost-sensitive fraud detection engine on 590,000 real-world payment transactions from IEEE-CIS. Because fraud represents only 3.5% of volume and transaction patterns evolve over time, I enforced a strict chronological 70/15/15 validation split to prevent future-data leakage. By moving from a logistic regression baseline to a feature-engineered LightGBM and tuning the decision threshold against business risk ($Cost_{FN} = 10 \times Cost_{FP}$), I improved out-of-time PR-AUC from 0.13 to 0.55 (+330%) and cut simulated financial fraud loss by 54.8% ($17,754 net savings on the holdout test set). Finally, I packaged the system into a sub-15ms FastAPI service, an interactive Streamlit operations dashboard, and multi-container Docker orchestration."*

---

## 2. Why PR-AUC Over ROC-AUC and Accuracy?
- **Accuracy is useless:** A naive model that classifies every transaction as legitimate scores **96.5% accuracy** while catching 0% of fraud, bankrupting the business.
- **ROC-AUC is deceptively optimistic:** The False Positive Rate (FPR) denominator includes all true negatives ($FP / (FP + TN)$). Because legitimate transactions ($TN$) are overwhelmingly massive (~96.5%), tens of thousands of false alarms barely budge the FPR, masking severe precision degradation.
- **PR-AUC (Average Precision) is the gold standard:** Precision ($TP / (TP + FP)$) directly exposes false alarms to true hits. The PR-AUC baseline is the fraud base rate (**0.035**). Our baseline scored **0.1284**, and the final LightGBM achieved **0.5522** (+330% relative gain).

---

## 3. Validation Strategy: Strict Chronological Split
- **Why NOT random K-Fold / train_test_split?** Fraud is a non-stationary temporal adversarial game. Random splitting leaks future card velocity, user counts, and evolving fraud rings into the past, causing severe optimistic bias (train on tomorrow, test on yesterday).
- **Our Implementation:** Sorted strictly by `TransactionDT`:
  - **Train (earliest 70%):** 413,378 rows (3.52% fraud)
  - **Validation (next 15%):** 88,581 rows (3.43% fraud) — used exclusively for model selection and threshold tuning.
  - **Test (latest 15%):** 88,581 rows (3.48% fraud) — held out and touched **strictly once** at Step 07.
- **Hard Assertion:** `max(train_DT: 10437996) < min(val_DT: 10438003) < min(test_DT: 13151880)` verified in `tests/test_splits.py`.

---

## 4. Class Imbalance: Why Did Unweighted LightGBM Win?
We benchmarked 4 distinct imbalance strategies on the chronological validation set:
1. `class_weight='balanced'`: PR-AUC = 0.5385
2. `RandomUnderSampler (1:10)`: PR-AUC = 0.5650
3. `scale_pos_weight=5.0`: PR-AUC = 0.5703
4. **Tuned Unweighted LightGBM + Post-Hoc Thresholding:** **PR-AUC = 0.6134** (Val) / **0.5522** (Test).

**The Core Insight:** 
Artificially reweighting or undersampling shifts the predicted probability calibration distribution away from true posterior probabilities $P(Y=1|X)$, hurting the model's ranking ability across fine-grained risk boundaries. Training unweighted preserves true relative rankings, and post-hoc threshold tuning directly aligns the decision boundary with asymmetric business costs.

---

## 5. Threshold Selection & Asymmetric Cost Optimization
- **Default 0.50 Threshold Failure:** Standard binary classification assumes false negatives (FN) and false positives (FP) carry equal cost. In payments, letting a fraudulent charge pass costs 10x more than an investigator review or SMS step-up challenge.
- **Cost Function:**
  $$\text{Total Cost} = (FN \times \$10.00) + (FP \times \$1.00)$$
- **Optimization:** We swept thresholds from $0.01$ to $0.99$ on the validation set. The optimal operating point was **$T^* = 0.0543$**.
- **Financial Impact on Holdout Test Set:**
  - Baseline Logistic Regression Cost: **$32,378.00**
  - Final LightGBM Cost at $T^* = 0.0543$: **$14,624.00**
  - **Net Savings:** **$17,754.00 (54.83% cost reduction)**.

---

## 6. Key Features & Explainability (Tree SHAP)
- **Top 5 Features:**
  1. `C13`: Frequency counter of transactions associated with the payment device/account.
  2. `TransactionAmt`: Extreme transaction amounts (both micro-testing <$5 and luxury spikes >$1,000).
  3. `C1`: Frequency count of associated cards.
  4. `card6`: Card type (credit cards show substantially higher fraud rates than debit cards).
  5. `V70` & `C14`: Device and identity verification counts.
- **Local Real-Time SHAP:** Integrated into the FastAPI `/predict` payload and Streamlit UI to give fraud investigators instant transparency into *why* a transaction was flagged.

---

## 7. Error Analysis: Model Failure Modes
1. **False Negatives (Missed Fraud):** Primarily low-dollar "card testing" transactions ($1.50 - $15.00) executed from familiar consumer domains (`gmail.com`). Solution: Sub-minute sliding velocity counters in Redis.
2. **False Positives (Genuine Flagged):** Legitimate high-ticket purchases ($800+) made during travel or holiday promotions without device identity telemetry. Solution: Automated SMS/Push step-up verification (3D-Secure 2.0).

---

## 8. Latency & Production Architecture
- **Inference Service (FastAPI):** <15ms end-to-end latency with Pydantic v2 validation. Direct in-memory LightGBM C++ inference.
- **Decision Engine Tiers:**
  - Score $< 0.0543 \rightarrow$ **`APPROVE`** (Instant zero-friction checkout)
  - $0.0543 \le \text{Score} < 0.25 \rightarrow$ **`CHALLENGE`** (Trigger 3D-Secure / OTP / manual review)
  - Score $\ge 0.25 \rightarrow$ **`DECLINE`** (Hard stop, protect merchant)
- **Docker Packaging:** Multi-stage `python:3.11-slim` with `libgomp1` installed to satisfy OpenMP shared memory requirements for LightGBM.

---

## 9. Interview "Gotchas" & Instant Answers
- **Q: "Why didn't you use SMOTE?"**
  *A: SMOTE creates synthetic points by linear interpolation between minority neighbors. In high-dimensional mixed-type fraud data (sparse categorical + skewed counts), interpolating coordinates generates unrealistic financial behavior that blurs real decision boundaries and slows down training on 590K rows.*
- **Q: "Why didn't you try Deep Learning?"**
  *A: On tabular data with high cardinality and extreme sparsity, gradient boosted decision trees (LightGBM/XGBoost) consistently outperform deep neural networks in training speed, inference latency (<15ms), memory efficiency, and native missing value handling without complex embedding warmups.*
- **Q: "What would you build next in enterprise production?"**
  *A: A Redis-backed sliding-window feature store (Feast) for sub-second card velocity aggregations (e.g., "count of transactions in past 5 minutes"), and a Graph Neural Network (GNN) to uncover coordinated fraud syndicates sharing IP/device subnets.*

# Exploratory Data Analysis (EDA) Summary

## 1. Dataset Dimensions & Memory Footprint
- **Total Transactions:** 590,540
- **Total Features:** 434 (394 from transactions, 40 from identity tables, linked via `TransactionID`)
- **Raw Memory Footprint:** ~1,848 MB
- **Optimized Memory Footprint (Downcasted):** ~1,073 MB (**47.9% reduction**)
- **Temporal Span:** 183 days (~6 months, from day 1 to day 182)

## 2. Class Imbalance & The Metric Paradox
- **Genuine Transactions (Class 0):** 569,877 (96.50%)
- **Fraudulent Transactions (Class 1):** 20,663 (**3.50%**)
- **Imbalance Ratio:** ~27.6 : 1
- **Key Takeaway for Modeling:** A naive dummy model predicting "never fraud" attains **96.50% accuracy**, yet catches **$0 of fraud**. Accuracy is mathematically bankrupt for this problem. We must use **PR-AUC (Precision-Recall AUC)** and cost-weighted trade-offs.

## 3. Missing Value Analysis
- **Columns with 0% missing:** 20
- **Columns with >50% missing:** 214 columns (~49.3% of all features)
- **Columns with >80% missing:** 74 columns (~17.1% of all features)
- **Columns with >90% missing:** 12 columns

### Top 10 Most Sparse Columns
| Column | Missing Percentage |
|---|---|
| `id_24` | 99.20% |
| `id_25` | 99.13% |
| `id_07` | 99.13% |
| `id_08` | 99.13% |
| `id_21` | 99.13% |
| `id_26` | 99.13% |
| `id_27` | 99.12% |
| `id_23` | 99.12% |
| `id_22` | 99.12% |
| `dist2` | 93.63% |

*Insight:* Identity features (`id_01` to `id_38`) and device information (`DeviceInfo`, `DeviceType`) are only populated when additional browser/device telemetry was collected (usually on web checkouts), causing ~75-80% missingness. However, transactions that *do* present identity data have a distinct fraud propensity.

## 4. Temporal Dynamics & Distribution Drift
- Fraud rate is **not static** across time. It fluctuates from ~1.5% up to ~9.0% on specific peak days.
- Transaction volume exhibits weekly cyclical patterns (weekend vs. weekday shifts).
- **Critical Interview Implication:** A random `train_test_split` would scramble transactions from the future into the past, resulting in massive data leakage (e.g. historical aggregate card features computed across future dates) and severe metric inflation. A **strict chronological split** (earliest 70% train, next 15% validation, final 15% test) is strictly required.

## 5. Monetary Patterns (`TransactionAmt`)
- **Median Transaction Amount:**
  - Genuine: $68.50
  - Fraud: $75.00
- **Mean Transaction Amount:**
  - Genuine: $134.51
  - Fraud: $149.24
- *Observation:* Fraudulent transactions exhibit higher variance and higher average values, confirming that financial fraud seeks high-yield unauthorized card transfers.

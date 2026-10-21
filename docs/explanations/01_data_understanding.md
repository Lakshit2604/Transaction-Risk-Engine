# Step 01: Data Acquisition and Understanding

## What was done (plain English, 5-8 lines)
Configured Kaggle API integration to securely download the primary IEEE-CIS Fraud Detection dataset (`train_transaction.csv` and `train_identity.csv`).
Extracted and validated the raw CSV files in `data/raw/` totaling ~677 MB on disk.
Engineered a memory downcasting utility in `src/data.py` converting numeric datatypes, slashing memory footprint by 47.9% (~1.85 GB down to ~1.07 GB).
Merged transaction and identity tables via left join on `TransactionID` and strictly ordered records chronologically by `TransactionDT`.
Conducted comprehensive exploratory data analysis inspecting class imbalance, missingness patterns, and temporal fraud rate drift over 182 days.
Generated an exploratory notebook `notebooks/01_eda.ipynb`, visualization charts in `reports/`, and a detailed findings report `reports/eda_summary.md`.

## Why this step matters
High-cardinality financial fraud datasets cannot simply be thrown into machine learning models without understanding their statistical structure and physical constraints. First, the IEEE-CIS dataset exceeds typical laptop RAM if loaded naively with 64-bit precision; downcasting is mandatory for reproducible local development. Second, establishing the exact ground truth fraud prevalence (~3.50%) mathematically disproves accuracy as an acceptable metric (predicting all zeros yields 96.5% accuracy but zero fraud prevented). Third, observing non-stationary fraud rates over time proves why standard random cross-validation leaks information and why a chronological split is non-negotiable.

## Files created / changed (path + one-line purpose each)
- `scripts/download_data.py`: CLI and automated Kaggle API script to fetch, extract, and verify dataset integrity.
- `src/data.py`: Production data loading, memory downcasting (`reduce_mem_usage`), table merging, and summary statistics.
- `scripts/generate_eda.py`: Script executing the full EDA workflow and generating charts, reports, and notebooks.
- `reports/eda_summary.md`: Structured analytical summary of dimensions, class distribution, missingness, and monetary patterns.
- `reports/eda_class_distribution.png`: Visualization showing the extreme 27.6:1 class imbalance.
- `reports/eda_fraud_trend_over_time.png`: Visualization of daily fraud rate volatility and transaction volume over 182 days.
- `reports/eda_missing_distribution.png`: Distribution of feature missingness across all 434 columns.
- `notebooks/01_eda.ipynb`: Interactive Jupyter notebook mirroring the data inspection and analysis pipeline.
- `docs/explanations/01_data_understanding.md`: Step documentation and interview prep guide.

## Key decisions and alternatives rejected
- **Downcasting to 32-bit/16-bit/8-bit vs. Using large cloud instances:** Chose intelligent memory downcasting (`int8`/`int16`/`int32` and `float32`) which shrank memory by ~48%. Rejecting brute-force cloud RAM upgrades keeps the project lightweight, portable, and runnable on consumer laptops without sacrificing numerical precision.
- **Left join on `TransactionID` vs. Inner join:** The identity table only has ~144k records out of ~590k transactions (representing transactions with device/browser telemetry). An inner join would have discarded ~75% of all transactions and biased the training distribution. A left join preserves every transaction while capturing identity signals where present.
- **Sorting by `TransactionDT` immediately on load vs. Unordered loading:** Transactions were sorted chronologically upon ingestion to eliminate any risk of future-to-past data leakage during subsequent windowing or feature aggregation.
- **Automated Kaggle API script vs. Manual only:** Provided dual-mode handling (automated Kaggle CLI download + clear fallback instructions) ensuring seamless automation on CI/CD or local machines.

## Results / outputs (numbers, plots, observations)
- **Dataset Dimensions:** 590,540 rows × 434 columns (394 transaction features, 40 identity features).
- **Target Distribution:**
  - Genuine (0): 569,877 (96.50%)
  - Fraud (1): 20,663 (3.50%)
  - Imbalance Ratio: ~27.6 : 1
- **Missing Data Profile:**
  - 214 columns (~49.3%) have >50% missing values.
  - 74 columns (~17.1%) have >80% missing values (primarily sparse identity and secondary distance fields).
- **Temporal Span:**
  - `TransactionDT` covers 182 days (~6 months) with daily fraud rate varying dynamically between 1.5% and 9.0%.
- **Monetary Profile (`TransactionAmt`):**
  - Genuine transactions: Median $68.50, Mean $134.51
  - Fraudulent transactions: Median $75.00, Mean $149.24 (higher variance and higher average ticket size).

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: What is the biggest challenge of working with tabular fraud datasets like IEEE-CIS?**
  *A:* The twin challenges of extreme class imbalance (~3.5% positive rate) and massive feature sparsity (over 200 columns with >50% missing). Naive evaluation metrics like accuracy fail completely, and naive models struggle with high-dimensional sparsity.
- **Q: How did you fit ~600,000 rows with 430+ columns into memory on a standard laptop?**
  *A:* Pandas defaults to 64-bit integers and 64-bit floats, consuming ~1.85 GB. I wrote a `reduce_mem_usage` function that inspects the minimum and maximum value of each column and downcasts types (e.g. categorical IDs to `int16`, flags to `int8`, and floats to `float32`). This reduced memory usage by 47.9% to ~1.07 GB without loss of significant precision.
- **Q: Why are there so many missing values in the identity features? Did you just drop them?**
  *A:* No, dropping them would lose critical signals. The identity table contains browser and device telemetry, which is only triggered during online web/mobile checkouts (not POS card-present swipes). The missingness itself is informative (Missing Not at Random - MNAR), which tree-based algorithms like LightGBM can naturally handle.
- **Q: What did the temporal analysis of `TransactionDT` reveal?**
  *A:* Fraud is non-stationary. The fraud rate isn't a constant 3.5% every day; it spikes up to 9% during specific attack periods. This proves that a standard random k-fold split would leak future attack patterns into the training set and artificially inflate validation scores. A chronological split is mandatory.

## Limitations / what I would improve
- In a production streaming system, data ingestion would ingest events incrementally through an event bus (e.g., Kafka) rather than batch CSV loading.
- We could enrich the timestamps with external holiday and calendar indicators once timezone offsets are estimated.

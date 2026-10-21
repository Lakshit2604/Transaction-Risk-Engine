# Step 03: Time-Based Split 🧠

## What was done (plain English, 5-8 lines)
Sorted the entire 590,540 cleaned transactions chronologically by `TransactionDT`.
Partitioned the dataset using the 70/15/15 ratio into Train (earliest 70%), Validation (next 15%), and Holdout Test (latest 15%).
Verified zero temporal overlap, proving `max(train time) = 10,437,996 < min(val time) = 10,438,003 < min(test time) = 13,151,880`.
Confirmed consistent class distributions across partitions (Train: 3.52%, Val: 3.43%, Test: 3.48%).
Persisted the partitions to `data/processed/train.parquet` (56.0 MB), `val.parquet` (11.9 MB), and `test.parquet` (12.8 MB).
Authored formal unit tests in `tests/test_splits.py` asserting temporal integrity and row conservation, all passing via pytest.

## Why this step matters
In real-world fraud detection, models are trained on past transactions to predict future unknown fraud. A random split (`train_test_split`) violates the arrow of time: future fraud events leak into the training set, while historical behavior leaks into evaluation. When historical aggregations (e.g. card transaction counts over past 7 days) are computed, a random split allows the model to "peek into the future," creating catastrophic target leakage. In production, such models collapse because the real future has not occurred yet. A strict time-based split is the only scientifically valid way to measure true out-of-time generalizability.

## Files created / changed (path + one-line purpose each)
- `src/data.py`: Added `split_time_data()`, `save_splits()`, and `load_splits()` enforcing chronological boundaries.
- `tests/test_splits.py`: Automated pytest test suite verifying temporal boundary separation and row conservation.
- `data/processed/train.parquet`: 413,378 earliest training transactions (56.0 MB).
- `data/processed/val.parquet`: 88,581 intermediate validation transactions for hyperparameter & threshold tuning (11.9 MB).
- `data/processed/test.parquet`: 88,581 latest out-of-time hold-out test transactions touched strictly once (12.8 MB).
- `docs/explanations/03_time_based_split.md`: Step documentation, line-by-line explanation, and interview questions.

## Key decisions and alternatives rejected
- **Strict Chronological Slicing vs. Scikit-learn `train_test_split(shuffle=True)`:** Standard shuffled splitting was rejected because it causes severe future-to-past data leakage and artificially inflates PR-AUC scores by 15-25%.
- **Time-based K-Fold / Expanding Window vs. Single 70/15/15 Split:** While rolling expanding-window validation is useful, a fixed 70/15/15 chronological split mirrors real enterprise batch training where models are trained on historical months (months 1-4), validated on month 5, and tested on month 6.
- **Saving distinct Parquet partitions vs. In-memory slicing during training:** Saving explicit files guarantees that downstream feature engineering and model scripts cannot accidentally access validation/test targets during encoder fitting.

## Line-by-Line Code Explanation (`split_time_data` in `src/data.py`)
```python
def split_time_data(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO, verbose: bool = True):
```
- Defines the time-splitting function with default ratios (0.70, 0.15, 0.15) imported from central `src/config.py`.

```python
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
```
- Verifies mathematical consistency so no records are accidentally dropped or unassigned.

```python
    df_sorted = df.sort_values(by=TIME_COL).reset_index(drop=True)
```
- Sorts every record chronologically by `TransactionDT` (seconds from reference epoch) and resets indices so index position corresponds strictly to time.

```python
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
```
- Computes split index cutoffs: record 0 to 413,377 (Train), 413,378 to 501,958 (Val), and 501,959 to 590,539 (Test).

```python
    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)
```
- Explicitly copies the slices to prevent pandas `SettingWithCopyWarning` and isolates each partition into independent DataFrames.

```python
    train_max_t = train_df[TIME_COL].max()
    val_min_t = val_df[TIME_COL].min()
    val_max_t = val_df[TIME_COL].max()
    test_min_t = test_df[TIME_COL].min()
```
- Extracts the exact boundary timestamps across all partitions.

```python
    assert train_max_t < val_min_t, f"Temporal overlap between Train and Val: {train_max_t} >= {val_min_t}"
    assert val_max_t < test_min_t, f"Temporal overlap between Val and Test: {val_max_t} >= {test_min_t}"
```
- Hard assertion: guarantees that the latest training timestamp is strictly before the earliest validation timestamp, and latest validation is strictly before test.

## Results / outputs (numbers, plots, observations)
- **Train Split:** 413,378 transactions (70%) | 14,538 frauds (**3.52%**) | Time span: [86,400 to 10,437,996] (~Day 1 to 121)
- **Validation Split:** 88,581 transactions (15%) | 3,042 frauds (**3.43%**) | Time span: [10,438,003 to 13,151,840] (~Day 121 to 152)
- **Test Split:** 88,581 transactions (15%) | 3,083 frauds (**3.48%**) | Time span: [13,151,880 to 15,811,131] (~Day 152 to 183)
- **Boundary Verification:**
  - `max(train) = 10,437,996 < min(val) = 10,438,003` (gap of 7 seconds)
  - `max(val) = 13,151,840 < min(test) = 13,151,880` (gap of 40 seconds)
- **Test Suite Status:** `tests/test_splits.py` (4 tests passed in 1.91s).

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: Why is standard K-Fold Cross Validation or random split invalid for credit card fraud detection?**
  *A:* Shuffled splitting causes look-ahead data leakage. In credit card fraud, patterns change dynamically as fraud syndicates adapt techniques. If you randomly split, transactions from tomorrow are used to predict transactions from yesterday. Furthermore, user-level aggregations (e.g. card frequency in past 24 hours) would incorporate future transactions into historical features. A chronological split reflects actual deployment.
- **Q: How does a time-based split impact model evaluation compared to random splitting?**
  *A:* Random splits yield artificially high performance (often PR-AUC > 0.90) because the test set contains known cards, identical IP sessions, and familiar attack waves from the training set. A time-based split produces lower, realistic scores (e.g. PR-AUC ~0.60 - 0.75) because the model must generalize to completely new fraud rings and shifted distributions.
- **Q: What is the purpose of having both a validation set and a test set under time splitting?**
  *A:* The validation set (the middle 15%) is used for hyperparameter tuning, model comparison (e.g. Logistic Regression vs LightGBM), and cost-optimal decision threshold tuning. The holdout test set (the latest 15%) is strictly sealed until all modeling decisions are locked, ensuring no implicit human overfitting on test metrics.
- **Q: Did you observe concept drift or fraud rate changes between train, val, and test?**
  *A:* The overall baseline prevalence remained stable (Train: 3.52%, Val: 3.43%, Test: 3.48%), but day-to-day volatility was observed during EDA. This ensures our model is evaluated under realistic non-stationary conditions.

## Limitations / what I would improve
- In enterprise pipelines, one could implement TimeSeriesSplit (Purged Group Time Series Split) with an embargo window between splits to eliminate boundary leakage from multi-day card authorizations.

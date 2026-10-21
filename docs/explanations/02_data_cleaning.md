# Step 02: Data Cleaning 🧠

## What was done (plain English, 5-8 lines)
Verified that all 590,540 rows have unique `TransactionID` values with zero duplicate records.
Engineered informative quality indicators prior to column filtering: `has_identity` (flagging digital identity telemetry capture) and `is_high_amount` (flagging transactions > $5,000).
Evaluated feature sparsity and dropped 12 virtually empty columns possessing >90% missing values (`dist2`, `D7`, `id_07`, `id_08`, `id_18`, `id_21`-`id_27`).
Standardized 29 categorical columns by trimming leading/trailing whitespaces, lowercasing tokens, and encoding missing entries explicitly as `'missing'`.
Re-applied numerical datatype downcasting to maintain minimal memory consumption.
Persisted the reproducible cleaned dataset to `data/processed/cleaned_data.parquet` (compressed from ~1.85 GB in memory to 79.8 MB on disk).

## Why this step matters
Raw financial telemetry is notoriously noisy, high-dimensional, and unstandardized. If missing values in categorical fields are discarded or naively filled with statistical modes, the model loses the critical signal that certain telemetry was never submitted (Missing Not at Random - MNAR). Dropping columns that have over 90% missing values cleans out noise and avoids tree-split fragmentation on near-empty features, while retaining columns that contain real signals. Persisting the output into Apache Parquet guarantees type fidelity (preserving downcasted ints, floats, and categories) and provides 10x faster I/O for subsequent pipeline stages without repeated parsing.

## Files created / changed (path + one-line purpose each)
- `src/data.py`: Added `clean_data()`, `save_processed_data()`, and `load_cleaned_data()` implementing reproducible cleaning rules.
- `data/processed/cleaned_data.parquet`: Cleaned production dataset (590,540 rows × 424 columns, 79.8 MB).
- `docs/explanations/02_data_cleaning.md`: Step documentation, line-by-line code explanation, and interview quiz.

## Key decisions and alternatives rejected
- **Dropping columns with >90% missingness vs. >50% missingness:** 214 columns in IEEE-CIS have >50% missingness, largely because browser identity data (`id_01` to `id_38`) is only recorded for online digital transactions. Dropping at 50% would have eliminated nearly all fraud identity signals. Choosing a 90% threshold selectively removed only 12 near-empty columns (e.g. `id_24` with 99.2% missingness) while preserving all rich identity fields.
- **Explicit `'missing'` category vs. Mode imputation for categoricals:** Mode imputation distorts the natural frequency of categories and injects false certainty. In payment security, an omitted email domain or absent device OS is an active operational signal. Treating `NaN` as `'missing'` allows tree models to split on missingness directly.
- **Flagging high amounts (`is_high_amount`) vs. Truncating / Dropping outliers:** Transactions above $5,000 are rare (only 18 in the entire dataset), but dropping them would discard genuine high-dollar fraud attacks. A binary indicator allows linear and tree models to adjust sensitivity for extreme transaction sizes without corrupting the raw monetary scale.
- **Apache Parquet vs. CSV for processed storage:** CSV loses datatype specifications on reload (turning categories back into generic objects and 16-bit integers back into 64-bit). Parquet retains exact schema, metadata, and downcasted types with Snappy compression (79.8 MB vs 700+ MB CSV).

## Line-by-Line Code Explanation (`clean_data` in `src/data.py`)
```python
def clean_data(df: pd.DataFrame, missing_threshold: float = 0.90, verbose: bool = True):
```
- Defines the modular cleaning function taking the raw merged DataFrame, a configurable missingness threshold, and a logging verbosity flag.

```python
    if ID_COL in df.columns:
        num_dupes = df[ID_COL].duplicated().sum()
        if num_dupes > 0:
            df = df.drop_duplicates(subset=[ID_COL]).reset_index(drop=True)
```
- Checks for duplicate transaction records on `TransactionID`. Guarantees 1-to-1 transaction uniqueness so no duplicate transaction inflates training or evaluation metrics.

```python
    df = df.copy()
    if "id_01" in df.columns:
        df["has_identity"] = df["id_01"].notnull().astype(np.int8)
    elif "DeviceType" in df.columns:
        df["has_identity"] = df["DeviceType"].notnull().astype(np.int8)
    else:
        df["has_identity"] = np.int8(0)
```
- Creates the `has_identity` flag. Since the identity table is left-joined, transactions lacking identity records have nulls in `id_01` and `DeviceType`. This creates a clean 1-byte (`int8`) boolean feature distinguishing digital checkouts from card swipes before any sparse columns are pruned.

```python
    if "TransactionAmt" in df.columns:
        df["is_high_amount"] = (df["TransactionAmt"] > 5000.0).astype(np.int8)
```
- Generates an indicator for extreme transaction amounts (> $5,000) rather than deleting valid fraud records.

```python
    missing_fraction = df.isnull().mean()
    dropped_cols = missing_fraction[missing_fraction > missing_threshold].index.tolist()
    protected_cols = {ID_COL, TIME_COL, TARGET_COL, "has_identity", "is_high_amount"}
    dropped_cols = [c for c in dropped_cols if c not in protected_cols]
    df = df.drop(columns=dropped_cols)
```
- Computes column-wise null percentage. Filters columns where null ratio exceeds 90%, protects target/id/time columns from accidental deletion, and drops the 12 extreme-sparsity features.

```python
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in cat_cols:
        s = df[col].astype(str).str.strip().str.lower()
        s = s.replace({"nan": "missing", "<na>": "missing", "none": "missing", "": "missing"})
        df[col] = s.astype("category")
```
- Detects all text/categorical features. Standardizes strings (removes spaces, enforces lowercase to merge duplicates like "Visa" / "visa"), converts missing values to `'missing'`, and casts each to memory-efficient pandas `category` dtype.

```python
    df = reduce_mem_usage(df, verbose=verbose)
    return df, stats
```
- Re-runs numeric downcasting across all integer/float columns to preserve minimal RAM footprint and returns the cleaned DataFrame alongside transformation statistics.

## Results / outputs (numbers, plots, observations)
- **Initial Shape:** (590,540, 434)
- **Final Cleaned Shape:** (590,540, 424)
- **Duplicates Dropped:** 0 (all 590,540 `TransactionID` values were strictly unique).
- **Columns Dropped (12 cols > 90% missing):** `['dist2', 'D7', 'id_07', 'id_08', 'id_18', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']`.
- **Engineered Quality Flags:** `has_identity` (24.4% positive), `is_high_amount` (18 transactions).
- **Categorical Columns Standardized:** 29 columns converted to clean `category` with `'missing'` tokens.
- **Disk Storage:** `cleaned_data.parquet` takes only **79.8 MB** with Snappy compression.

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: Why didn't you drop all columns with more than 50% missing values?**
  *A:* In fraud detection, missingness is often "Missing Not at Random" (MNAR). For instance, identity features like device OS or browser version only exist when transactions occur on web or mobile apps. Dropping columns at 50% missingness would purge nearly all identity signals. We only dropped columns with >90% missingness where fewer than 10% of samples contained any data.
- **Q: Why encode missing categorical values as `'missing'` instead of using the mode?**
  *A:* Mode imputation falsely assigns the most common category (e.g., claiming an unknown browser is "Chrome"), which conceals anomalies. Fraud rings frequently mask user-agents or route through headless browsers that omit standard headers. Creating an explicit `'missing'` category enables tree models to treat omitted telemetry as an informative feature.
- **Q: How did you handle extreme monetary outliers in `TransactionAmt`?**
  *A:* We never blindly delete extreme amount transactions in fraud detection because fraudsters intentionally target high amounts to maximize payout. Instead of trimming or capping, we created an explicit `is_high_amount` boolean indicator and kept the true dollar amount for downstream log-transforms.
- **Q: Why store intermediate clean data in Parquet instead of CSV?**
  *A:* Parquet is a columnar binary format. It serializes downcasted types (`int8`, `float32`, `category`) directly, compresses from ~1.85 GB to 79.8 MB, and avoids re-parsing text strings into numbers, speeding up read times by over 10x.

## Limitations / what I would improve
- In production, string normalization rules (e.g. mapping rare browser versions to top-N families) should be maintained in a version-controlled registry.
- Outlier detection could be expanded to multi-dimensional anomalies (e.g. Isolation Forest on Amount vs. Velocity) once features are engineered in Step 04.

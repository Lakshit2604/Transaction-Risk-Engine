# Step 04: Feature Engineering

## What was done (plain English, 5-8 lines)
Engineered an object-oriented feature extraction pipeline in `src/features.py` implementing `FraudFeaturePipeline`.
Extracted cyclical temporal features (`hour_of_day`, `day_of_week`) from elapsed seconds in `TransactionDT`.
Engineered monetary anomaly features from `TransactionAmt`: log-scaled amount, cents decimal component, and whole-dollar flags.
Built synthetic card entity identifiers and calculated historical card frequency tables and monetary mean/deviation metrics strictly fit on `train_df`.
Created domain interaction features, including sender-to-recipient email domain matching (`email_domain_match`).
Encoded all categorical features using an ordinal encoder fitted purely on training data with unseen categories mapped safely to -1.
Persisted the fitted pipeline to `models/feature_pipeline.joblib` and exported engineered matrices (`train_features.parquet`, `val_features.parquet`, `test_features.parquet`).

## Why this step matters
Raw transaction attributes (like a raw card number or transaction timestamp) hold weak linear signals on their own. Fraud manifests in behavioral deviations—such as a $500 purchase on a card whose historical mean is $40, or foreign transactions with odd cents decimals occurring at 3 AM. Furthermore, feature engineering is where junior data science projects frequently fail due to target leakage. By strictly fitting all frequency distributions, group aggregations, and categorical mappings on `train_df` only, we ensure the pipeline can be safely applied to future out-of-time validation, holdout test, and live production API requests without any look-ahead leakage.

## Files created / changed (path + one-line purpose each)
- `src/features.py`: Production feature pipeline class `FraudFeaturePipeline` supporting batch and single-record API transformations.
- `models/feature_pipeline.joblib`: Serialized fitted feature pipeline artifact (750.8 KB) reusable by the FastAPI backend.
- `data/processed/train_features.parquet`: Engineered training feature matrix (413,378 rows × 435 columns, 59.3 MB).
- `data/processed/val_features.parquet`: Engineered validation feature matrix (88,581 rows × 435 columns, 12.5 MB).
- `data/processed/test_features.parquet`: Engineered holdout test feature matrix (88,581 rows × 435 columns, 13.4 MB).
- `tests/test_features.py`: Unit tests validating pipeline fitting, single-record transformation, and unseen category handling.
- `docs/explanations/04_feature_engineering.md`: Step documentation and interview prep guide.

## Key decisions and alternatives rejected
- **Fit on Train Only vs. Global Fitting:** Fitting encoders or aggregations across the whole dataset is a fatal mistake in fraud detection. Computing `card1_freq` or `card1_mean_amt` using the entire dataset leaks future transaction volume into the training set. Every mapping in `FraudFeaturePipeline` is learned strictly from `train_df`.
- **Ordinal / Integer Encoding with Unseen Handling vs. One-Hot Encoding:** One-hot encoding high-cardinality features (e.g. hundreds of email domains or thousands of device types) explodes the feature space into thousands of sparse columns, causing RAM exhaustion and slow tree training. Ordinal encoding maps categories to dense integers (with unseen categories assigned `-1`), which tree-based models like LightGBM natively leverage.
- **Cent Decimal Extraction (`TransactionAmt_decimal`):** In online payments, domestic purchases predominantly end in integer dollars (`$25.00`) or standard pricing (`$9.99`, `$0.50`). Cross-border currency conversions and automated testing scripts frequently produce unusual cent decimals (e.g. `$41.372`), providing a potent signal for fraud detection.
- **Unified Batch & Single-Record Interface (`transform_single`):** Often, ML engineers build complex Pandas pipelines in notebooks that cannot run on a single incoming JSON payload in production. By designing `FraudFeaturePipeline.transform_single(dict)` from day one, the exact same pipeline serves both batch training and real-time FastAPI endpoints.

## Results / outputs (numbers, plots, observations)
- **Engineered Feature Count:** 434 total features (added temporal, monetary relative, frequency, and interaction features).
- **Key Feature Categories Engineered:**
  - *Time:* `hour_of_day` (0-23), `day_of_week` (0-6).
  - *Amount:* `TransactionAmt_log`, `TransactionAmt_decimal`, `TransactionAmt_is_integer`.
  - *Entity & Match:* `card_id` (`card1_card2_card3_card5`), `email_domain_match`.
  - *Frequencies (train-fit):* `card1_freq`, `card_id_freq`, `addr1_freq`, `P_emaildomain_freq`.
  - *Card Aggregations (train-fit):* `amt_diff_card1_mean`, `amt_ratio_card1_mean`, `amt_zscore_card1`.
- **Batch Processing Throughput:** Transformed all 590,540 rows across Train, Val, and Test in under **11 seconds**.
- **Unit Tests:** 3 new feature engineering tests + 4 split tests = 7 passing tests in `pytest`.

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: How did you ensure your feature engineering didn't introduce data leakage?**
  *A:* I encapsulated all feature transformations into a stateful `FraudFeaturePipeline` class. The `.fit()` method is executed exclusively on the 70% chronological training split. All statistics—including card frequency tables, mean transaction amounts, and categorical label encodings—are learned strictly from historical train records. The `.transform()` method then applies these frozen lookup tables to validation, test, and production API requests without updating them on test data.
- **Q: What is the significance of the decimal part of `TransactionAmt`?**
  *A:* Legitimate domestic purchases in the US/Europe typically end in clean numbers (`.00`, `.50`, `.99`). In contrast, stolen foreign credit cards converted across exchange rates, or automated fraud bots running test authorizations, generate fragmented fractional decimals. Creating `TransactionAmt_decimal` and an `is_integer` flag exposes this subtle artifact directly to the tree splits.
- **Q: How did you create behavioral aggregate features for cards without leaking target information?**
  *A:* Rather than aggregating target fraud labels (target encoding, which easily overfits), I created behavioral feature aggregations—specifically `card1` mean transaction amount and frequency. For each transaction, we compute `amt_diff_card1_mean` (how much this purchase deviates from the card's typical spend) and `amt_ratio_card1_mean`. Unusually large transactions for a given card are a classic hallmark of account takeover.
- **Q: How do you handle unseen categories during production inference?**
  *A:* During training, our ordinal encoder builds an explicit dictionary of known category tokens. When an unseen category arrives at inference time (e.g. a brand-new device type or email domain not present in the training set), `.map().fillna(-1)` assigns it to an out-of-vocabulary token `-1`. This prevents runtime crashes and allows tree models to split on novelty.

## Limitations / what I would improve
- In a production streaming environment, card frequency and rolling velocity (e.g., number of swipes in past 10 minutes) would be maintained via a low-latency real-time feature store (such as Redis or Feast) rather than static historical tables.

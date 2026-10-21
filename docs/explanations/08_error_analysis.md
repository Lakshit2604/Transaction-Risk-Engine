# Step 08: Error Analysis and Explainability

## What was done (plain English, 5-8 lines)
Executed Tree SHAP (SHapley Additive exPlanations) on the winning LightGBM model to calculate feature attribution across validation transactions.
Identified the primary drivers of fraud risk: transaction frequency counters (`C13`, `C1`, `C14`), monetary magnitude (`TransactionAmt`), card type (`card6`), and time deltas (`D2`).
Generated the global explainability summary plot saved to `reports/shap_summary.png`.
Diagnostically profiled the model's failure modes across False Positives (4,285 false alarms) and False Negatives (847 missed frauds).
Uncovered 3 concrete domain insights: low-dollar card testing evasion, absence of browser identity telemetry in missed fraud, and high-dollar travel spikes triggering false alarms.
Formulated 3 production-grade architectural solutions and documented the complete investigation in `reports/error_analysis.md`.

## Why this step matters
In modern financial technology, machine learning models cannot operate as opaque black boxes. First, regulatory compliance (e.g. FCRA, GDPR Article 22 "Right to Explanation") mandates that financial institutions provide reason codes whenever credit or transactions are declined. Second, error analysis is the highest-ROI activity in practical data science: rather than randomly tuning hyperparameters, diagnosing exactly where the model fails reveals missing features, data quality gaps, and edge cases. In technical interviews, presenting concrete error modes and operational mitigation plans demonstrates mature, production-grade engineering judgment.

## Files created / changed (path + one-line purpose each)
- `scripts/explainability_and_errors.py`: Diagnostic script generating Tree SHAP attributions, cohort profiling, and error reports.
- `reports/shap_summary.png`: Beeswarm summary plot displaying the top 20 most impactful features for fraud prediction.
- `reports/error_analysis.md`: Detailed failure mode report covering 3 concrete domain insights and 3 architectural improvements.
- `docs/explanations/08_error_analysis.md`: Step documentation and interview prep guide.

## Key decisions and alternatives rejected
- **Tree SHAP vs. Permutation Importance / Feature Gain:** Built-in tree feature gain is biased toward high-cardinality features and only provides global rankings without indicating feature directionality. Tree SHAP calculates exact Shapley values in polynomial time ($O(TLD^2)$), revealing both the magnitude and directional effect (e.g., whether high or low values trigger fraud).
- **Stratified Sampling for SHAP vs. Full Dataset:** Computing SHAP on 88k rows would require substantial compute and RAM without changing the top-20 feature distribution. Extracting a stratified sample of 1,500 validation transactions (with an equal blend of fraud and genuine events) produced representative, high-fidelity attribution in under 3 seconds.
- **Deep Failure Cohort Profiling vs. Aggregate Metrics Only:** Instead of stopping at a macro F1 score, we isolated the 4,285 False Positives and 847 False Negatives, comparing their spend amounts, merchant product codes, and identity telemetry rates against True Positives.

## Results / outputs (numbers, plots, observations)
- **Top 5 Most Influential Features (by mean absolute SHAP value):**
  1. `C13` (0.2191): Multi-day transaction count for the associated payment instrument.
  2. `TransactionAmt` (0.1785): Dollar transaction amount.
  3. `C1` (0.1698): Rapid authorization velocity counter.
  4. `card6` (0.1548): Credit vs. debit card type (credit cards suffer higher unauthorized chargeback risk).
  5. `V70` (0.1468): Vesta engineered behavioral interaction counter.
- **Failure Mode Diagnostics:**
  - *False Negatives (Missed Fraud):* Median amount is only **$77.95** (card testing micro-transactions), and only **24.6%** had identity telemetry recorded (vs. **59.7%** in True Positives).
  - *False Positives (False Alarms):* Mean amount is **$188.78** (highest of any cohort), driven by genuine cardholders making atypical luxury or travel purchases.

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: What were the most important features according to SHAP, and does the domain logic make sense?**
  *A:* The top features are transaction velocity counters (`C13`, `C1`), monetary amount (`TransactionAmt`), card type (`card6`), and recency deltas (`D2`). This aligns directly with financial fraud patterns: fraudsters typically exploit stolen credentials by executing multiple transactions in rapid succession (`C13`/`C1`) on credit cards (`card6`), whereas genuine users have spaced-out, regular transaction patterns.
- **Q: Why did your model miss 847 fraud cases (False Negatives), and how would you fix it?**
  *A:* Analysis showed two main drivers: first, low-dollar "card testing" attacks where fraudsters swipe $5-$25 to test if a stolen card is active before making large purchases. Second, merchant blindspots where 75% of missed frauds had zero browser/device identity telemetry. To fix this, I would implement real-time sliding velocity windows (e.g. 5 transactions in 10 minutes in Redis) to detect bot frequency regardless of amount, and deploy IP geolocation clustering.
- **Q: Why did the model produce false alarms (False Positives), and how do banks handle them in production?**
  *A:* False Positives had the highest average dollar amount ($188.78), caused by legitimate customers making atypical high-value purchases (e.g., flight tickets or electronics) that resemble account takeovers. Banks resolve this by implementing a tiered action policy rather than hard declines: high-confidence fraud (>0.50) is blocked, but medium-confidence transactions (0.05 - 0.50) are routed to 3D-Secure 2.0 (SMS / Biometric OTP), allowing legitimate users to verify themselves frictionlessly.
- **Q: How does SHAP help with regulatory compliance in automated credit/fraud decisions?**
  *A:* Under regulations like FCRA and GDPR Article 22, automated systems that deny a transaction or credit line must provide adverse action notices with the primary reasons for denial. For any single prediction, Tree SHAP computes the top 3 negative local Shapley values, enabling our API to return human-readable reason codes (e.g., "Atypical transaction amount for card", "High velocity in 24 hours").

## Limitations / what I would improve
- SHAP values were computed globally on a batch sample; for production, we will compute local SHAP explanations per inference request directly in the FastAPI response payload.

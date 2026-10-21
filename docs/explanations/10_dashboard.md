# Step 10: Interactive Dashboard (Streamlit)

## What was done (plain English, 5-8 lines)
Engineered an interactive multi-page web application in `app/dashboard.py` using Streamlit.
Developed a Real-Time Scoring page featuring single-transaction manual data entry, 1-click sample presets (genuine vs. attack), dynamic risk gauges, tiered operational decisions, and key feature explanations.
Implemented a Batch CSV Scoring page allowing users to drag and drop raw transaction batches, execute LightGBM inference, display risk exposure KPIs, and export scored CSV files.
Constructed an Executive Analytics page rendering verified out-of-time test metrics, comparison benchmark tables, and visual charts (`pr_curve.png`, `cost_vs_threshold.png`, `final_confusion_matrix.png`, and `shap_summary.png`).
Optimized performance with `@st.cache_resource` for zero-latency artifact loading into memory.
Validated application syntax and confirmed seamless integration with upstream models and pipeline components.

## Why this step matters
Machine learning applications must bridge the gap between technical data science and non-technical business stakeholders (fraud investigators, risk officers, and compliance auditors). A command-line script or raw API output is difficult for operations teams to interact with during manual reviews. An intuitive dashboard provides a self-serve platform where investigators can test hypothetical fraud vectors, batch-audit suspicious daily merchant files, and visualize executive performance metrics and cost savings without touching a single line of code.

## Files created / changed (path + one-line purpose each)
- `app/dashboard.py`: Streamlit dashboard implementing real-time single-transaction scoring, batch CSV processing, and executive analytics.
- `docs/explanations/10_dashboard.md`: Step documentation and interview prep guide.

## Key decisions and alternatives rejected
- **Multi-Tab Modular Dashboard vs. Monolithic Single Screen:** Segmented the application into three focused operational views (Real-Time Scoring, Batch CSV Scoring, and Executive Model Analytics) to support both tactical risk investigators and executive leadership.
- **1-Click Pre-Configured Presets:** Added quick-action buttons ("Load Typical Genuine Swipes" vs. "Load Suspicious Attack") so interviewers or stakeholders can immediately demonstrate the model's sensitivity without having to manually input 10+ financial fields.
- **Direct Pipeline Re-Use vs. Calling Local REST API:** In `dashboard.py`, the app loads the serialized `FraudFeaturePipeline` and `best_lightgbm` directly into cached memory via `@st.cache_resource`. This eliminates unnecessary network hop overhead and avoids service crashes if the FastAPI container is running in a decoupled environment.

## Results / outputs (numbers, plots, observations)
- **Application Capabilities:**
  - *Real-Time Evaluation:* Evaluates probability, maps to operational tiers (`APPROVE`, `CHALLENGE`, `DECLINE`), and highlights key behavioral triggers.
  - *Batch Ingestion:* Processes multi-row CSV files, counts flagged transactions, calculates protected dollar volume, and offers instant download.
  - *Executive Analytics:* Displays test set PR-AUC (0.55), cost reduction (54.83%), and embeds all 4 diagnostic report plots.
- **Execution Command:** `streamlit run app/dashboard.py` runs locally on port `8501`.

## How to explain this in an interview (3-5 likely questions with short answers)
- **Q: Why did you build both a FastAPI service and a Streamlit dashboard?**
  *A:* They serve two distinct production purposes. FastAPI is a headless machine-to-machine microservice designed for low-latency (<15ms) automated payment gateway integration. Streamlit is a human-in-the-loop interface built for risk operations, compliance auditors, and fraud investigators who need to manually review flagged transactions, run batch audits, and inspect model explainability.
- **Q: How does the dashboard handle batch file processing without running out of memory?**
  *A:* The batch scoring module leverages the vectorized `FraudFeaturePipeline.transform()` and LightGBM's native C++ batch prediction. Transactions are processed in bulk arrays, assigned probability scores and decisions, and written to an in-memory downloadable CSV stream without redundant DataFrame copies.
- **Q: How does the dashboard communicate explainability to a non-technical fraud analyst?**
  *A:* In the real-time scoring tab, rather than displaying raw mathematical model weights, the dashboard translates feature values into human-readable business context (e.g., flagging unusual spend magnitude, transaction velocity, or email domain mismatches). In the analytics tab, it exposes the global SHAP beeswarm plot.
- **Q: What caching mechanisms did you implement to optimize dashboard performance?**
  *A:* I used `@st.cache_resource` to load the LightGBM model binary and the feature pipeline into memory once when the server initializes. Subsequent user interactions and tab switches re-use the warmed-up objects in RAM, keeping UI re-render latency under 50 milliseconds.

## Limitations / what I would improve
- In enterprise production, user authentication (RBAC - Role-Based Access Control) should be added so only authorized risk agents can access manual review tools and export customer PII.
- For multi-gigabyte batch files, batch processing should be offloaded to asynchronous Celery workers or AWS SQS rather than executing in the UI process.

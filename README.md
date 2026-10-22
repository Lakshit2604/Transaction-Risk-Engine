# Transaction Risk Engine: Production-Style Fraud Detection Pipeline

[![CI](https://github.com/Lakshit2604/Transaction-Risk-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Lakshit2604/Transaction-Risk-Engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-FFA500?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

An end-to-end, cost-sensitive fraud detection engine built on **590,540 real-world payment transactions** from the [IEEE-CIS Fraud Detection Competition](https://www.kaggle.com/c/ieee-fraud-detection). Engineered with strict chronological splits to prevent temporal leakage, decision thresholds chosen by minimising validation cost under an asymmetric penalty assumption ($Cost_{FN} = 10 \times Cost_{FP}$), explainable Tree SHAP feature attribution, a production-style FastAPI microservice, an operations Streamlit dashboard, and Docker containerization.

> **Dataset & Transparency Notice:**  
> The original dataset is hosted on Kaggle and is **not** redistributed in this repository. All API curl examples, dashboard presets, and test cases use purely **synthetic data**. AI assistance (Antigravity IDE) was utilized for pair-programming, scaffolding, and refactoring while the author directed, audited, verified, and can explain every technical decision, mathematical derivation, and line of code.

---

## 🌐 Live Demo

* 🖥️ **Interactive Operations Dashboard:** **[transaction-risk-engine.streamlit.app](https://transaction-risk-engine.streamlit.app/)**  
  *(Manual transaction inspection, 1-click synthetic presets, batch CSV auditing, and model analytics)*
* 📖 **Interactive API Documentation (Swagger UI):** **[transaction-risk-engine.onrender.com/docs](https://transaction-risk-engine.onrender.com/docs)**  
  *(Explore and test the live `/health` and `/predict` endpoints)*
* 💓 **API Health Endpoint:** **[transaction-risk-engine.onrender.com/health](https://transaction-risk-engine.onrender.com/health)**

> [!NOTE]  
> **Cold Start Advisory:** The API is hosted on Render's free tier, which spins down after 15 minutes of inactivity. If visiting after an idle period, the initial request or dashboard connection may take **30 to 60 seconds** to wake up the container. Subsequent requests execute normally.

---

## 📊 Benchmark & Evaluation Results

All metrics were computed strictly on the **out-of-time holdout test set (88,581 transactions)**, which was set aside at the beginning and touched strictly once for final reporting.

> **Cost Assumption:** The business cost assumes an asymmetric 10:1 penalty ratio ($Cost_{FN} = \$10.00$ for missed fraud, $Cost_{FP} = \$1.00$ for a false alert). All financial savings figures represent simulated savings under this assumed operating model.

### Comprehensive Test Set Comparison

| Model | Operating Threshold ($T$) | PR-AUC | ROC-AUC | Precision | Recall | F1 Score | False Positives (FP) | False Negatives (FN) | Simulated Cost ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | `0.5000` (Default) | 0.1284 | 0.7747 | 8.54% | 70.29% | 0.1522 | 23,218 | 916 | $32,378.00 |
| **Logistic Regression (Baseline)** | `0.6934` (Cost-Tuned) | 0.1284 | 0.7747 | 13.75% | 28.84% | 0.1862 | 5,576 | 2,194 | $27,516.00 |
| **LightGBM (Final Engine)** | `0.5000` (Default) | **0.5522** | **0.9128** | **80.01%** | 35.84% | **0.4951** | **276** | 1,978 | $20,056.00 |
| **LightGBM (Final Engine)** | `0.0543` (Cost-Tuned)* | **0.5522** | **0.9128** | 29.15% | **69.45%** | 0.4106 | 5,204 | **942** | **$14,624.00** |

*\*Threshold $T^* = 0.0543$ was chosen by minimising validation cost on the validation split.*

### Key Quantitative Takeaways
* **PR-AUC (Primary Metric):** Improved from **0.1284 to 0.5522 (+330.1% relative improvement)** over the logistic regression baseline.
* **Simulated Cost Reduction:** Reduced total simulated test loss from **$32,378.00 down to $14,624.00 (54.83% cost reduction)**, saving **$17,754.00** in simulated fraud losses compared to the default baseline. Even when comparing against the cost-tuned baseline ($27,516.00), LightGBM delivers a **46.85% cost reduction ($12,892.00 saved)**.
* **False Alarm Reduction:**
  - At the cost-tuned operating threshold ($T^* = 0.0543$), LightGBM produces **5,204 false alarms** compared to the baseline's **23,218 false alarms**—a **77.59% reduction in false positives (4.46x fewer false alarms)** while catching 2,141 frauds.
  - At the default threshold ($T = 0.5000$), false alarms drop by **98.81% (84.1x fewer false alarms)** from 23,218 to just 276.

---

## 📈 Cost-Ratio Sensitivity Analysis

Because business costs vary across payment gateways, we performed a cost-ratio sensitivity sweep across $5:1$, $10:1$, and $20:1$ cost penalty ratios on the validation split and evaluated the resulting thresholds on the test set. Full data is archived in [`reports/cost_sensitivity.json`](reports/cost_sensitivity.json).

| Cost Ratio ($Cost_{FN} : Cost_{FP}$) | Cost-Optimal Threshold ($T^*$) | LightGBM Test Recall | LightGBM Test Precision | Baseline Test Cost | LightGBM Test Cost | Simulated Savings | Cost Reduction (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **5 : 1** ($5 FN, $1 FP) | `0.1102` | 57.38% | 42.99% | $15,365.00 | $8,916.00 | $6,449.00 | **41.97%** |
| **10 : 1** ($10 FN, $1 FP) | `0.0611` | 67.27% | 31.19% | $27,516.00 | $14,666.00 | $12,850.00 | **46.70%** |
| **20 : 1** ($20 FN, $1 FP) | `0.0316` | 77.49% | 20.70% | $41,288.00 | $23,031.00 | $18,257.00 | **44.22%** |

*Observation:* As the penalty for missing a fraudulent transaction increases from 5x to 20x, the cost-minimizing threshold shifts lower from `0.1102` to `0.0316`, increasing recall from 57.4% to 77.5% at the expense of inspecting more false alerts.

---

## ⚡ Measured System Latency

Latency was benchmarked across two environments using synthetic payloads:

1. **Local Standalone Engine (Mac M-Series, Python 3.13):**
   - **Feature Transformation + LightGBM Inference:** **9.24 ms**
   - **Tree SHAP Top-3 Attribution:** **16.48 ms**
   - **Total Local Inference Compute:** **25.72 ms**

2. **Deployed Cloud Microservice (Render Free Tier, 20 Warm Calls over Public HTTPS):**
   - **Cold Start (Initial Wake-up from Idle Sleep):** ~35–50 seconds
   - **First Call (Post-Boot):** 415.95 ms
   - **Minimum Latency:** 453.59 ms
   - **Median Latency:** **555.47 ms**
   - **Mean Latency:** 618.27 ms
   - **p95 Latency:** **1,399.51 ms**  
   *(Cloud latency includes international internet round-trip network transit, TLS handshakes, FastAPI validation, LightGBM scoring, and Tree SHAP explanation generation).*

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[IEEE-CIS Dataset<br>590,540 Transactions] --> B[Data Cleaning & Memory Downcasting<br>-47.9% RAM Reduction]
        B --> C[Strict Chronological Split<br>Train: 70% | Val: 15% | Test: 15%]
    end

    subgraph Feature Engineering & Modeling
        C --> D[Leak-Free Feature Pipeline<br>Temporal, Aggregations, Ratios, Out-of-Fold Frequency]
        D --> E[Model Benchmarking<br>Logistic Reg vs Class Weights vs Undersampling vs LightGBM]
        E --> F[Validation Cost Optimization<br>T* chosen by minimising validation cost]
        F --> G[Explainability Layer<br>Tree SHAP Top-3 Local Risk Drivers]
    end

    subgraph Evaluation & Serving
        G --> H[Final Holdout Test Evaluation<br>PR-AUC: 0.5522 | Cost: $14,624]
        H --> I[Production-Style FastAPI Service<br>/predict with Pydantic v2 & Multi-Tier Decisions]
        H --> J[Streamlit Operations Dashboard<br>Synthetic Presets, Batch CSV Audit, Executive KPIs]
        I --> K[Docker & GitHub Actions CI<br>libgomp1 OpenMP Support & Automated Pytest]
        J --> K
    end
```

---

## 🔍 Model Explainability & Risk Drivers (Tree SHAP)

To provide transparency for payment operations, the engine computes Tree SHAP feature attributions for flagged transactions.

<p align="center">
  <img src="reports/shap_summary.png" alt="SHAP Beeswarm Feature Importance" width="750"/>
</p>

### Verified Top Feature Insights
* **Transaction Amount (`TransactionAmt`):** Non-linear risk distribution. Extreme transaction values (both micro-testing amounts <$3 and large luxury ticket sizes >$1,000) strongly increase fraud risk.
* **Card Type (`card6`):** Differentiates between **credit** and **debit** cards. Credit cards historically carry substantially higher chargeback and dispute exposure than debit cards. *(Note: Card network branding such as Visa/Mastercard is captured separately in `card4`).*
* **Anonymised Velocity Counters (`C1`–`C14`):** Proprietary count features representing frequency of associated cards, devices, and email interactions. Anonymised features `C13` and `C1` serve as strong indicators of rapid velocity anomalies.
* **Anonymised Behavioral Indicators (`V1`–`V339`):** Proprietary engineered behavioral features created by Vesta. Anonymised signals like `V70` reflect verification counts and device consistency.

---

## 📸 Dashboard & API Screenshots

*(Screenshots can be added to `docs/images/` for rich portfolio presentation)*

| Screenshot | Description | Target Path |
|:---|:---|:---|
| **Single-Transaction Scoring** | Real-time risk gauge, decision badge, and SHAP drivers | `docs/images/01_single_transaction_scoring.png` |
| **Batch CSV Scoring** | Batch file upload, risk breakdown KPIs, and CSV export | `docs/images/02_batch_scoring_page.png` |
| **Model Performance Analytics** | Diagnostic PR curves, cost curves, and confusion matrix | `docs/images/03_model_analytics_page.png` |
| **FastAPI Swagger Documentation** | Interactive OpenAPI docs and schema testing at `/docs` | `docs/images/04_api_swagger_docs.png` |

---

## ⚠️ Limitations & Future Work

* **Dataset Age (2019):** Trained on 2019 e-commerce transaction data. Modern fraud vectors (e.g., account takeover, synthetic identity fraud, crypto off-ramping) exhibit newer behavioral signatures.
* **Anonymised Features:** Over 300 columns (`V` and `C` series) are anonymised by the data provider, preventing domain-specific semantic feature engineering on those attributes.
* **Simulated Business Costs:** The 10:1 cost ratio ($Cost_{FN} = \$10, Cost_{FP} = \$1$) is an assumed heuristic. Production financial institutions dynamically price fraud costs by merchant category code (MCC), interchange rates, and dispute fees.
* **Single E-Commerce Dataset:** Evaluated on one payment provider dataset; performance on cross-border multi-currency transactions or instant bank transfers (e.g., FedNow, UPI) may require retraining.
* **Concept Drift & Live Streaming:** The current setup operates on batch data. Future work includes integrating Redis for sub-minute sliding window velocity features and Evidently AI for continuous covariate shift monitoring.
* **Free-Tier Cold Starts:** Deployed on free cloud tiers (Render), which hibernate after 15 minutes of inactivity and take 30–60 seconds to wake up. Enterprise deployments would run on auto-scaled Kubernetes (EKS/GKE) or AWS ECS behind an Application Load Balancer.

---

## 💻 Quickstart & Local Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Lakshit2604/Transaction-Risk-Engine.git
cd Transaction-Risk-Engine

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run FastAPI Service Locally
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Example synthetic test payload:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "TransactionAmt": 350.00,
       "ProductCD": "W",
       "card1": 13926,
       "card4": "visa",
       "card6": "credit",
       "P_emaildomain": "gmail.com",
       "C1": 2.0,
       "C13": 5.0
     }'
```

### 4. Run Streamlit Operations Dashboard Locally
```bash
streamlit run app/dashboard.py
```
Dashboard will open at [http://localhost:8501](http://localhost:8501).

### 5. Run with Docker Compose
```bash
docker compose up --build
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

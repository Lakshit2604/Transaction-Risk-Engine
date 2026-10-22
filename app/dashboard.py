"""Streamlit interactive fraud detection dashboard.

Features:
1. Live Transaction Risk Scorer (single transaction manual entry + 1-click synthetic presets).
2. Batch CSV Upload & Scoring (upload raw CSV, view risk distributions, export scored results).
3. Executive Analytics & Model Performance (PR-AUC, cost savings, PR curves, confusion matrix, and SHAP).
"""

import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="Transaction Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File system paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.features import FraudFeaturePipeline

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODELS_DIR / "best_lightgbm.joblib"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
FINAL_METRICS_PATH = REPORTS_DIR / "final_metrics.json"

# API Configuration from environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def check_api_health(url: str) -> Tuple[bool, Dict[str, Any]]:
    """Check connectivity to the FastAPI risk microservice."""
    try:
        resp = requests.get(f"{url}/health", timeout=3.0)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


@st.cache_resource
def load_local_artifacts():
    """Load local artifacts if present on disk for local evaluation or fallback."""
    if not MODEL_PATH.is_file() or not PIPELINE_PATH.is_file():
        return None, None, 0.0543

    model = joblib.load(MODEL_PATH)
    pipeline = FraudFeaturePipeline.load(PIPELINE_PATH)

    opt_threshold = 0.0543
    if FINAL_METRICS_PATH.is_file():
        with open(FINAL_METRICS_PATH, "r") as f:
            data = json.load(f)
            opt_threshold = data.get("optimal_cost_threshold_from_val", 0.0543)

    return model, pipeline, opt_threshold


local_model, local_pipeline, default_threshold = load_local_artifacts()

# -------------------------------------------------------------
# Sidebar: Connectivity & Controls
# -------------------------------------------------------------
st.sidebar.title("🛡️ Risk Engine Navigation")
app_mode = st.sidebar.radio(
    "Choose View:",
    ["Real-Time Scoring", "Batch CSV Scoring", "Model Analytics & Performance"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🔌 API Connection Status")
api_online, api_meta = check_api_health(API_URL)

if api_online:
    st.sidebar.success(f"🟢 **API Online** (`{API_URL}`)")
    operating_threshold = api_meta.get("operating_threshold", default_threshold)
else:
    st.sidebar.warning("🟡 **API Connecting / Sleeping**")
    st.sidebar.caption(
        f"Target: `{API_URL}`\n\n"
        "Free-tier cloud containers (Render) spin down after 15m of inactivity. "
        "A cold-start wake-up takes roughly 30 to 60 seconds."
    )
    if st.sidebar.button("🔄 Retry Connection"):
        st.rerun()
    operating_threshold = default_threshold

st.sidebar.markdown(f"**Operating Threshold ($T^*$):** `{operating_threshold:.4f}`")
st.sidebar.markdown("**Decision Policy:**")
st.sidebar.markdown(f"- 🟢 `P < {operating_threshold:.3f}`: **APPROVE**")
st.sidebar.markdown(f"- 🟡 `{operating_threshold:.3f} ≤ P < 0.50`: **CHALLENGE (OTP)**")
st.sidebar.markdown("- 🔴 `P ≥ 0.50`: **DECLINE (Block)**")
st.sidebar.markdown("---")
st.sidebar.info("Flagship ML Project: IEEE-CIS Transaction Fraud Detection on 590K records.")

# -------------------------------------------------------------
# TAB 1: Real-Time Scoring
# -------------------------------------------------------------
if app_mode == "Real-Time Scoring":
    st.title("⚡ Real-Time Transaction Risk Evaluation")
    st.markdown(
        "Evaluate incoming payment transactions in real-time. Features automated risk categorization, "
        "decisioning, and explainable Tree SHAP attribution drivers."
    )

    if not api_online and local_model is None:
        st.error(
            "⚠️ **API is currently offline and local model artifacts are not available.** "
            "If deployed to Render, please allow ~30–60 seconds for the free-tier container to wake up, "
            "then click **Retry Connection** in the sidebar."
        )

    # 1-Click Sample Previews (Synthetic data only, no real IEEE-CIS rows)
    st.subheader("1-Click Presets (Synthetic)")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])

    preset = None
    if col_btn1.button("🟢 Load Typical Genuine Swipes"):
        preset = {
            "TransactionAmt": 45.0,
            "ProductCD": "W",
            "card1": 10000,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "C1": 1.0,
            "C13": 2.0,
            "TransactionDT": 100000.0,
        }
    if col_btn2.button("🔴 Load Suspicious Attack"):
        preset = {
            "TransactionAmt": 1850.50,
            "ProductCD": "C",
            "card1": 95000,
            "card4": "credit",
            "card6": "credit",
            "P_emaildomain": "anonymous.com",
            "R_emaildomain": "mail.com",
            "C1": 45.0,
            "C13": 95.0,
            "TransactionDT": 15000000.0,
        }

    # Form inputs
    with st.form("single_tx_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 💳 Monetary & Product")
            amt = st.number_input(
                "Transaction Amount ($)",
                min_value=0.10,
                max_value=50000.0,
                value=preset["TransactionAmt"] if preset else 150.0,
                step=10.0,
            )
            product = st.selectbox(
                "Product Code",
                ["W", "C", "H", "R", "S"],
                index=["W", "C", "H", "R", "S"].index(preset["ProductCD"]) if preset else 0,
            )

        with col2:
            st.markdown("##### 🏦 Card Details")
            card1 = st.number_input(
                "Bank / Card ID (card1)",
                min_value=1000,
                max_value=99999,
                value=int(preset["card1"]) if preset else 13500,
            )
            card4 = st.selectbox(
                "Card Network (card4)",
                ["visa", "mastercard", "discover", "american express"],
                index=0 if not preset else (0 if preset["card4"] == "visa" else 1),
            )
            card6 = st.selectbox(
                "Card Type (card6)",
                ["debit", "credit"],
                index=0 if not preset else (0 if preset["card6"] == "debit" else 1),
            )

        with col3:
            st.markdown("##### 🌐 Network & Anonymised Velocity")
            p_email = st.text_input(
                "Purchaser Email Domain",
                value=preset["P_emaildomain"] if preset else "gmail.com",
            )
            r_email = st.text_input(
                "Recipient Email Domain",
                value=preset["R_emaildomain"] if preset else "gmail.com",
            )
            c1_val = st.number_input(
                "Velocity Counter (C1 - anonymised)",
                min_value=0.0,
                max_value=500.0,
                value=float(preset["C1"]) if preset else 1.0,
            )
            c13_val = st.number_input(
                "Spend Counter (C13 - anonymised)",
                min_value=0.0,
                max_value=500.0,
                value=float(preset["C13"]) if preset else 2.0,
            )

        submit = st.form_submit_button("⚡ Score Transaction Risk")

    if submit or preset:
        payload = {
            "TransactionID": 9999991,
            "TransactionDT": 12000000.0,
            "TransactionAmt": float(amt),
            "ProductCD": str(product),
            "card1": int(card1),
            "card4": str(card4),
            "card6": str(card6),
            "P_emaildomain": str(p_email),
            "R_emaildomain": str(r_email),
            "C1": float(c1_val),
            "C13": float(c13_val),
        }

        prob = None
        decision = None
        risk_level = None
        top_features = []

        # Mode A: Score via live REST API if available
        if api_online:
            try:
                with st.spinner("Scoring via Risk Engine Microservice..."):
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=12.0)
                if resp.status_code == 200:
                    api_data = resp.json()
                    prob = api_data["fraud_probability"]
                    decision = api_data["decision"]
                    risk_level = api_data["risk_level"]
                    top_features = api_data.get("top_contributing_features", [])
                else:
                    st.error(f"API returned status {resp.status_code}: {resp.text}")
            except Exception as e:
                st.warning(f"⚠️ Microservice request encountered an issue ({str(e)}). Attempting local fallback...")

        # Mode B: Fallback to local model execution if API unavailable
        if prob is None and local_model is not None and local_pipeline is not None:
            X_single = local_pipeline.transform_single(payload)
            prob = float(local_model.predict_proba(X_single)[:, 1][0])
            if prob >= 0.50:
                decision = "DECLINE"
                risk_level = "HIGH RISK (Decline)"
            elif prob >= operating_threshold:
                decision = "CHALLENGE"
                risk_level = "MEDIUM RISK (Step-Up 3D-Secure)"
            else:
                decision = "APPROVE"
                risk_level = "LOW RISK (Approved)"

        if prob is not None:
            st.markdown("---")
            st.subheader("🎯 Evaluation Result")

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Predicted Fraud Probability", f"{prob * 100:.2f}%")

            if decision == "DECLINE":
                res_col2.error("🚨 Decision: DECLINE")
                res_col3.metric("Risk Level", risk_level or "HIGH")
            elif decision == "CHALLENGE":
                res_col2.warning("⚠️ Decision: CHALLENGE (OTP)")
                res_col3.metric("Risk Level", risk_level or "MEDIUM")
            else:
                res_col2.success("✅ Decision: APPROVE")
                res_col3.metric("Risk Level", risk_level or "LOW")

            # Explainability / Drivers
            st.markdown("##### 🔍 Key Risk Drivers for this Transaction")
            if top_features:
                driver_rows = []
                for feat in top_features:
                    driver_rows.append({
                        "Feature": feat.get("feature"),
                        "SHAP Impact Score": f"+{feat.get('impact_score'):.4f}",
                        "Operational Explanation": feat.get("description"),
                    })
                st.table(pd.DataFrame(driver_rows))
            else:
                heuristic_reasons = []
                if amt > 500:
                    heuristic_reasons.append({"Feature": "Transaction Amount", "Detail": f"${amt:.2f}", "Role": "Elevated ticket size"})
                if c1_val > 10 or c13_val > 20:
                    heuristic_reasons.append({"Feature": "Velocity Counters", "Detail": f"C1={c1_val}, C13={c13_val}", "Role": "Elevated transaction velocity"})
                if p_email != r_email:
                    heuristic_reasons.append({"Feature": "Email Domain Mismatch", "Detail": f"{p_email} vs {r_email}", "Role": "Discrepancy between purchaser and recipient"})
                if card6 == "credit":
                    heuristic_reasons.append({"Feature": "Card Type", "Detail": "credit", "Role": "Higher historical chargeback exposure than debit"})
                if not heuristic_reasons:
                    heuristic_reasons.append({"Feature": "Baseline History", "Detail": "Normal", "Role": "No anomalous flags detected"})
                st.table(pd.DataFrame(heuristic_reasons))

# -------------------------------------------------------------
# TAB 2: Batch CSV Scoring
# -------------------------------------------------------------
elif app_mode == "Batch CSV Scoring":
    st.title("📂 Batch Transaction Audit & Scoring")
    st.markdown("Upload a CSV file containing unlabelled transactions to run high-throughput batch scoring.")

    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_batch):,} rows** and **{len(df_batch.columns)} columns**.")

            if st.button("🚀 Execute Batch Scoring"):
                if local_pipeline is not None and local_model is not None:
                    with st.spinner("Transforming features and executing LightGBM scoring..."):
                        X_batch = local_pipeline.transform(df_batch)
                        preds = local_model.predict_proba(X_batch)[:, 1]

                        df_scored = df_batch.copy()
                        df_scored["fraud_probability"] = np.round(preds, 4)
                        df_scored["decision"] = np.where(
                            preds >= 0.50,
                            "DECLINE",
                            np.where(preds >= operating_threshold, "CHALLENGE", "APPROVE"),
                        )

                    st.success("Batch scoring completed successfully!")

                    # Summary Metrics
                    b_col1, b_col2, b_col3 = st.columns(3)
                    n_declined = int((df_scored["decision"] == "DECLINE").sum())
                    n_challenge = int((df_scored["decision"] == "CHALLENGE").sum())
                    n_approved = int((df_scored["decision"] == "APPROVE").sum())

                    b_col1.metric("Blocked / Declined", f"{n_declined:,} ({(n_declined/len(df_scored))*100:.1f}%)")
                    b_col2.metric("Challenged (3D-Secure)", f"{n_challenge:,} ({(n_challenge/len(df_scored))*100:.1f}%)")
                    b_col3.metric("Auto-Approved", f"{n_approved:,} ({(n_approved/len(df_scored))*100:.1f}%)")

                    st.dataframe(df_scored.head(50))

                    csv_data = df_scored.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Scored CSV Report",
                        data=csv_data,
                        file_name="fraud_scored_transactions.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Batch scoring currently requires local model artifacts in models/.")
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# -------------------------------------------------------------
# TAB 3: Model Analytics & Performance
# -------------------------------------------------------------
elif app_mode == "Model Analytics & Performance":
    st.title("📊 Production Model Performance & Analytics")
    st.markdown(
        "Verified evaluation metrics on the holdout test set (88,581 transactions), "
        "demonstrating the business and statistical improvements over the baseline."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("PR-AUC (Primary)", "0.5522", "+330.1% vs Baseline")
    kpi2.metric("ROC-AUC", "0.9128", "+17.8% vs Baseline")
    kpi3.metric("Cost Reduction", "54.83%", "$17,754 Saved")
    kpi4.metric("Operating Cutoff ($T^*$)", f"{operating_threshold:.4f}", "Cost-Optimized")

    st.markdown("---")
    st.subheader("📈 Diagnostic Performance Curves")

    tab_pr, tab_cost, tab_cm, tab_shap = st.tabs([
        "PR Curve", "Cost vs. Threshold", "Confusion Matrix", "SHAP Feature Importance"
    ])

    with tab_pr:
        pr_path = REPORTS_DIR / "pr_curve.png"
        if pr_path.is_file():
            st.image(str(pr_path), caption="Precision-Recall Curve on Holdout Test Set")
        else:
            st.info("PR curve plot not found in reports/")

    with tab_cost:
        cost_path = REPORTS_DIR / "cost_vs_threshold.png"
        if cost_path.is_file():
            st.image(str(cost_path), caption="Simulated Business Cost vs Decision Threshold")
        else:
            st.info("Cost curve plot not found in reports/")

    with tab_cm:
        cm_path = REPORTS_DIR / "final_confusion_matrix.png"
        if cm_path.is_file():
            st.image(str(cm_path), caption="Confusion Matrix at Optimal Operating Threshold")
        else:
            st.info("Confusion matrix plot not found in reports/")

    with tab_shap:
        shap_path = REPORTS_DIR / "shap_summary.png"
        if shap_path.is_file():
            st.image(str(shap_path), caption="Tree SHAP Beeswarm Feature Impact Summary")
        else:
            st.info("SHAP plot not found in reports/")

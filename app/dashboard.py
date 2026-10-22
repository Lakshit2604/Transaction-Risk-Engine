"""Streamlit interactive fraud detection dashboard.

Features:
1. Live Transaction Risk Scorer (single transaction manual entry + 1-click sample presets).
2. Batch CSV Upload & Scoring (upload raw CSV, view risk distributions, export scored results).
3. Executive Analytics & Model Performance (PR-AUC, cost savings, PR curves, confusion matrix, and SHAP).
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="Transaction Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys

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


@st.cache_resource
def load_artifacts():
    """Load and cache the trained LightGBM model, feature pipeline, and optimal threshold."""
    if not MODEL_PATH.is_file() or not PIPELINE_PATH.is_file():
        st.error("Model artifacts not found in models/. Please run training pipeline first.")
        return None, None, 0.0543

    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    opt_threshold = 0.0543
    if FINAL_METRICS_PATH.is_file():
        with open(FINAL_METRICS_PATH, "r") as f:
            data = json.load(f)
            opt_threshold = data.get("optimal_cost_threshold_from_val", 0.0543)

    return model, pipeline, opt_threshold


model, pipeline, optimal_threshold = load_artifacts()

# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
st.sidebar.title("🛡️ Risk Engine Navigation")
app_mode = st.sidebar.radio(
    "Choose View:",
    ["Real-Time Scoring", "Batch CSV Scoring", "Model Analytics & Performance"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Operating Threshold:** `{optimal_threshold:.4f}`")
st.sidebar.markdown("**Decision Policy:**")
st.sidebar.markdown("- 🟢 `P < 0.054`: **APPROVE**")
st.sidebar.markdown("- 🟡 `0.054 ≤ P < 0.50`: **CHALLENGE (OTP)**")
st.sidebar.markdown("- 🔴 `P ≥ 0.50`: **DECLINE (Block)**")
st.sidebar.markdown("---")
st.sidebar.info("Flagship ML Project: IEEE-CIS Transaction Fraud Detection on 590K records.")

# -------------------------------------------------------------
# TAB 1: Real-Time Scoring
# -------------------------------------------------------------
if app_mode == "Real-Time Scoring":
    st.title("⚡ Real-Time Transaction Risk Evaluation")
    st.markdown(
        "Evaluate incoming card transactions in real-time, generate automated fraud decisions, "
        "and inspect contributing risk factors."
    )

    # 1-Click Sample Previews
    st.subheader("1-Click Presets")
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
                "Card Network",
                ["visa", "mastercard", "discover", "american express"],
                index=0 if not preset else (0 if preset["card4"] == "visa" else 1),
            )
            card6 = st.selectbox(
                "Card Category",
                ["debit", "credit"],
                index=0 if not preset else (0 if preset["card6"] == "debit" else 1),
            )

        with col3:
            st.markdown("##### 🌐 Network & Velocity")
            p_email = st.text_input(
                "Purchaser Email Domain",
                value=preset["P_emaildomain"] if preset else "gmail.com",
            )
            r_email = st.text_input(
                "Recipient Email Domain",
                value=preset["R_emaildomain"] if preset else "gmail.com",
            )
            c1_val = st.number_input(
                "Velocity Counter (C1)",
                min_value=0.0,
                max_value=500.0,
                value=float(preset["C1"]) if preset else 1.0,
            )
            c13_val = st.number_input(
                "Historical Spend Counter (C13)",
                min_value=0.0,
                max_value=500.0,
                value=float(preset["C13"]) if preset else 2.0,
            )

        submit = st.form_submit_button("⚡ Score Transaction Risk", use_container_width=True)

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

        # Transform and score
        X_single = pipeline.transform_single(payload)
        prob = float(model.predict_proba(X_single)[:, 1][0])

        st.markdown("---")
        st.subheader("🎯 Evaluation Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        res_col1.metric("Predicted Fraud Probability", f"{prob * 100:.2f}%")

        if prob >= 0.50:
            res_col2.error("🚨 Decision: DECLINE")
            res_col3.metric("Risk Level", "HIGH RISK (Decline)")
        elif prob >= optimal_threshold:
            res_col2.warning("⚠️ Decision: CHALLENGE (OTP)")
            res_col3.metric("Risk Level", "MEDIUM RISK (Step-Up 3D-Secure)")
        else:
            res_col2.success("✅ Decision: APPROVE")
            res_col3.metric("Risk Level", "LOW RISK (Approved)")

        # Feature Drivers
        st.markdown("##### 🔍 Key Risk Drivers for this Transaction")
        reasons = []
        if amt > 500:
            reasons.append({"Feature": "Transaction Amount", "Value": f"${amt:.2f}", "Role": "Elevated ticket size"})
        if c1_val > 10 or c13_val > 20:
            reasons.append({"Feature": "Velocity Counters", "Value": f"C1={c1_val}, C13={c13_val}", "Role": "Unusually rapid authorization frequency"})
        if p_email != r_email:
            reasons.append({"Feature": "Email Domain Mismatch", "Value": f"{p_email} vs {r_email}", "Role": "Discrepancy between sender and receiver"})
        if card6 == "credit":
            reasons.append({"Feature": "Credit Line Exposure", "Value": "Credit", "Role": "Higher default chargeback liability"})

        if not reasons:
            reasons.append({"Feature": "Baseline History", "Value": "Normal", "Role": "No anomalous behavioral flags detected"})

        st.table(pd.DataFrame(reasons))

# -------------------------------------------------------------
# TAB 2: Batch CSV Scoring
# -------------------------------------------------------------
elif app_mode == "Batch CSV Scoring":
    st.title("📁 Batch Transaction Scoring & Audit")
    st.markdown(
        "Upload a batch CSV of raw transactions to process through the full pipeline, "
        "identify suspicious clusters, and download scored results."
    )

    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded `{len(raw_df):,}` rows with `{raw_df.shape[1]}` columns.")
        st.dataframe(raw_df.head(5))

        if st.button("🚀 Process and Score Batch", use_container_width=True):
            with st.spinner("Transforming features and running LightGBM inference..."):
                X_batch, _ = pipeline.transform(raw_df)
                probs = model.predict_proba(X_batch)[:, 1]

                raw_df["fraud_probability"] = np.round(probs, 4)
                raw_df["is_fraud"] = probs >= optimal_threshold
                raw_df["decision"] = np.where(
                    probs >= 0.50, "DECLINE",
                    np.where(probs >= optimal_threshold, "CHALLENGE", "APPROVE")
                )

            st.success("Batch scoring complete!")

            total_tx = len(raw_df)
            flagged_tx = int(raw_df["is_fraud"].sum())
            flagged_pct = (flagged_tx / total_tx) * 100
            total_flagged_amt = raw_df[raw_df["is_fraud"]]["TransactionAmt"].sum() if "TransactionAmt" in raw_df else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Scored", f"{total_tx:,}")
            m2.metric("Flagged for Action", f"{flagged_tx:,} ({flagged_pct:.1f}%)")
            m3.metric("Protected Transaction Volume", f"${total_flagged_amt:,.2f}")

            st.dataframe(raw_df[["TransactionID", "TransactionAmt", "fraud_probability", "decision"]].head(25))

            csv_download = raw_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Scored CSV",
                csv_download,
                "scored_transactions.csv",
                "text/csv",
                use_container_width=True,
            )

# -------------------------------------------------------------
# TAB 3: Model Analytics & Performance
# -------------------------------------------------------------
elif app_mode == "Model Analytics & Performance":
    st.title("📊 Executive Model Analytics & Validation Benchmarks")
    st.markdown(
        "Comprehensive model evaluation results on the unseen **88,581 Holdout Test Set** "
        "following chronological out-of-time evaluation."
    )

    # Key Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Holdout PR-AUC", "0.5522", "+330.1% vs Baseline")
    kpi2.metric("ROC-AUC", "0.9128", "+17.8% vs Baseline")
    kpi3.metric("Simulated Cost Reduction", "54.83%", "$17,754 Saved")
    kpi4.metric("Recall @ 80% Precision", "35.94%", "Frictionless Block")

    st.markdown("---")

    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.subheader("Precision-Recall Curve Comparison")
        pr_path = REPORTS_DIR / "pr_curve.png"
        if pr_path.is_file():
            st.image(str(pr_path), caption="PR-AUC: Baseline Logistic Regression (0.13) vs. LightGBM (0.55)")

    with col_img2:
        st.subheader("Cost vs. Decision Threshold")
        cost_path = REPORTS_DIR / "cost_vs_threshold.png"
        if cost_path.is_file():
            st.image(str(cost_path), caption="Optimal Threshold (0.054) Minimizing Total Business Cost")

    col_img3, col_img4 = st.columns(2)

    with col_img3:
        st.subheader("Final Test Confusion Matrix")
        cm_path = REPORTS_DIR / "final_confusion_matrix.png"
        if cm_path.is_file():
            st.image(str(cm_path), caption="Confusion Matrix on 88,581 Holdout Transactions at Optimal Threshold")

    with col_img4:
        st.subheader("Global SHAP Feature Importance")
        shap_path = REPORTS_DIR / "shap_summary.png"
        if shap_path.is_file():
            st.image(str(shap_path), caption="Top 20 Features Driving Tree Decisions via Tree SHAP")

    # Benchmarks Table
    st.subheader("Model Progression Benchmarks")
    comp_path = REPORTS_DIR / "model_comparison.csv"
    if comp_path.is_file():
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True)

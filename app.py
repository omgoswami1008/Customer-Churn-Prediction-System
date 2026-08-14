"""
Project: Customer Churn Prediction System
App    : Streamlit dashboard (professional design)

Pages
-----
1. Home             - project overview, KPIs, example scenarios.
2. Churn Predictor  - live risk score for a single customer.
3. Data Explorer    - interactive EDA on the cleaned dataset.
4. Model Report     - holdout metrics and saved figures.
5. About            - pipeline, stack and deployment guide.

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_customer  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook", palette="Set2")

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RISK_LOW, RISK_HIGH = 0.30, 0.60

PRIMARY = "#2b6cb0"
ACCENT = "#3182ce"
DANGER = "#e53e3e"
SUCCESS = "#38a169"
WARNING = "#dd6b20"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stSidebar"], .stApp {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}

/* Hide default Streamlit chrome for a clean, branded look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* KPI metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(26,32,44,0.05);
}
div[data-testid="stMetricLabel"] { color: #718096; font-weight: 600; }
div[data-testid="stMetricValue"] {
    font-size: 1.65rem; font-weight: 800; color: #1a202c;
}
div[data-testid="stMetricDelta"] { color: #2b6cb0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 6px 16px; font-weight: 600;
}

/* Buttons */
.stButton button, .stFormSubmitButton button {
    border-radius: 10px; font-weight: 600;
}
.stButton button[kind="primary"],
.stFormSubmitButton button[kind="primary"] {
    background: linear-gradient(120deg, #2b6cb0, #3182ce);
    border: none;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="background:linear-gradient(120deg,#1a365d,#2b6cb0 55%,#3182ce);
                    border-radius:16px;padding:26px 32px;color:white;margin-bottom:20px;
                    box-shadow:0 6px 20px rgba(43,108,176,0.25);">
          <div style="font-size:1.9rem;font-weight:800;letter-spacing:-0.5px;">{title}</div>
          <div style="opacity:0.94;font-size:1.02rem;margin-top:6px;max-width:820px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(icon: str, title: str, body: str, color: str = PRIMARY) -> None:
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-top:4px solid {color};
                    border-radius:14px;padding:16px 18px;height:100%;
                    box-shadow:0 2px 8px rgba(26,32,44,0.05);">
          <div style="font-size:1.5rem;">{icon}</div>
          <div style="font-weight:700;font-size:1.02rem;margin:6px 0 4px;color:#1a202c;">{title}</div>
          <div style="color:#4a5568;font-size:0.9rem;line-height:1.55;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <div style="text-align:center;color:#a0aec0;font-size:0.82rem;margin-top:48px;
                    border-top:1px solid #e2e8f0;padding-top:16px;">
          📡 Customer Churn Prediction System &nbsp;·&nbsp; Python · Scikit-learn ·
          Streamlit &nbsp;·&nbsp; Built as a professional portfolio project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    if not CLEAN_DATA_PATH.exists():
        st.error("`data/processed/clean_data.csv` not found. Run the pipeline first (see About).")
        st.stop()
    return pd.read_csv(CLEAN_DATA_PATH)


def load_figure(name: str) -> Path | None:
    path = FIGURES_DIR / name
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Shared UI bits
# ---------------------------------------------------------------------------

def risk_badge(prob: float) -> None:
    if prob < RISK_LOW:
        level, color = "Low", SUCCESS
    elif prob < RISK_HIGH:
        level, color = "Medium", WARNING
    else:
        level, color = "High", DANGER
    st.markdown(
        f"""
        <div style="background:{color};color:white;border-radius:12px;padding:14px;
                    text-align:center;font-size:1.15rem;font-weight:700;
                    box-shadow:0 4px 12px {color}55;">
          Risk level: {level}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return level


# ---------------------------------------------------------------------------
# Page 1 - Home
# ---------------------------------------------------------------------------

PREFILLS = {
    "high_risk": {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 79.95,
        "TotalCharges": 159.90,
    },
    "low_risk": {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes",
        "Dependents": "Yes", "tenure": 71, "PhoneService": "Yes",
        "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 92.45,
        "TotalCharges": 6563.95,
    },
}


def goto_predictor(prefill_key: str) -> None:
    st.session_state["prefill"] = PREFILLS[prefill_key]
    st.session_state["nav"] = "🎯 Churn Predictor"
    st.rerun()


def run_home_page() -> None:
    hero(
        "📡 Customer Churn Prediction System",
        "A production-style machine-learning project that predicts whether a telecom "
        "customer will churn, using a cross-validated Random Forest. Explore the data, "
        "score customers in real time and download a full model report.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", "7,043")
    c2.metric("Churn rate", "26.5%")
    c3.metric("Holdout ROC-AUC", "0.842")
    c4.metric("Model", "Random Forest")

    st.markdown("### Why it matters")
    col1, col2, col3 = st.columns(3)
    with col1:
        card("💰", "Acquisition is costly",
             "Winning a new customer can cost 5–7x more than keeping an existing one. "
             "Catching churn early protects revenue.", DANGER)
    with col2:
        card("🎯", "Targeted retention",
             "The model flags at-risk accounts so the team can focus offers and support "
             "where they matter most.", ACCENT)
    with col3:
        card("📈", "Explainable signals",
             "Feature importance shows what drives churn: short tenure, month-to-month "
             "contracts and missing add-ons.", SUCCESS)

    st.markdown("### Try a live prediction")
    st.markdown(
        "Jump straight to the predictor with a realistic profile — press a button to "
        "see the form pre-filled."
    )
    b1, b2, _ = st.columns([1, 1, 2])
    with b1:
        if st.button("🔴 High-risk example", use_container_width=True):
            goto_predictor("high_risk")
    with b2:
        if st.button("🟢 Low-risk example", use_container_width=True):
            goto_predictor("low_risk")


# ---------------------------------------------------------------------------
# Page 2 - Predictor
# ---------------------------------------------------------------------------

def build_recommendations(customer: dict) -> list[str]:
    """Turn the customer's answers into targeted retention actions."""
    recs = []

    if customer["Contract"] == "Month-to-month":
        recs.append("Offer a 12/24-month contract with a discount to lock in loyalty.")
    if customer["InternetService"] == "Fiber optic":
        recs.append("Fiber-optic customers churn more often — check service quality / speed complaints.")
    if customer["TechSupport"] in ("No", "No internet service"):
        recs.append("Offer a free Tech Support add-on trial.")
    if customer["OnlineSecurity"] in ("No", "No internet service"):
        recs.append("Promote the Online Security add-on, a proven churn-reducer.")
    if customer["OnlineBackup"] in ("No", "No internet service"):
        recs.append("Cross-sell the Online Backup add-on.")
    if customer["tenure"] < 12:
        recs.append("New customers (< 12 months) are the riskiest — schedule a welcome call / onboarding.")
    if customer["PaymentMethod"] == "Electronic check":
        recs.append("Electronic-check payers churn more — encourage auto-pay / credit-card billing.")
    if customer["PaperlessBilling"] == "Yes" and customer["SeniorCitizen"] == "Yes":
        recs.append("Senior customers on paperless billing may miss bills — consider a monthly summary.")

    return recs if recs else [
        "Few classic risk signals detected — keep the customer engaged with standard retention."
    ]


def run_predictor_page() -> None:
    hero(
        "🎯 Churn Risk Predictor",
        "Enter a customer's profile to estimate the probability they will churn. "
        "The result includes a risk band and targeted retention recommendations.",
    )

    # Optional pre-fill from the Home page example buttons.
    prefill = st.session_state.pop("prefill", None)
    if prefill:
        for key, value in prefill.items():
            st.session_state[f"pf_{key}"] = value

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("👤 Customer profile")
        with st.form("churn_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                gender = st.radio("Gender", ["Female", "Male"], key="pf_gender",
                                  horizontal=True)
                senior = st.radio("Senior citizen", ["No", "Yes"], key="pf_SeniorCitizen",
                                  horizontal=True)
                partner = st.radio("Partner", ["No", "Yes"], key="pf_Partner",
                                   horizontal=True)
                dependents = st.radio("Dependents", ["No", "Yes"], key="pf_Dependents",
                                      horizontal=True)
                tenure = st.number_input("Tenure (months)", 0, 72, 12, step=1,
                                         key="pf_tenure")

            with c2:
                phone = st.radio("Phone service", ["No", "Yes"], key="pf_PhoneService",
                                 horizontal=True)
                multiple_lines = st.selectbox(
                    "Multiple lines",
                    ["No", "Yes", "No phone service"],
                    key="pf_MultipleLines",
                    help="Only relevant if phone service is on.",
                )
                internet = st.selectbox(
                    "Internet service", ["DSL", "Fiber optic", "No"],
                    key="pf_InternetService",
                )
                paperless = st.radio("Paperless billing", ["No", "Yes"],
                                     key="pf_PaperlessBilling", horizontal=True)
                monthly = st.number_input("Monthly charges (USD)", 0.0, 200.0, 60.0,
                                          step=1.0, key="pf_MonthlyCharges")

            with c3:
                opts = ["No", "Yes", "No internet service"]
                online_security = st.selectbox("Online security", opts, key="pf_OnlineSecurity")
                online_backup = st.selectbox("Online backup", opts, key="pf_OnlineBackup")
                device_prot = st.selectbox("Device protection", opts, key="pf_DeviceProtection")
                tech_support = st.selectbox("Tech support", opts, key="pf_TechSupport")
                streaming_tv = st.selectbox("Streaming TV", opts, key="pf_StreamingTV")
                streaming_movies = st.selectbox("Streaming movies", opts, key="pf_StreamingMovies")

            c4, c5 = st.columns(2)
            with c4:
                contract = st.selectbox(
                    "Contract", ["Month-to-month", "One year", "Two year"],
                    key="pf_Contract",
                )
            with c5:
                payment = st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"],
                    key="pf_PaymentMethod",
                )

            total = st.slider(
                "Total charges (USD)", 0.0, 9000.0, 0.0, step=10.0,
                key="pf_TotalCharges",
                help="Lifetime total billed to date.",
            )

            submitted = st.form_submit_button(
                "🚀 Predict churn risk", type="primary", width="stretch"
            )

    if not submitted:
        with right:
            card("💡", "How it works",
                 "Fill in the profile on the left and press **Predict churn risk**. "
                 "The model scores 28 engineered features and returns a probability, "
                 "a risk band and actionable retention steps.",
                 ACCENT)
            st.markdown("---")
            st.caption("**Model:** tuned Random Forest · 5-fold CV · holdout ROC-AUC ≈ 0.84")
        footer()
        return

    customer = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone,
        "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_prot, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }

    try:
        result = predict_customer(customer)
    except Exception as exc:  # artifacts missing etc.
        st.error(f"Prediction failed: {exc}. Re-run the training pipeline (see About).")
        footer()
        return

    with right:
        prob = result["churn_probability"]
        risk = result["churn_risk"]

        st.subheader("🎯 Risk score")
        st.metric("Churn probability", f"{prob * 100:.1f}%")
        st.progress(prob, text=f"Probability of churn: {prob * 100:.1f}%")
        risk_badge(prob)

        pred = "⚠️ At risk of churning" if result["predicted_class"] == 1 else "✅ Likely to stay"
        st.markdown(f"**Decision (0.5 threshold):** {pred}")

        st.markdown("---")
        st.subheader("📋 Retention recommendations")
        for rec in build_recommendations(customer):
            st.markdown(f"- {rec}")

    footer()


# ---------------------------------------------------------------------------
# Page 3 - Data Explorer
# ---------------------------------------------------------------------------

def churn_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby(col)["Churn"]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
    )


def run_eda_page() -> None:
    hero(
        "📊 Data Explorer",
        "Explore the cleaned telecom dataset: churn behaviour by contract, service, "
        "tenure and charges.",
    )
    df = load_clean_data()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(df):,}")
    m2.metric("Churn rate", f"{100 * (df['Churn'] == 'Yes').mean():.1f}%")
    m3.metric("Avg tenure", f"{df['tenure'].mean():.0f} mo")
    m4.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")

    st.markdown("### Churn by contract & payment")
    col1, col2 = st.columns(2)
    with col1:
        rate = churn_rate(df, "Contract")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.barplot(data=rate, x="Contract", y="rate", hue="Churn", ax=ax)
        ax.set_title("Churn rate by contract")
        ax.set_ylabel("Share of customers")
        ax.legend(title="Churn")
        st.pyplot(fig); plt.close(fig)
    with col2:
        rate = churn_rate(df, "PaymentMethod")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.barplot(data=rate, x="PaymentMethod", y="rate", hue="Churn", ax=ax)
        ax.set_title("Churn rate by payment method")
        ax.set_ylabel("Share of customers")
        ax.tick_params(axis="x", rotation=18)
        ax.legend(title="Churn")
        st.pyplot(fig); plt.close(fig)

    st.markdown("### Tenure & charges")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.histplot(data=df, x="tenure", hue="Churn", multiple="dodge", bins=20, ax=ax)
        ax.set_title("Tenure distribution by churn")
        st.pyplot(fig); plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.histplot(data=df, x="MonthlyCharges", hue="Churn", multiple="dodge", bins=20, ax=ax)
        ax.set_title("Monthly charges by churn")
        st.pyplot(fig); plt.close(fig)

    with st.expander("🔎 Browse the cleaned dataset"):
        st.dataframe(df.head(100), width="stretch")
        st.caption(f"Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    footer()


# ---------------------------------------------------------------------------
# Page 4 - Model Report
# ---------------------------------------------------------------------------

def run_report_page() -> None:
    hero(
        "📈 Model Report",
        "Holdout evaluation (1,405 customers) for the tuned Random Forest, selected "
        "via 5-fold cross-validation on ROC-AUC.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", "0.842")
    c2.metric("PR-AUC", "0.658")
    c3.metric("Accuracy", "0.774")
    c4.metric("F1-score", "0.631")

    st.markdown(
        "> At the optimal decision threshold (**0.41**, chosen with Youden's J), "
        "recall rises to **0.81** — the model flags most at-risk customers so "
        "retention teams can act early."
    )

    st.markdown("### Visual results")
    fig_names = {
        "Confusion matrix": "confusion_matrix.png",
        "ROC curve": "roc_curve.png",
        "Precision-Recall curve": "pr_curve.png",
        "Feature importance": "feature_importance.png",
    }
    col1, col2 = st.columns(2)
    for (title, name), col in zip(fig_names.items(), [col1, col2] * 2):
        path = load_figure(name)
        with col:
            st.subheader(title)
            if path:
                st.image(str(path), width="stretch")
            else:
                st.warning(f"{name} not generated yet. Run `python -m src.evaluate`.")

    pdf = PROJECT_ROOT / "reports" / "model_report.pdf"
    if pdf.exists():
        with open(pdf, "rb") as f:
            st.download_button(
                "⬇️ Download full PDF report", f.read(),
                file_name="model_report.pdf", mime="application/pdf",
            )

    footer()


# ---------------------------------------------------------------------------
# Page 5 - About
# ---------------------------------------------------------------------------

def run_about_page() -> None:
    hero(
        "ℹ️ About the project",
        "A complete, modular machine-learning pipeline — from raw data to a deployed "
        "interactive dashboard.",
    )

    col1, col2 = st.columns(2)
    with col1:
        card("🧬", "Pipeline",
             "Six clean modules (`data_loader` → `preprocessing` → `feature_engineering` "
             "→ `train_model` → `evaluate` → `predict`) plus this Streamlit app.", PRIMARY)
        card("🛠", "Tech stack",
             "Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · "
             "Streamlit · Joblib.", ACCENT)
    with col2:
        card("🎯", "Key results",
             "Holdout ROC-AUC **0.842**. Balanced class weights handle the 26% churn "
             "imbalance; Youden's J threshold lifts recall to 0.81.", SUCCESS)
        card("🚀", "Deployment",
             "Free **Streamlit Community Cloud** via GitHub. Push the repo, connect it "
             "in the dashboard, and the app goes live in minutes.", DANGER)

    st.markdown("### How to re-run everything")
    st.code(
        "python -m src.preprocessing       # clean data\n"
        "python -m src.feature_engineering # encode features\n"
        "python -m src.train_model         # tune + train model\n"
        "python -m src.evaluate            # metrics + PDF report\n"
        "streamlit run app.py              # launch dashboard"
    )

    st.markdown("### Repository structure")
    st.code(
        "customer-churn-prediction/\n"
        "├── data/raw|processed/       datasets\n"
        "├── notebooks/                EDA notebooks\n"
        "├── src/                      modular pipeline (6 modules)\n"
        "├── models/                   trained artifacts\n"
        "├── reports/figures/          evaluation outputs\n"
        "├── .streamlit/config.toml    app theme\n"
        "├── app.py                    Streamlit dashboard\n"
        "├── requirements.txt          pinned dependencies\n"
        "└── README.md"
    )

    footer()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PAGES = {
    "🏠 Home": run_home_page,
    "🎯 Churn Predictor": run_predictor_page,
    "📊 Data Explorer": run_eda_page,
    "📈 Model Report": run_report_page,
    "ℹ️ About": run_about_page,
}


def main() -> None:
    options = list(PAGES.keys())

    # Persist the current page across reruns (widget state + programmatic nav).
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = options[0]
    nav = st.session_state.pop("nav", None)
    if nav in options:
        st.session_state["current_page"] = nav

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:8px 0 12px;">
              <div style="font-size:2rem;">📡</div>
              <div style="font-weight:800;font-size:1.1rem;color:#1a202c;">
                Churn Prediction
              </div>
              <div style="color:#718096;font-size:0.82rem;">ML portfolio project</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "Navigate", options,
            index=options.index(st.session_state["current_page"]),
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(
            "**Model:** tuned Random Forest\n"
            "**Holdout AUC:** 0.842\n"
            "**Data:** IBM Telco · 7,043 customers\n"
            "**Built with:** Python + Streamlit"
        )

    st.session_state["current_page"] = page
    PAGES[page]()
    footer()


if __name__ == "__main__":
    main()

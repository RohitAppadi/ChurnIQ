"""
Customer Churn Prediction System
Production-ready Streamlit app with authentication, paywall, and SaaS-style UI.
"""

import streamlit as st
import pandas as pd
import joblib
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Prediction Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #e2e8f0;
    }
    .stApp {
        background: #080d1a;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d1525;
        border-right: 1px solid #1e2d45;
    }
    section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] .sidebar-brand {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #38bdf8 !important;
        letter-spacing: -0.5px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1e2d45;
    }

    /* ── Headings ── */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
        letter-spacing: -0.5px;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f1e35 0%, #0d2240 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 1.2rem 1.5rem !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-2px); }
    div[data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }

    /* ── File uploader ── */
    div[data-testid="stFileUploader"] {
        background: #0d1525;
        border: 2px dashed #1e3a5f;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
        padding: 0.55rem 1.5rem !important;
        transition: opacity 0.2s, transform 0.15s !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Dataframe ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Alert / info boxes ── */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ── Login form wrapper ── */
    .login-wrapper {
        max-width: 420px;
        margin: 6vh auto 0;
        background: #0d1525;
        border: 1px solid #1e3a5f;
        border-radius: 18px;
        padding: 2.5rem 2.5rem 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .login-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .login-sub {
        text-align: center;
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 1.6rem;
    }

    /* ── Paywall card ── */
    .paywall-card {
        background: linear-gradient(135deg, #0f1e35, #0a1628);
        border: 1px solid #1e3a5f;
        border-radius: 18px;
        padding: 2.5rem;
        text-align: center;
        max-width: 520px;
        margin: 4vh auto;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .paywall-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .paywall-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
    }
    .paywall-body { color: #64748b; font-size: 0.95rem; line-height: 1.6; }

    /* ── Section divider ── */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-left: 3px solid #0ea5e9;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }

    /* ── Risk badge colours (in dataframe not possible, used in caption) ── */
    .badge-high   { color: #f87171; font-weight: 600; }
    .badge-medium { color: #fbbf24; font-weight: 600; }
    .badge-low    { color: #34d399; font-weight: 600; }

    /* ── Hide default Streamlit menu / footer ── */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# AUTH CONFIG
# Passwords below are bcrypt hashes.
# Pre-hashed via: stauth.Hasher(['password123','demo456']).generate()
# Replace with your own hashed passwords in production.
# ─────────────────────────────────────────────
AUTH_CONFIG = {
    "credentials": {
        "usernames": {
            "rohit": {
                "name": "Rohit Sharma",
                "email": "rohit@churniq.ai",
                # plain: password123
                "password": "$2b$12$KIGnhMXs1nFz5yJrOm7XmuHnz1gR3jAGK9HMwLvKhv.gCF1WXMkoy",
            },
            "demo": {
                "name": "Demo User",
                "email": "demo@churniq.ai",
                # plain: demo456
                "password": "$2b$12$Rb0.qr5hB0XT/eOjXhEoYOrxdKeFW1HBlOqOmcS/dDG/JMdg1bpXW",
            },
        }
    },
    "cookie": {
        "name": "churniq_auth",
        "key": "churniq_super_secret_key_2024",   # change in production
        "expiry_days": 1,
    },
    "preauthorized": {"emails": []},
}

# ─────────────────────────────────────────────
# PRO-ACCESS LIST  (usernames with full access)
# ─────────────────────────────────────────────
PRO_USERS = {"rohit"}

# ─────────────────────────────────────────────
# ML HELPERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading prediction engine…")
def load_model():
    """Load and cache model artifacts from disk."""
    try:
        model = joblib.load("models/churn_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        return model, scaler, feature_columns
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Ensure the `models/` folder contains all three .pkl files.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        st.stop()


def preprocess_data(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Clean raw CSV, drop leakage columns, encode categoricals,
    and align to trained feature set.

    Returns a DataFrame ready for scaling → prediction.
    Raises ValueError with a descriptive message on bad input.
    """
    # ── 1. Clean column names ──
    df.columns = df.columns.str.replace('\ufeff', '', regex=True).str.strip()

    # ── 2. Drop known leakage / ID columns ──
    leakage_cols = [
        "CustomerID", "Count", "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "CLTV", "Churn Reason",
    ]
    df = df.drop(columns=leakage_cols, errors="ignore")

    # ── 3. Separate features from target ──
    if "Churn Value" not in df.columns:
        raise ValueError(
            "Required column **'Churn Value'** not found. "
            "Please upload the original IBM Telco Customer Churn CSV."
        )
    X = df.drop("Churn Value", axis=1)

    # ── 4. Drop high-cardinality geographic columns ──
    high_cardinality = ["Country", "State", "City", "Zip Code"]
    X = X.drop(columns=high_cardinality, errors="ignore")

    # ── 5. One-hot encode ──
    X = pd.get_dummies(X, drop_first=True)

    # ── 6. Align to trained feature columns ──
    X = X.reindex(columns=feature_columns, fill_value=0)

    return X


def predict(X: pd.DataFrame, model, scaler):
    """Scale features and return (labels, probabilities)."""
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]
    return labels, probs


def risk_segment(prob: float) -> str:
    """Map churn probability to a human-readable risk tier."""
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    return "Low Risk"


# ─────────────────────────────────────────────
# SIDEBAR  (built after authentication)
# ─────────────────────────────────────────────

def render_sidebar(name: str, authenticator):
    """Render sidebar navigation and user info."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">⚡ ChurnIQ</div>', unsafe_allow_html=True)
        st.markdown("*Customer Intelligence Platform*")
        st.markdown("---")

        st.markdown(f"👤 **{name}**")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠  Home", "📂  Upload & Predict", "📊  Results"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        authenticator.logout("🚪  Logout", location="sidebar")

    # Normalise label
    return page.split("  ", 1)[-1].strip()


# ─────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────

def page_home(name: str):
    st.markdown(f"## Welcome back, {name.split()[0]} 👋")
    st.markdown(
        "**ChurnIQ** uses a trained machine-learning model to identify customers "
        "most likely to leave — before they do."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div style="background:#0d1525;border:1px solid #1e3a5f;border-radius:14px;padding:1.5rem;">
            <div style="font-size:2rem">📂</div>
            <h4 style="font-family:Syne,sans-serif;color:#f1f5f9;margin:0.5rem 0 0.25rem">Upload CSV</h4>
            <p style="color:#64748b;font-size:0.88rem">Upload the raw IBM Telco CSV and let ChurnIQ handle the rest.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="background:#0d1525;border:1px solid #1e3a5f;border-radius:14px;padding:1.5rem;">
            <div style="font-size:2rem">🤖</div>
            <h4 style="font-family:Syne,sans-serif;color:#f1f5f9;margin:0.5rem 0 0.25rem">Predict</h4>
            <p style="color:#64748b;font-size:0.88rem">ML model scores every customer with a churn probability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div style="background:#0d1525;border:1px solid #1e3a5f;border-radius:14px;padding:1.5rem;">
            <div style="font-size:2rem">📊</div>
            <h4 style="font-family:Syne,sans-serif;color:#f1f5f9;margin:0.5rem 0 0.25rem">Act</h4>
            <p style="color:#64748b;font-size:0.88rem">Download segmented results and prioritise retention efforts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.info(
        "**How to use:** Navigate to **Upload & Predict** in the sidebar, "
        "upload your CSV, and view results in **Results**."
    )


def page_upload(username: str):
    """Upload page — gated behind PRO_USERS."""

    # ── Paywall ──
    if username not in PRO_USERS:
        st.markdown(
            """
            <div class="paywall-card">
              <div class="paywall-icon">🔒</div>
              <div class="paywall-title">Pro Feature</div>
              <div class="paywall-body">
                Churn predictions are available on the <strong>Pro Plan</strong>.<br><br>
                Upgrade your account to unlock batch CSV predictions, risk segmentation,
                revenue-at-risk analytics, and priority support.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Load model (cached) ──
    model, scaler, feature_columns = load_model()

    st.markdown("## 📂 Upload Customer Data")
    st.markdown("Upload the **original IBM Telco Customer Churn** CSV file.")

    uploaded_file = st.file_uploader(
        "Drop your CSV here, or click to browse",
        type=["csv"],
        help="Must include 'Churn Value' and the standard IBM Telco columns.",
    )

    if not uploaded_file:
        return  # nothing to do yet

    # ── Read CSV ──
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return

    if raw_df.empty:
        st.warning("The uploaded file appears to be empty.")
        return

    # ── Preprocess ──
    with st.spinner("Preprocessing data…"):
        try:
            X = preprocess_data(raw_df.copy(), feature_columns)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            return

    # ── Predict ──
    with st.spinner("Running predictions…"):
        try:
            labels, probs = predict(X, model, scaler)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    # ── Build result dataframe ──
    result_df = raw_df.copy()
    result_df.columns = result_df.columns.str.replace('\ufeff', '', regex=True).str.strip()
    result_df["Predicted_Churn"] = labels
    result_df["Churn_Probability"] = probs
    result_df["Risk Segment"] = result_df["Churn_Probability"].apply(risk_segment)

    # Persist in session so Results page can show it
    st.session_state["result_df"] = result_df

    st.success("✅ Predictions complete! Head to **Results** in the sidebar.")


def page_results():
    """Results page — shows metrics, preview, and download."""
    st.markdown("## 📊 Prediction Results")

    if "result_df" not in st.session_state:
        st.info("No predictions yet. Upload a CSV on the **Upload & Predict** page first.")
        return

    result_df: pd.DataFrame = st.session_state["result_df"]

    # ── KPI metrics ──
    total_customers = len(result_df)
    high_risk = len(result_df[result_df["Risk Segment"] == "High Risk"])
    medium_risk = len(result_df[result_df["Risk Segment"] == "Medium Risk"])

    rev_col = next(
        (c for c in result_df.columns if c.lower().replace(" ", "") == "monthlycharges"),
        None,
    )
    revenue_at_risk = (
        result_df[result_df["Risk Segment"] == "High Risk"][rev_col].sum()
        if rev_col
        else 0.0
    )

    st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("High Risk 🔴", f"{high_risk:,}")
    c3.metric("Medium Risk 🟡", f"{medium_risk:,}")
    c4.metric(
        "Revenue at Risk 💸",
        f"${revenue_at_risk:,.0f}" if rev_col else "N/A",
        help="Sum of Monthly Charges for High Risk customers." if rev_col else "Monthly Charges column not found.",
    )

    # ── Risk distribution bar ──
    st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
    dist = result_df["Risk Segment"].value_counts().reset_index()
    dist.columns = ["Risk Segment", "Count"]
    st.bar_chart(dist.set_index("Risk Segment"), color="#0ea5e9")

    # ── Data preview ──
    st.markdown('<div class="section-header">Predictions Preview</div>', unsafe_allow_html=True)

    display_cols = [
        c for c in [
            "CustomerID", "Gender", "Senior Citizen", "Monthly Charges",
            "Predicted_Churn", "Churn_Probability", "Risk Segment",
        ]
        if c in result_df.columns
    ] or result_df.columns.tolist()

    st.dataframe(
        result_df[display_cols].head(20).style.format(
            {"Churn_Probability": "{:.1%}"}
        ),
        use_container_width=True,
    )

    # ── Download ──
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Full Predictions CSV",
        data=csv_bytes,
        file_name="churniq_predictions.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# MAIN — Auth gate → App
# ─────────────────────────────────────────────

def main():
    # ── Build authenticator ──
    authenticator = stauth.Authenticate(
        AUTH_CONFIG["credentials"],
        AUTH_CONFIG["cookie"]["name"],
        AUTH_CONFIG["cookie"]["key"],
        AUTH_CONFIG["cookie"]["expiry_days"],
        AUTH_CONFIG["preauthorized"],
    )

    # ── Login form ──
    name, auth_status, username = authenticator.login(
        "ChurnIQ · Sign In",
        location="main",
    )

    # ── Handle auth states ──
    if auth_status is False:
        st.error("Incorrect username or password. Please try again.")
        return

    if auth_status is None:
        st.markdown(
            """
            <div style="text-align:center;margin-top:2rem;">
              <div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;color:#38bdf8;">⚡ ChurnIQ</div>
              <div style="color:#64748b;margin-top:0.25rem;font-size:0.9rem;">
                Enter your credentials above to access the platform.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Authenticated ──
    page = render_sidebar(name, authenticator)

    if page == "Home":
        page_home(name)
    elif page == "Upload & Predict":
        page_upload(username)
    elif page == "Results":
        page_results()


if __name__ == "__main__":
    main()
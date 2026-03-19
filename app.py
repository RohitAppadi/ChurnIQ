"""
ChurnIQ · Customer Churn Prediction System
Production-ready Streamlit app.

Compatible with: streamlit-authenticator >= 0.3.0
Requirements:
    pip install streamlit streamlit-authenticator joblib pandas PyYAML bcrypt
"""

import streamlit as st
import pandas as pd
import joblib
import streamlit_authenticator as stauth

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  ← must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Prediction Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #e2e8f0; }
.stApp { background: #080d1a; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1525 !important;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.sidebar-brand {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    color: #38bdf8 !important;
    letter-spacing: -0.5px;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.4px;
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
    font-size: 0.75rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem !important;
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
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    overflow: hidden;
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
.paywall-icon  { font-size: 3rem; margin-bottom: 0.5rem; }
.paywall-title { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.75rem; }
.paywall-body  { color: #64748b; font-size: 0.95rem; line-height: 1.7; }

/* ── Feature cards (Home page) ── */
.feat-card {
    background: #0d1525;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.5rem;
    height: 100%;
}
.feat-card h4 { font-family: 'Syne', sans-serif; color: #f1f5f9; margin: 0.5rem 0 0.25rem; }
.feat-card p  { color: #64748b; font-size: 0.88rem; }

/* ── Section divider label ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-left: 3px solid #0ea5e9;
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem;
}

/* ── Login form cosmetics ── */
div[data-testid="stForm"] {
    background: #0d1525;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# AUTH CONFIG  (streamlit-authenticator v0.3 / v0.4 schema)
#
# • Passwords are plain text here; auto_hash=True (the default)
#   means the library hashes them automatically on first run.
# • v0.3+ removed the `pre_authorized` 5th argument from Authenticate().
#   The constructor now only takes 4 positional args.
# ─────────────────────────────────────────────────────────────
AUTH_CONFIG: dict = {
    "credentials": {
        "usernames": {
            "rohit": {
                "first_name": "Rohit",
                "last_name": "Sharma",
                "email": "rohit@churniq.ai",
                "password": "password123",   # auto-hashed on first run
                "logged_in": False,
            },
            "demo": {
                "first_name": "Demo",
                "last_name": "User",
                "email": "demo@churniq.ai",
                "password": "demo456",       # auto-hashed on first run
                "logged_in": False,
            },
        }
    },
    "cookie": {
        "name": "churniq_auth_cookie",
        "key":  "churniq_super_secret_key_xyz_2024",   # change in production!
        "expiry_days": 1,
    },
}

# ─────────────────────────────────────────────────────────────
# PRO ACCESS LIST — usernames with full prediction access
# ─────────────────────────────────────────────────────────────
PRO_USERS: set = {"rohit"}


# ─────────────────────────────────────────────────────────────
# ML HELPERS
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading prediction engine…")
def load_model():
    """
    Load and cache the three model artifacts from disk.
    Paths:
        models/churn_model.pkl
        models/scaler.pkl
        models/feature_columns.pkl
    """
    try:
        model           = joblib.load("models/churn_model.pkl")
        scaler          = joblib.load("models/scaler.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        return model, scaler, feature_columns
    except FileNotFoundError as exc:
        st.error(
            f"❌ Model file not found: `{exc.filename}`. "
            "Make sure the `models/` folder contains **churn_model.pkl**, "
            "**scaler.pkl**, and **feature_columns.pkl**."
        )
        st.stop()
    except Exception as exc:
        st.error(f"❌ Unexpected error loading model artifacts: {exc}")
        st.stop()


def preprocess_data(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Reproduce the exact preprocessing pipeline used during model training.

    Steps:
        1. Strip BOM characters and whitespace from column names
        2. Drop leakage / identifier columns
        3. Validate and separate X from 'Churn Value' target
        4. Drop high-cardinality geographic columns
        5. One-hot encode categorical features
        6. Align columns to trained feature set (fill missing with 0)

    Returns a DataFrame ready for scaler.transform() → model.predict().
    Raises ValueError with a user-friendly message on invalid input.
    """
    # 1. Clean column names
    df.columns = df.columns.str.replace('\ufeff', '', regex=True).str.strip()

    # 2. Drop leakage / identifier columns
    leakage_cols = [
        "CustomerID", "Count", "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "CLTV", "Churn Reason",
    ]
    df = df.drop(columns=leakage_cols, errors="ignore")

    # 3. Validate that the target column exists
    if "Churn Value" not in df.columns:
        raise ValueError(
            "Required column **'Churn Value'** is missing. "
            "Please upload the original **IBM Telco Customer Churn** CSV."
        )
    X = df.drop("Churn Value", axis=1)

    # 4. Drop high-cardinality geographic columns
    high_cardinality = ["Country", "State", "City", "Zip Code"]
    X = X.drop(columns=high_cardinality, errors="ignore")

    # 5. One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # 6. Align with trained feature columns (add missing cols filled with 0)
    X = X.reindex(columns=feature_columns, fill_value=0)

    return X


def run_predict(X: pd.DataFrame, model, scaler):
    """Scale features and return (binary labels, churn probabilities)."""
    X_scaled = scaler.transform(X)
    labels   = model.predict(X_scaled)
    probs    = model.predict_proba(X_scaled)[:, 1]
    return labels, probs


def risk_segment(prob: float) -> str:
    """Map a churn probability float to a three-tier risk label."""
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    return "Low Risk"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(name: str, authenticator) -> str:
    """Render the sidebar navigation panel. Returns the selected page name."""
    with st.sidebar:
        st.markdown('<p class="sidebar-brand">⚡ ChurnIQ</p>', unsafe_allow_html=True)
        st.caption("Customer Intelligence Platform")
        st.markdown("---")

        st.markdown(f"👤 **{name}**")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠  Home", "📂  Upload & Predict", "📊  Results"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        # v0.3 / v0.4 logout signature: (button_name, location, key)
        authenticator.logout("🚪  Logout", location="sidebar", key="sidebar_logout")

    # Strip the emoji + spaces prefix to get a plain page name
    return page.split("  ", 1)[-1].strip()


# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────

def page_home(name: str) -> None:
    first = name.split()[0] if name else "there"
    st.markdown(f"## Welcome back, {first} 👋")
    st.markdown(
        "**ChurnIQ** uses a trained machine-learning model to identify customers "
        "most likely to leave — before they do."
    )

    c1, c2, c3 = st.columns(3, gap="large")
    cards = [
        ("📂", "Upload CSV",
         "Upload the raw IBM Telco CSV and let ChurnIQ handle the rest."),
        ("🤖", "Predict",
         "The ML model scores every customer with a churn probability."),
        ("📊", "Act",
         "Download segmented results and prioritise retention efforts."),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'<div class="feat-card">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<h4>{title}</h4>'
                f'<p>{body}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.info(
        "**How to use →** Open **Upload & Predict** in the sidebar, "
        "upload your CSV, then view results in **Results**."
    )


# ─────────────────────────────────────────────────────────────
# PAGE: UPLOAD & PREDICT
# ─────────────────────────────────────────────────────────────

def page_upload(username: str) -> None:
    """Upload page — gated behind the PRO_USERS paywall."""

    # ── Paywall ──────────────────────────────────────────────
    if username not in PRO_USERS:
        st.markdown(
            """
            <div class="paywall-card">
              <div class="paywall-icon">🔒</div>
              <div class="paywall-title">Pro Feature</div>
              <div class="paywall-body">
                Churn predictions are available on the <strong>Pro Plan</strong>.<br><br>
                Upgrade your account to unlock batch CSV predictions,
                risk segmentation, revenue-at-risk analytics, and priority support.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Load model (cached) ───────────────────────────────────
    model, scaler, feature_columns = load_model()

    st.markdown("## 📂 Upload Customer Data")
    st.markdown(
        "Upload the **original IBM Telco Customer Churn** CSV file. "
        "The file must include a `Churn Value` column."
    )

    uploaded_file = st.file_uploader(
        "Drop your CSV here, or click to browse",
        type=["csv"],
        help="Standard IBM Telco Customer Churn dataset format.",
    )

    if not uploaded_file:
        return

    # ── Parse ────────────────────────────────────────────────
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not parse the CSV file: `{exc}`")
        return

    if raw_df.empty:
        st.warning("The uploaded file appears to be empty. Please check the file and try again.")
        return

    # ── Preprocess ───────────────────────────────────────────
    with st.spinner("Preprocessing data…"):
        try:
            X = preprocess_data(raw_df.copy(), feature_columns)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Preprocessing failed: `{exc}`")
            return

    # ── Predict ──────────────────────────────────────────────
    with st.spinner("Running predictions…"):
        try:
            labels, probs = run_predict(X, model, scaler)
        except Exception as exc:
            st.error(f"Prediction error: `{exc}`")
            return

    # ── Build result DataFrame ────────────────────────────────
    result_df = raw_df.copy()
    result_df.columns = result_df.columns.str.replace('\ufeff', '', regex=True).str.strip()
    result_df["Predicted_Churn"]   = labels
    result_df["Churn_Probability"] = probs
    result_df["Risk Segment"]      = result_df["Churn_Probability"].apply(risk_segment)

    # Persist so the Results page can read it
    st.session_state["result_df"] = result_df

    st.success("✅ Predictions complete! Navigate to **Results** in the sidebar.")


# ─────────────────────────────────────────────────────────────
# PAGE: RESULTS
# ─────────────────────────────────────────────────────────────

def page_results() -> None:
    st.markdown("## 📊 Prediction Results")

    if "result_df" not in st.session_state:
        st.info("No predictions yet. Upload a CSV on the **Upload & Predict** page first.")
        return

    result_df: pd.DataFrame = st.session_state["result_df"]

    # ── KPI calculations ──────────────────────────────────────
    total_customers = len(result_df)
    high_risk_df    = result_df[result_df["Risk Segment"] == "High Risk"]
    medium_risk_df  = result_df[result_df["Risk Segment"] == "Medium Risk"]
    high_risk_count = len(high_risk_df)
    med_risk_count  = len(medium_risk_df)

    # Locate Monthly Charges column (case-insensitive, space-insensitive)
    rev_col = next(
        (c for c in result_df.columns if c.lower().replace(" ", "") == "monthlycharges"),
        None,
    )
    revenue_at_risk = high_risk_df[rev_col].sum() if rev_col else None

    # ── Metric row ────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("High Risk 🔴",    f"{high_risk_count:,}")
    c3.metric("Medium Risk 🟡",  f"{med_risk_count:,}")
    c4.metric(
        "Revenue at Risk 💸",
        f"${revenue_at_risk:,.0f}" if revenue_at_risk is not None else "N/A",
        help=(
            "Sum of Monthly Charges for High Risk customers."
            if rev_col else
            "Column 'Monthly Charges' not found in the uploaded file."
        ),
    )

    # ── Risk distribution chart ───────────────────────────────
    st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
    dist = (
        result_df["Risk Segment"]
        .value_counts()
        .reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
        .reset_index()
    )
    dist.columns = ["Risk Segment", "Count"]
    st.bar_chart(dist.set_index("Risk Segment"), color="#0ea5e9", use_container_width=True)

    # ── Preview table ─────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Predictions Preview (first 20 rows)</div>',
        unsafe_allow_html=True,
    )
    preferred_cols = [
        "CustomerID", "Gender", "Senior Citizen",
        "Monthly Charges", "Predicted_Churn",
        "Churn_Probability", "Risk Segment",
    ]
    display_cols = [c for c in preferred_cols if c in result_df.columns] or result_df.columns.tolist()

    st.dataframe(
        result_df[display_cols].head(20).style.format({"Churn_Probability": "{:.1%}"}),
        use_container_width=True,
    )

    # ── Download button ───────────────────────────────────────
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Full Predictions CSV",
        data=csv_bytes,
        file_name="churniq_predictions.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# BRANDING shown on the login screen before authentication
# ─────────────────────────────────────────────────────────────

def _render_login_branding() -> None:
    st.markdown(
        """
        <div style="text-align:center;margin-top:1.5rem;">
          <div style="font-family:Syne,sans-serif;font-size:0.8rem;color:#1e3a5f;
                      letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;">
            Powered by
          </div>
          <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#38bdf8;">
            ⚡ ChurnIQ
          </div>
          <div style="color:#64748b;margin-top:0.3rem;font-size:0.88rem;">
            Customer Intelligence Platform · ML-powered churn prediction
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN — Authentication gate → page router
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # ── Instantiate authenticator (v0.3/v0.4 — only 4 positional args) ──
    authenticator = stauth.Authenticate(
        AUTH_CONFIG["credentials"],
        AUTH_CONFIG["cookie"]["name"],
        AUTH_CONFIG["cookie"]["key"],
        AUTH_CONFIG["cookie"]["expiry_days"],
    )

    # ── Show login form ──
    # In v0.3+ login() returns None; results live in st.session_state
    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    name        = st.session_state.get("name", "")
    username    = st.session_state.get("username", "")

    # ── Route on auth status ──
    if auth_status is False:
        st.error("❌ Incorrect username or password. Please try again.")
        _render_login_branding()
        return

    if auth_status is None:
        # Not yet logged in — show branding below the form
        _render_login_branding()
        return

    # ── Fully authenticated ──
    page = render_sidebar(name, authenticator)

    if page == "Home":
        page_home(name)
    elif page == "Upload & Predict":
        page_upload(username)
    elif page == "Results":
        page_results()


if __name__ == "__main__":
    main()
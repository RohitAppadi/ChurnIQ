# FULL UPGRADED VERSION (CLEAN + PREMIUM UI)

import streamlit as st
import pandas as pd
import joblib
import streamlit_authenticator as stauth

st.set_page_config(page_title="ChurnIQ · Prediction Engine", page_icon="⚡", layout="wide")

# ---------------- THEME ----------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

def is_dark():
    return st.session_state.get("dark_mode", True)

# ---------------- CSS ----------------
def inject_css():
    st.markdown("""
    <style>
    .glass-card {
        background: rgba(20,30,60,0.6);
        backdrop-filter: blur(20px);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
    }
    .fade {
        animation: fadeIn 0.5s ease;
    }
    @keyframes fadeIn {
        from {opacity:0; transform: translateY(10px);} 
        to {opacity:1; transform: translateY(0);} 
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- AUTH ----------------
AUTH_CONFIG = {
    "credentials": {
        "usernames": {
            "rohit": {"name": "Rohit", "password": "rohit123"},
            "demo": {"name": "Demo", "password": "demo456"},
            "mahesh": {"name": "Mahesh", "password": "mahesh123"},
            "karan": {"name": "Karan", "password": "karan123"},
            "areen": {"name": "Areen", "password": "areen123"},
            "devang": {"name": "Devang", "password": "devang123"},
        }
    },
    "cookie": {"name": "churn_cookie", "key": "abc", "expiry_days": 1},
}

PRO_USERS = {"rohit","mahesh","karan","areen","devang"}

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

# ---------------- SIDEBAR ----------------
def sidebar(name):
    with st.sidebar:
        st.title("⚡ ChurnIQ")
        st.write(f"Welcome {name}")

        page = st.radio("", ["Home","Upload","Results"])
    return page

# ---------------- HOME ----------------
def page_home(name):
    st.markdown('<div class="fade">', unsafe_allow_html=True)

    st.markdown(f"<div class='hero-title'>Predict Churn. Retain Revenue.</div>", unsafe_allow_html=True)

    cols = st.columns(3)
    features = [("⚡","Fast","Real-time predictions"),("📊","Insights","Risk segmentation"),("🧠","AI","Decision engine")]

    for col,(i,t,d) in zip(cols,features):
        with col:
            st.markdown(f"""
            <div class='glass-card'>
            <h3>{i} {t}</h3>
            <p>{d}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
def page_upload(username):
    if username.lower() not in PRO_USERS:
        st.warning("Upgrade to Pro")
        return

    file = st.file_uploader("Upload CSV")
    if file:
        df = pd.read_csv(file)
        st.session_state["data"] = df
        st.success("Uploaded")

# ---------------- RESULTS ----------------
def page_results():
    if "data" not in st.session_state:
        st.info("Upload first")
        return

    df = st.session_state["data"]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write(df.head())
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN ----------------
def main():
    inject_css()

    authenticator = stauth.Authenticate(
        AUTH_CONFIG['credentials'],
        AUTH_CONFIG['cookie']['name'],
        AUTH_CONFIG['cookie']['key'],
        AUTH_CONFIG['cookie']['expiry_days']
    )

    authenticator.login("Login", "main")

    if st.session_state.get("authentication_status"):
        name = st.session_state.get("name")
        username = st.session_state.get("username")

        page = sidebar(name)

        if page == "Home":
            page_home(name)
        elif page == "Upload":
            page_upload(username)
        elif page == "Results":
            page_results()

        authenticator.logout("Logout")

    elif st.session_state.get("authentication_status") is False:
        st.error("Wrong credentials")

if __name__ == "__main__":
    main()

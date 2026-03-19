"""
ChurnIQ · Customer Churn Prediction System
Production-ready Streamlit app — Premium UI Edition.

Compatible with: streamlit-authenticator >= 0.3.0
Requirements:
    pip install streamlit streamlit-authenticator joblib pandas PyYAML bcrypt
"""

import streamlit as st
import pandas as pd
import joblib
import streamlit_authenticator as stauth

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Prediction Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME STATE INIT
# ─────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True


def is_dark() -> bool:
    return st.session_state.get("dark_mode", True)


# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  (theme-aware via CSS variables injected per render)
# ─────────────────────────────────────────────────────────────
def inject_css() -> None:
    dark = is_dark()

    # ── Theme token sets ──
    if dark:
        bg_root       = "#050b18"
        bg_surface    = "rgba(13, 21, 37, 0.72)"
        bg_surface2   = "rgba(15, 28, 52, 0.60)"
        border_color  = "rgba(56, 189, 248, 0.15)"
        border_hover  = "rgba(56, 189, 248, 0.45)"
        text_primary  = "#f0f6ff"
        text_secondary= "#7ea3c4"
        text_muted    = "#3d5a78"
        accent        = "#38bdf8"
        accent2       = "#6ee7f7"
        accent_glow   = "rgba(56,189,248,0.25)"
        btn_from      = "#0ea5e9"
        btn_to        = "#1d4ed8"
        metric_val    = "#38bdf8"
        hero_title    = "linear-gradient(135deg, #e0f2fe 0%, #38bdf8 40%, #6ee7f7 100%)"
        scrollbar_bg  = "#0d1525"
        scrollbar_th  = "#1e3a5f"
        mesh1         = "radial-gradient(ellipse 80% 60% at 15% 10%, rgba(56,189,248,0.10) 0%, transparent 65%)"
        mesh2         = "radial-gradient(ellipse 60% 50% at 85% 85%, rgba(99,102,241,0.08) 0%, transparent 60%)"
        mesh3         = "radial-gradient(ellipse 50% 40% at 50% 50%, rgba(14,165,233,0.05) 0%, transparent 70%)"
        sidebar_bg    = "rgba(7,14,27,0.95)"
        upload_bg     = "rgba(13,21,37,0.55)"
        upload_border = "rgba(56,189,248,0.30)"
        upload_hover  = "rgba(56,189,248,0.55)"
        paywall_bg    = "rgba(10,16,30,0.80)"
        input_bg      = "rgba(13,21,37,0.85)"
    else:
        bg_root       = "#f0f5ff"
        bg_surface    = "rgba(255,255,255,0.72)"
        bg_surface2   = "rgba(241,245,255,0.80)"
        border_color  = "rgba(14,165,233,0.18)"
        border_hover  = "rgba(14,165,233,0.50)"
        text_primary  = "#0c1929"
        text_secondary= "#375778"
        text_muted    = "#8faec8"
        accent        = "#0284c7"
        accent2       = "#0ea5e9"
        accent_glow   = "rgba(2,132,199,0.18)"
        btn_from      = "#0284c7"
        btn_to        = "#1d4ed8"
        metric_val    = "#0284c7"
        hero_title    = "linear-gradient(135deg, #0c1929 0%, #0284c7 50%, #0ea5e9 100%)"
        scrollbar_bg  = "#dbeafe"
        scrollbar_th  = "#93c5fd"
        mesh1         = "radial-gradient(ellipse 80% 60% at 15% 10%, rgba(14,165,233,0.12) 0%, transparent 65%)"
        mesh2         = "radial-gradient(ellipse 60% 50% at 85% 85%, rgba(99,102,241,0.08) 0%, transparent 60%)"
        mesh3         = "radial-gradient(ellipse 50% 40% at 50% 50%, rgba(14,165,233,0.06) 0%, transparent 70%)"
        sidebar_bg    = "rgba(232,242,255,0.97)"
        upload_bg     = "rgba(224,242,254,0.55)"
        upload_border = "rgba(14,165,233,0.35)"
        upload_hover  = "rgba(14,165,233,0.60)"
        paywall_bg    = "rgba(224,242,254,0.80)"
        input_bg      = "rgba(255,255,255,0.88)"

    st.markdown(f"""
<style>
/* ═══════════════════════════════════════════════
   0. FONTS
═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Outfit:wght@300;400;500;600&display=swap');

/* ═══════════════════════════════════════════════
   1. ROOT & ANIMATED BACKGROUND
═══════════════════════════════════════════════ */
html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif;
    color: {text_primary};
    transition: color 0.4s ease, background 0.4s ease;
}}

.stApp {{
    background: {bg_root};
    background-image: {mesh1}, {mesh2}, {mesh3};
    min-height: 100vh;
    overflow-x: hidden;
}}

/* Animated floating orbs behind everything */
.stApp::before {{
    content: '';
    position: fixed;
    top: -30%;
    left: -20%;
    width: 60vw;
    height: 60vw;
    background: radial-gradient(circle, {accent_glow} 0%, transparent 65%);
    border-radius: 50%;
    animation: orb-drift 18s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}}
.stApp::after {{
    content: '';
    position: fixed;
    bottom: -20%;
    right: -10%;
    width: 45vw;
    height: 45vw;
    background: radial-gradient(circle, rgba(99,102,241,0.10) 0%, transparent 65%);
    border-radius: 50%;
    animation: orb-drift 22s ease-in-out infinite alternate-reverse;
    pointer-events: none;
    z-index: 0;
}}
@keyframes orb-drift {{
    0%   {{ transform: translate(0,0) scale(1); }}
    50%  {{ transform: translate(4vw, 3vh) scale(1.08); }}
    100% {{ transform: translate(-3vw, 5vh) scale(0.95); }}
}}

/* ═══════════════════════════════════════════════
   2. SCROLLBAR
═══════════════════════════════════════════════ */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {scrollbar_bg}; }}
::-webkit-scrollbar-thumb {{
    background: {scrollbar_th};
    border-radius: 99px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {accent}; }}

/* ═══════════════════════════════════════════════
   3. SIDEBAR
═══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    backdrop-filter: blur(24px) saturate(1.4) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.4) !important;
    border-right: 1px solid {border_color} !important;
    transition: background 0.4s ease;
}}
section[data-testid="stSidebar"] * {{
    color: {text_secondary} !important;
    transition: color 0.3s ease;
}}
section[data-testid="stSidebar"] strong {{
    color: {text_primary} !important;
}}

.sidebar-brand {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, {accent}, {accent2}) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.5px;
    display: inline-block;
}}
.sidebar-tagline {{
    font-size: 0.72rem;
    color: {text_muted} !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: -2px;
}}
.sidebar-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {border_color}, transparent);
    margin: 0.85rem 0;
    border: none;
}}
.sidebar-user {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.6rem 0.9rem;
    background: {bg_surface2};
    border: 1px solid {border_color};
    border-radius: 10px;
    font-size: 0.88rem;
    backdrop-filter: blur(12px);
}}
.sidebar-user-dot {{
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 8px #22c55e;
    flex-shrink: 0;
}}

/* ═══════════════════════════════════════════════
   4. TYPOGRAPHY
═══════════════════════════════════════════════ */
h1, h2, h3, h4 {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: {text_primary} !important;
    letter-spacing: -0.4px;
    transition: color 0.4s ease;
}}
p, li, span, label {{
    color: {text_secondary};
    line-height: 1.7;
    transition: color 0.4s ease;
}}

/* ═══════════════════════════════════════════════
   5. GLASS CARDS
═══════════════════════════════════════════════ */
.glass-card {{
    background: {bg_surface};
    backdrop-filter: blur(20px) saturate(1.5);
    -webkit-backdrop-filter: blur(20px) saturate(1.5);
    border: 1px solid {border_color};
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1),
                border-color 0.25s ease,
                box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
    animation: fade-up 0.5s ease both;
}}
.glass-card::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, transparent 60%);
    pointer-events: none;
    border-radius: inherit;
}}
.glass-card:hover {{
    transform: translateY(-4px) scale(1.005);
    border-color: {border_hover};
    box-shadow: 0 12px 40px {accent_glow}, 0 2px 8px rgba(0,0,0,0.15);
}}

.glass-card-sm {{
    background: {bg_surface};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid {border_color};
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    animation: fade-up 0.45s ease both;
}}
.glass-card-sm:hover {{
    transform: translateY(-3px);
    border-color: {border_hover};
    box-shadow: 0 8px 28px {accent_glow};
}}

/* ═══════════════════════════════════════════════
   6. METRIC CARDS
═══════════════════════════════════════════════ */
div[data-testid="metric-container"] {{
    background: {bg_surface} !important;
    backdrop-filter: blur(20px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.5) !important;
    border: 1px solid {border_color} !important;
    border-radius: 16px !important;
    padding: 1.3rem 1.5rem !important;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1),
                border-color 0.25s ease,
                box-shadow 0.25s ease !important;
    animation: fade-up 0.5s ease both;
    position: relative;
    overflow: hidden;
}}
div[data-testid="metric-container"]::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {accent}, transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}}
div[data-testid="metric-container"]:hover {{
    transform: translateY(-4px) !important;
    border-color: {border_hover} !important;
    box-shadow: 0 12px 36px {accent_glow} !important;
}}
div[data-testid="metric-container"]:hover::after {{ opacity: 1; }}

div[data-testid="metric-container"] label {{
    color: {text_muted} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.11em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: {metric_val} !important;
    text-shadow: 0 0 24px {accent_glow};
}}

/* ═══════════════════════════════════════════════
   7. HERO SECTION
═══════════════════════════════════════════════ */
.hero-wrapper {{
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    position: relative;
    animation: fade-up 0.7s ease both;
}}
.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: {bg_surface};
    border: 1px solid {border_color};
    border-radius: 99px;
    padding: 0.3rem 1rem;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {accent} !important;
    font-weight: 600;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}}
.hero-badge-dot {{
    width: 6px; height: 6px;
    background: {accent};
    border-radius: 50%;
    box-shadow: 0 0 8px {accent};
    animation: pulse-dot 2s ease-in-out infinite;
}}
@keyframes pulse-dot {{
    0%, 100% {{ transform: scale(1); opacity: 1; }}
    50%       {{ transform: scale(1.5); opacity: 0.6; }}
}}
.hero-title {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    background: {hero_title};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -1.5px;
    margin-bottom: 1rem;
}}
.hero-sub {{
    font-size: clamp(0.9rem, 1.8vw, 1.1rem);
    color: {text_secondary};
    max-width: 560px;
    margin: 0 auto 2rem;
    line-height: 1.75;
}}
.hero-glow {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 500px;
    height: 200px;
    background: radial-gradient(ellipse, {accent_glow} 0%, transparent 70%);
    pointer-events: none;
    z-index: -1;
}}

/* ═══════════════════════════════════════════════
   8. FEATURE CARDS (Home)
═══════════════════════════════════════════════ */
.feat-card {{
    background: {bg_surface};
    backdrop-filter: blur(18px) saturate(1.4);
    -webkit-backdrop-filter: blur(18px) saturate(1.4);
    border: 1px solid {border_color};
    border-radius: 18px;
    padding: 1.8rem 1.6rem;
    height: 100%;
    transition: transform 0.28s cubic-bezier(.34,1.56,.64,1),
                border-color 0.25s ease,
                box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
    animation: fade-up 0.6s ease both;
}}
.feat-card::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, transparent 50%);
    pointer-events: none;
    border-radius: inherit;
}}
.feat-card:hover {{
    transform: translateY(-6px) scale(1.01);
    border-color: {border_hover};
    box-shadow: 0 16px 48px {accent_glow}, 0 4px 16px rgba(0,0,0,0.12);
}}
.feat-icon {{
    font-size: 2.2rem;
    margin-bottom: 0.8rem;
    display: block;
    filter: drop-shadow(0 0 10px {accent_glow});
}}
.feat-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: {text_primary} !important;
    margin-bottom: 0.4rem;
}}
.feat-body {{
    color: {text_secondary};
    font-size: 0.88rem;
    line-height: 1.65;
}}

/* ═══════════════════════════════════════════════
   9. FILE UPLOADER
═══════════════════════════════════════════════ */
div[data-testid="stFileUploader"] {{
    background: {upload_bg} !important;
    backdrop-filter: blur(16px) !important;
    border: 2px dashed {upload_border} !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, background 0.25s ease !important;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color: {upload_hover} !important;
    box-shadow: 0 0 28px {accent_glow} !important;
    background: {bg_surface} !important;
}}

/* ═══════════════════════════════════════════════
   10. BUTTONS
═══════════════════════════════════════════════ */
.stButton > button,
.stDownloadButton > button {{
    background: linear-gradient(135deg, {btn_from} 0%, {btn_to} 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    padding: 0.6rem 1.6rem !important;
    transition: transform 0.2s cubic-bezier(.34,1.56,.64,1),
                box-shadow 0.2s ease,
                opacity 0.2s ease !important;
    box-shadow: 0 4px 18px {accent_glow} !important;
    position: relative;
    overflow: hidden;
}}
.stButton > button::before,
.stDownloadButton > button::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);
    border-radius: inherit;
    pointer-events: none;
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 30px {accent_glow}, 0 2px 8px rgba(0,0,0,0.2) !important;
    opacity: 0.95 !important;
}}
.stButton > button:active,
.stDownloadButton > button:active {{
    transform: scale(0.97) !important;
}}

/* ═══════════════════════════════════════════════
   11. DATAFRAME
═══════════════════════════════════════════════ */
div[data-testid="stDataFrame"] {{
    border: 1px solid {border_color} !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    backdrop-filter: blur(12px) !important;
    animation: fade-up 0.5s ease both;
}}

/* ═══════════════════════════════════════════════
   12. SECTION HEADERS
═══════════════════════════════════════════════ */
.section-header {{
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: {accent} !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 2rem 0 1.1rem;
}}
.section-header::before {{
    content: '';
    display: inline-block;
    width: 18px;
    height: 2px;
    background: linear-gradient(90deg, {accent}, {accent2});
    border-radius: 99px;
    flex-shrink: 0;
}}
.section-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, {border_color}, transparent);
}}

/* ═══════════════════════════════════════════════
   13. PAYWALL CARD
═══════════════════════════════════════════════ */
.paywall-card {{
    background: {paywall_bg};
    backdrop-filter: blur(24px) saturate(1.5);
    -webkit-backdrop-filter: blur(24px) saturate(1.5);
    border: 1px solid {border_color};
    border-radius: 22px;
    padding: 3rem 2.5rem;
    text-align: center;
    max-width: 500px;
    margin: 5vh auto;
    box-shadow: 0 24px 64px {accent_glow}, 0 8px 24px rgba(0,0,0,0.18);
    animation: fade-up 0.5s ease both;
    position: relative;
    overflow: hidden;
}}
.paywall-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {accent}, {accent2}, transparent);
}}
.paywall-lock {{
    font-size: 3.5rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 16px {accent_glow});
    animation: float-icon 3s ease-in-out infinite;
}}
@keyframes float-icon {{
    0%, 100% {{ transform: translateY(0); }}
    50%       {{ transform: translateY(-6px); }}
}}
.paywall-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    color: {text_primary} !important;
    margin-bottom: 0.75rem;
}}
.paywall-body {{
    color: {text_secondary};
    font-size: 0.93rem;
    line-height: 1.75;
}}
.paywall-tag {{
    display: inline-block;
    margin-top: 1.4rem;
    background: linear-gradient(135deg, {btn_from}, {btn_to});
    color: #fff !important;
    padding: 0.45rem 1.4rem;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    box-shadow: 0 4px 18px {accent_glow};
}}

/* ═══════════════════════════════════════════════
   14. LOGIN FORM
═══════════════════════════════════════════════ */
div[data-testid="stForm"] {{
    background: {bg_surface} !important;
    backdrop-filter: blur(24px) saturate(1.4) !important;
    border: 1px solid {border_color} !important;
    border-radius: 18px !important;
    padding: 1.8rem !important;
    animation: fade-up 0.5s ease both;
    box-shadow: 0 16px 48px {accent_glow};
}}

/* ═══════════════════════════════════════════════
   15. INPUT FIELDS
═══════════════════════════════════════════════ */
div[data-testid="stTextInput"] input,
div[data-testid="stPasswordInput"] input {{
    background: {input_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 10px !important;
    color: {text_primary} !important;
    font-family: 'Outfit', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    backdrop-filter: blur(8px) !important;
}}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stPasswordInput"] input:focus {{
    border-color: {accent} !important;
    box-shadow: 0 0 0 3px {accent_glow} !important;
    outline: none !important;
}}

/* ═══════════════════════════════════════════════
   16. ALERTS / INFO BOXES
═══════════════════════════════════════════════ */
div[data-testid="stAlert"] {{
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
    border-left-width: 3px !important;
    animation: fade-up 0.4s ease both;
}}

/* ═══════════════════════════════════════════════
   17. ANIMATIONS
═══════════════════════════════════════════════ */
@keyframes fade-up {{
    from {{ opacity: 0; transform: translateY(18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fade-in {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes slide-right {{
    from {{ opacity: 0; transform: translateX(-16px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}
.anim-delay-1 {{ animation-delay: 0.1s; }}
.anim-delay-2 {{ animation-delay: 0.2s; }}
.anim-delay-3 {{ animation-delay: 0.3s; }}

/* ═══════════════════════════════════════════════
   18. BAR CHART
═══════════════════════════════════════════════ */
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {{
    background: {bg_surface} !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid {border_color} !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    animation: fade-up 0.5s ease both;
}}

/* ═══════════════════════════════════════════════
   19. RADIO (sidebar nav)
═══════════════════════════════════════════════ */
div[data-testid="stRadio"] label {{
    display: flex !important;
    align-items: center !important;
    padding: 0.45rem 0.7rem !important;
    border-radius: 8px !important;
    transition: background 0.2s ease !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
}}
div[data-testid="stRadio"] label:hover {{
    background: {bg_surface2} !important;
}}

/* ═══════════════════════════════════════════════
   20. HIDE STREAMLIT CHROME
═══════════════════════════════════════════════ */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# AUTH CONFIG  (streamlit-authenticator v0.3 / v0.4 schema)
# ─────────────────────────────────────────────────────────────
AUTH_CONFIG: dict = {
    "credentials": {
        "usernames": {
            "rohit": {
                "first_name": "Rohit",
                "last_name": "Appadi",
                "email": "rohit@churniq.ai",
                "password": "password123",
                "logged_in": False,
            },
            "demo": {
                "first_name": "Demo",
                "last_name": "User",
                "email": "demo@churniq.ai",
                "password": "demo456",
                "logged_in": False,
            },
        }
    },
    "cookie": {
        "name": "churniq_auth_cookie",
        "key":  "churniq_super_secret_key_xyz_2024",
        "expiry_days": 1,
    },
}

PRO_USERS: set = {"rohit"}


# ─────────────────────────────────────────────────────────────
# ML HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading prediction engine…")
def load_model():
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
    df.columns = df.columns.str.replace('\ufeff', '', regex=True).str.strip()
    leakage_cols = [
        "CustomerID", "Count", "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "CLTV", "Churn Reason",
    ]
    df = df.drop(columns=leakage_cols, errors="ignore")
    if "Churn Value" not in df.columns:
        raise ValueError(
            "Required column **'Churn Value'** is missing. "
            "Please upload the original **IBM Telco Customer Churn** CSV."
        )
    X = df.drop("Churn Value", axis=1)
    high_cardinality = ["Country", "State", "City", "Zip Code"]
    X = X.drop(columns=high_cardinality, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def run_predict(X: pd.DataFrame, model, scaler):
    X_scaled = scaler.transform(X)
    labels   = model.predict(X_scaled)
    probs    = model.predict_proba(X_scaled)[:, 1]
    return labels, probs


def risk_segment(prob: float) -> str:
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    return "Low Risk"


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(name: str, authenticator) -> str:
    with st.sidebar:
        st.markdown('<p class="sidebar-brand">⚡ ChurnIQ</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-tagline">Customer Intelligence Platform</p>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Theme toggle
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.markdown(
                f'<span style="font-size:0.8rem;color:{"#7ea3c4" if is_dark() else "#375778"}">🌙 Dark mode</span>',
                unsafe_allow_html=True
            )
        with col_t2:
            toggled = st.toggle("", value=st.session_state["dark_mode"], key="theme_toggle", label_visibility="collapsed")
            if toggled != st.session_state["dark_mode"]:
                st.session_state["dark_mode"] = toggled
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # User badge
        first = name.split()[0] if name else name
        st.markdown(
            f'<div class="sidebar-user">'
            f'<span class="sidebar-user-dot"></span>'
            f'<span style="font-size:0.88rem"><strong>{first}</strong></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["🏠  Home", "📂  Upload & Predict", "📊  Results"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        authenticator.logout("🚪  Logout", location="sidebar", key="sidebar_logout")

    return page.split("  ", 1)[-1].strip()


# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────

def page_home(name: str) -> None:
    first = name.split()[0] if name else "there"

    # Hero
    st.markdown(f"""
    <div class="hero-wrapper">
      <div class="hero-glow"></div>
      <div>
        <span class="hero-badge">
          <span class="hero-badge-dot"></span>
          AI-Powered · Real-Time Predictions
        </span>
      </div>
      <div class="hero-title">Predict Churn.<br>Retain Revenue.</div>
      <div class="hero-sub">
        Welcome back, <strong>{first}</strong> — ChurnIQ turns raw customer data
        into actionable risk intelligence in seconds.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3, gap="large")
    cards = [
        ("📂", "Upload CSV", "Drop the IBM Telco CSV and ChurnIQ handles cleaning, encoding, and alignment automatically."),
        ("🤖", "ML Prediction", "A trained ensemble model scores every customer with a precise churn probability."),
        ("📊", "Act on Insights", "Download risk-segmented results and focus retention spend where it matters most."),
    ]
    for i, (col, (icon, title, body)) in enumerate(zip([c1, c2, c3], cards)):
        with col:
            st.markdown(
                f'<div class="feat-card anim-delay-{i+1}">'
                f'<span class="feat-icon">{icon}</span>'
                f'<div class="feat-title">{title}</div>'
                f'<div class="feat-body">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("**How to use →** Open **Upload & Predict** in the sidebar, upload your CSV, then view results in **Results**.")


# ─────────────────────────────────────────────────────────────
# PAGE: UPLOAD & PREDICT
# ─────────────────────────────────────────────────────────────

def page_upload(username: str) -> None:
    # Paywall
    if username not in PRO_USERS:
        st.markdown("""
        <div class="paywall-card">
          <div class="paywall-lock">🔒</div>
          <div class="paywall-title">Pro Feature</div>
          <div class="paywall-body">
            Churn predictions are available on the <strong>Pro Plan</strong>.<br><br>
            Upgrade to unlock batch CSV predictions, risk segmentation,
            revenue-at-risk analytics, and priority support.
          </div>
          <div class="paywall-tag">⚡ Upgrade to Pro</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    model, scaler, feature_columns = load_model()

    # Page header
    st.markdown("""
    <div style="animation: fade-up 0.5s ease both;">
      <h2 style="margin-bottom:0.25rem">📂 Upload Customer Data</h2>
      <p style="margin-top:0">Upload the <strong>original IBM Telco Customer Churn</strong> CSV file.
      The file must include a <code>Churn Value</code> column.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your CSV here, or click to browse",
        type=["csv"],
        help="Standard IBM Telco Customer Churn dataset format.",
    )

    if not uploaded_file:
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not parse the CSV file: `{exc}`")
        return

    if raw_df.empty:
        st.warning("The uploaded file appears to be empty.")
        return

    with st.spinner("Preprocessing data…"):
        try:
            X = preprocess_data(raw_df.copy(), feature_columns)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Preprocessing failed: `{exc}`")
            return

    with st.spinner("Running predictions…"):
        try:
            labels, probs = run_predict(X, model, scaler)
        except Exception as exc:
            st.error(f"Prediction error: `{exc}`")
            return

    result_df = raw_df.copy()
    result_df.columns = result_df.columns.str.replace('\ufeff', '', regex=True).str.strip()
    result_df["Predicted_Churn"]   = labels
    result_df["Churn_Probability"] = probs
    result_df["Risk Segment"]      = result_df["Churn_Probability"].apply(risk_segment)

    st.session_state["result_df"] = result_df
    st.success("✅ Predictions complete! Navigate to **Results** in the sidebar.")


# ─────────────────────────────────────────────────────────────
# PAGE: RESULTS
# ─────────────────────────────────────────────────────────────

def page_results() -> None:
    st.markdown("""
    <div style="animation: fade-up 0.5s ease both;">
      <h2 style="margin-bottom:0.1rem">📊 Prediction Results</h2>
    </div>
    """, unsafe_allow_html=True)

    if "result_df" not in st.session_state:
        st.info("No predictions yet. Upload a CSV on the **Upload & Predict** page first.")
        return

    result_df: pd.DataFrame = st.session_state["result_df"]

    total_customers = len(result_df)
    high_risk_df    = result_df[result_df["Risk Segment"] == "High Risk"]
    medium_risk_df  = result_df[result_df["Risk Segment"] == "Medium Risk"]
    high_risk_count = len(high_risk_df)
    med_risk_count  = len(medium_risk_df)

    rev_col = next(
        (c for c in result_df.columns if c.lower().replace(" ", "") == "monthlycharges"),
        None,
    )
    revenue_at_risk = high_risk_df[rev_col].sum() if rev_col else None

    # Metrics
    st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Total Customers",  f"{total_customers:,}")
    c2.metric("High Risk 🔴",     f"{high_risk_count:,}")
    c3.metric("Medium Risk 🟡",   f"{med_risk_count:,}")
    c4.metric(
        "Revenue at Risk 💸",
        f"${revenue_at_risk:,.0f}" if revenue_at_risk is not None else "N/A",
        help="Sum of Monthly Charges for High Risk customers." if rev_col else "Monthly Charges column not found.",
    )

    # Risk distribution
    st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
    dist = (
        result_df["Risk Segment"]
        .value_counts()
        .reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
        .reset_index()
    )
    dist.columns = ["Risk Segment", "Count"]
    st.bar_chart(dist.set_index("Risk Segment"), color="#0ea5e9", use_container_width=True)

    # Preview
    st.markdown('<div class="section-header">Predictions Preview — first 20 rows</div>', unsafe_allow_html=True)
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

    # Export
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Full Predictions CSV",
        data=csv_bytes,
        file_name="churniq_predictions.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# LOGIN BRANDING
# ─────────────────────────────────────────────────────────────

def _render_login_branding() -> None:
    dark = is_dark()
    accent      = "#38bdf8" if dark else "#0284c7"
    accent_glow = "rgba(56,189,248,0.20)" if dark else "rgba(2,132,199,0.14)"
    text_sub    = "#7ea3c4" if dark else "#375778"
    text_muted  = "#3d5a78" if dark else "#8faec8"
    hero_grad   = "linear-gradient(135deg,#e0f2fe 0%,#38bdf8 40%,#6ee7f7 100%)" if dark else "linear-gradient(135deg,#0c1929 0%,#0284c7 50%,#0ea5e9 100%)"

    st.markdown(f"""
    <div style="text-align:center;margin-top:2rem;animation:fade-up 0.6s ease both;">
      <div style="display:inline-block;font-family:Syne,sans-serif;font-size:3rem;font-weight:800;
                  background:{hero_grad};-webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;letter-spacing:-1.5px;
                  filter:drop-shadow(0 0 32px {accent_glow});">
        ⚡ ChurnIQ
      </div>
      <div style="color:{text_sub};margin-top:0.4rem;font-size:0.95rem;font-family:Outfit,sans-serif;">
        Customer Intelligence Platform
      </div>
      <div style="color:{text_muted};margin-top:0.3rem;font-size:0.8rem;letter-spacing:0.06em;">
        ML-powered churn prediction · Sign in to continue
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # Inject theme CSS first
    inject_css()

    authenticator = stauth.Authenticate(
        AUTH_CONFIG["credentials"],
        AUTH_CONFIG["cookie"]["name"],
        AUTH_CONFIG["cookie"]["key"],
        AUTH_CONFIG["cookie"]["expiry_days"],
    )

    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    name        = st.session_state.get("name", "")
    username    = st.session_state.get("username", "")

    if auth_status is False:
        st.error("❌ Incorrect username or password. Please try again.")
        _render_login_branding()
        return

    if auth_status is None:
        _render_login_branding()
        return

    page = render_sidebar(name, authenticator)

    if page == "Home":
        page_home(name)
    elif page == "Upload & Predict":
        page_upload(username)
    elif page == "Results":
        page_results()


if __name__ == "__main__":
    main()
"""
ChurnIQ · Customer Churn Prediction System
Production-ready Streamlit app — Ultra Premium UI Edition v2.

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
# THEME STATE
# ─────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True


def is_dark() -> bool:
    return st.session_state.get("dark_mode", True)


# ─────────────────────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────────────────────
def inject_css() -> None:
    dark = is_dark()

    if dark:
        bg_root        = "#03070f"
        bg_base        = "#060d1a"
        bg_surface     = "rgba(10, 18, 35, 0.75)"
        bg_surface2    = "rgba(14, 24, 46, 0.65)"
        bg_glass       = "rgba(12, 20, 40, 0.70)"
        border_color   = "rgba(56, 189, 248, 0.12)"
        border_hover   = "rgba(56, 189, 248, 0.50)"
        border_active  = "rgba(56, 189, 248, 0.80)"
        text_primary   = "#eef4ff"
        text_secondary = "#6d9ab5"
        text_muted     = "#2e4a63"
        accent         = "#38bdf8"
        accent2        = "#7dd3fc"
        accent3        = "#0ea5e9"
        accent_glow    = "rgba(56,189,248,0.22)"
        accent_glow_sm = "rgba(56,189,248,0.12)"
        indigo_glow    = "rgba(99,102,241,0.12)"
        btn_from       = "#0ea5e9"
        btn_to         = "#1d4ed8"
        metric_val     = "#38bdf8"
        hero_grad      = "linear-gradient(135deg, #e0f2fe 0%, #38bdf8 45%, #818cf8 100%)"
        login_bg       = "rgba(8, 14, 28, 0.85)"
        sidebar_bg     = "rgba(4, 9, 20, 0.96)"
        upload_bg      = "rgba(10, 18, 35, 0.60)"
        upload_border  = "rgba(56,189,248,0.28)"
        upload_hover   = "rgba(56,189,248,0.60)"
        paywall_bg     = "rgba(8,14,28,0.88)"
        input_bg       = "rgba(8,14,28,0.90)"
        scrollbar_bg   = "#060d1a"
        scrollbar_th   = "#1a3048"
        orb1           = "rgba(56,189,248,0.14)"
        orb2           = "rgba(99,102,241,0.10)"
        orb3           = "rgba(14,165,233,0.08)"
        shine          = "rgba(255,255,255,0.04)"
        shine2         = "rgba(255,255,255,0.02)"
        # Toggle-specific
        toggle_track   = "rgba(56,189,248,0.18)"
        toggle_knob    = "#eef4ff"
    else:
        bg_root        = "#eef4ff"
        bg_base        = "#f4f8ff"
        bg_surface     = "rgba(255,255,255,0.80)"
        bg_surface2    = "rgba(238,244,255,0.85)"
        bg_glass       = "rgba(255,255,255,0.75)"
        border_color   = "rgba(14,165,233,0.16)"
        border_hover   = "rgba(14,165,233,0.48)"
        border_active  = "rgba(14,165,233,0.80)"
        text_primary   = "#071628"
        text_secondary = "#3a6080"
        text_muted     = "#9ab8cc"
        accent         = "#0284c7"
        accent2        = "#0ea5e9"
        accent3        = "#0369a1"
        accent_glow    = "rgba(2,132,199,0.16)"
        accent_glow_sm = "rgba(2,132,199,0.08)"
        indigo_glow    = "rgba(99,102,241,0.08)"
        btn_from       = "#0284c7"
        btn_to         = "#1d4ed8"
        metric_val     = "#0284c7"
        hero_grad      = "linear-gradient(135deg, #0c1929 0%, #0284c7 50%, #38bdf8 100%)"
        login_bg       = "rgba(240,248,255,0.90)"
        sidebar_bg     = "rgba(230,242,255,0.97)"
        upload_bg      = "rgba(224,242,254,0.60)"
        upload_border  = "rgba(14,165,233,0.32)"
        upload_hover   = "rgba(14,165,233,0.58)"
        paywall_bg     = "rgba(224,242,254,0.85)"
        input_bg       = "rgba(255,255,255,0.92)"
        scrollbar_bg   = "#dbeafe"
        scrollbar_th   = "#93c5fd"
        orb1           = "rgba(14,165,233,0.12)"
        orb2           = "rgba(99,102,241,0.08)"
        orb3           = "rgba(2,132,199,0.06)"
        shine          = "rgba(255,255,255,0.55)"
        shine2         = "rgba(255,255,255,0.30)"
        # Toggle-specific
        toggle_track   = "rgba(14,165,233,0.20)"
        toggle_knob    = "#071628"

    st.markdown(f"""
<style>
/* ══════════════════════════════════════════════════════════
   FONTS
══════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ══════════════════════════════════════════════════════════
   CUSTOM PROPERTIES
══════════════════════════════════════════════════════════ */
:root {{
  --accent:         {accent};
  --accent2:        {accent2};
  --accent-glow:    {accent_glow};
  --border:         {border_color};
  --border-hover:   {border_hover};
  --text-primary:   {text_primary};
  --text-secondary: {text_secondary};
  --text-muted:     {text_muted};
  --surface:        {bg_surface};
  --surface2:       {bg_surface2};
  --glass:          {bg_glass};
  --shine:          {shine};
  --shine2:         {shine2};
  --btn-from:       {btn_from};
  --btn-to:         {btn_to};
  --radius-sm:      10px;
  --radius-md:      16px;
  --radius-lg:      22px;
  --radius-xl:      28px;
}}

/* ══════════════════════════════════════════════════════════
   BASE RESET
══════════════════════════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"] {{
  font-family: 'Outfit', sans-serif;
  color: {text_primary};
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}

/* ══════════════════════════════════════════════════════════
   ANIMATED BACKGROUND
══════════════════════════════════════════════════════════ */
.stApp {{
  background-color: {bg_root};
  min-height: 100vh;
  overflow-x: hidden;
  position: relative;
  transition: background-color 0.6s ease;
}}

/* Grain texture overlay */
.stApp::before {{
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
  opacity: 0.6;
}}

/* Ambient orbs */
.stApp::after {{
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 70% 55% at 10%  8%,  {orb1} 0%, transparent 60%),
    radial-gradient(ellipse 55% 45% at 90% 90%,  {orb2} 0%, transparent 60%),
    radial-gradient(ellipse 45% 35% at 55% 45%,  {orb3} 0%, transparent 60%),
    radial-gradient(ellipse 30% 25% at 80% 20%,  {indigo_glow} 0%, transparent 55%);
  pointer-events: none;
  z-index: 0;
  animation: ambient-shift 20s ease-in-out infinite alternate;
}}
@keyframes ambient-shift {{
  0%   {{ opacity: 0.7; transform: scale(1);    }}
  50%  {{ opacity: 1.0; transform: scale(1.04); }}
  100% {{ opacity: 0.8; transform: scale(0.98); }}
}}

/* Ensure content sits above background layers */
section[data-testid="stMain"] > div,
section[data-testid="stSidebar"] > div {{
  position: relative;
  z-index: 1;
}}

/* ══════════════════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════════════════ */
::-webkit-scrollbar              {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track        {{ background: {scrollbar_bg}; }}
::-webkit-scrollbar-thumb        {{ background: {scrollbar_th}; border-radius: 99px; }}
::-webkit-scrollbar-thumb:hover  {{ background: {accent}; }}

/* ══════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
  background: {sidebar_bg} !important;
  backdrop-filter: blur(32px) saturate(1.6) !important;
  -webkit-backdrop-filter: blur(32px) saturate(1.6) !important;
  border-right: 1px solid {border_color} !important;
  transition: background 0.5s ease !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 1.4rem !important; }}
section[data-testid="stSidebar"] * {{
  color: {text_secondary} !important;
  transition: color 0.4s ease !important;
}}
section[data-testid="stSidebar"] strong {{ color: {text_primary} !important; }}

/* ─── Sidebar brand / tag / divider / user badge ─── */
.sb-brand {{
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 800;
  background: linear-gradient(135deg, {accent} 0%, {accent2} 60%, #818cf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.6px;
  display: inline-block;
  line-height: 1;
}}
.sb-tag {{
  font-size: 0.68rem;
  color: {text_muted} !important;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  font-weight: 500;
  margin-top: 2px;
}}
.sb-hr {{
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, {border_color} 40%, {border_color} 60%, transparent 100%);
  border: none;
  margin: 0.8rem 0;
}}
.sb-user {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.55rem 0.85rem;
  background: {bg_surface2};
  border: 1px solid {border_color};
  border-radius: var(--radius-sm);
  font-size: 0.86rem;
  backdrop-filter: blur(12px);
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}}
.sb-user:hover {{
  border-color: {border_hover};
  box-shadow: 0 0 16px {accent_glow_sm};
}}
.sb-dot {{
  width: 7px; height: 7px;
  background: #22c55e;
  border-radius: 50%;
  box-shadow: 0 0 8px #22c55e88;
  flex-shrink: 0;
  animation: dot-pulse 2.5s ease-in-out infinite;
}}
@keyframes dot-pulse {{
  0%, 100% {{ box-shadow: 0 0 6px #22c55e88; }}
  50%       {{ box-shadow: 0 0 14px #22c55ecc; }}
}}

/* ══════════════════════════════════════════════════════════
   FIX 1 — TOGGLE BUTTON — always visible in both themes
══════════════════════════════════════════════════════════ */
/* Track (off state) */
div[data-testid="stToggle"] > label > div:first-child {{
  background: {toggle_track} !important;
  border: 1.5px solid {border_hover} !important;
  border-radius: 99px !important;
  transition:
    background    0.3s ease,
    border-color  0.3s ease,
    box-shadow    0.3s ease !important;
}}
/* Track (on state) */
div[data-testid="stToggle"] > label > input:checked + div {{
  background: linear-gradient(135deg, {btn_from}, {btn_to}) !important;
  border-color: {accent} !important;
  box-shadow: 0 0 14px {accent_glow} !important;
}}
/* Knob (off state) */
div[data-testid="stToggle"] > label > div:first-child > div {{
  background: {toggle_knob} !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.40) !important;
  transition:
    transform   0.25s cubic-bezier(.34,1.56,.64,1),
    background  0.3s ease !important;
}}
/* Knob (on state) */
div[data-testid="stToggle"] > label > input:checked + div > div {{
  background: #ffffff !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.35) !important;
}}
/* Label text */
div[data-testid="stToggle"] label span {{
  color: {text_secondary} !important;
  font-size: 0.82rem !important;
  font-family: 'Outfit', sans-serif !important;
  transition: color 0.3s ease !important;
}}

/* ══════════════════════════════════════════════════════════
   TYPOGRAPHY
══════════════════════════════════════════════════════════ */
h1, h2, h3, h4 {{
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  color: {text_primary} !important;
  letter-spacing: -0.5px;
  transition: color 0.4s ease;
}}
p, li, span {{
  color: {text_secondary};
  line-height: 1.72;
  transition: color 0.4s ease;
}}

/* ══════════════════════════════════════════════════════════
   GLASS SYSTEM
══════════════════════════════════════════════════════════ */
.glass {{
  background: var(--glass);
  backdrop-filter: blur(24px) saturate(1.6);
  -webkit-backdrop-filter: blur(24px) saturate(1.6);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  position: relative;
  overflow: hidden;
  transition:
    transform     0.3s cubic-bezier(.34,1.56,.64,1),
    border-color  0.3s ease,
    box-shadow    0.3s ease;
}}
.glass::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, var(--shine) 0%, transparent 55%);
  pointer-events: none;
  border-radius: inherit;
}}
.glass::after {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent 10%, {accent} 50%, transparent 90%);
  opacity: 0;
  transition: opacity 0.3s ease;
}}
.glass:hover {{ border-color: var(--border-hover); }}
.glass:hover::after {{ opacity: 0.6; }}
.glass-lift:hover {{
  transform: translateY(-5px) scale(1.008);
  box-shadow: 0 18px 52px var(--accent-glow), 0 4px 16px rgba(0,0,0,0.15);
}}

/* ══════════════════════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════════════════════ */
div[data-testid="metric-container"] {{
  background: {bg_glass} !important;
  backdrop-filter: blur(24px) saturate(1.6) !important;
  -webkit-backdrop-filter: blur(24px) saturate(1.6) !important;
  border: 1px solid {border_color} !important;
  border-radius: var(--radius-md) !important;
  padding: 1.35rem 1.5rem !important;
  position: relative;
  overflow: hidden;
  transition:
    transform     0.3s cubic-bezier(.34,1.56,.64,1),
    border-color  0.3s ease,
    box-shadow    0.3s ease !important;
  animation: reveal 0.55s cubic-bezier(.22,1,.36,1) both;
}}
div[data-testid="metric-container"]::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, {shine} 0%, transparent 55%);
  pointer-events: none;
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
  transform: translateY(-5px) !important;
  border-color: {border_hover} !important;
  box-shadow: 0 14px 40px {accent_glow}, 0 4px 16px rgba(0,0,0,0.12) !important;
}}
div[data-testid="metric-container"]:hover::after {{ opacity: 1; }}
div[data-testid="metric-container"] label {{
  color: {text_muted} !important;
  font-size: 0.70rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.13em !important;
  text-transform: uppercase !important;
  font-family: 'Outfit', sans-serif !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
  font-family: 'Syne', sans-serif !important;
  font-size: 2.05rem !important;
  font-weight: 800 !important;
  color: {metric_val} !important;
  letter-spacing: -0.5px;
  text-shadow: 0 0 28px {accent_glow};
}}

/* ══════════════════════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════════════════════ */
div[data-testid="stFileUploader"] {{
  background: {upload_bg} !important;
  backdrop-filter: blur(18px) !important;
  border: 1.5px dashed {upload_border} !important;
  border-radius: var(--radius-lg) !important;
  padding: 1.8rem 1.5rem !important;
  transition:
    border-color  0.25s ease,
    box-shadow    0.25s ease,
    background    0.25s ease !important;
  position: relative;
}}
div[data-testid="stFileUploader"]:hover {{
  border-color: {upload_hover} !important;
  box-shadow: 0 0 32px {accent_glow}, inset 0 0 24px {accent_glow_sm} !important;
  background: {bg_glass} !important;
}}

/* ══════════════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════════════ */
.stButton > button,
.stDownloadButton > button {{
  background: linear-gradient(135deg, {btn_from} 0%, {btn_to} 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.05em !important;
  padding: 0.6rem 1.65rem !important;
  position: relative;
  overflow: hidden;
  transition:
    transform   0.22s cubic-bezier(.34,1.56,.64,1),
    box-shadow  0.22s ease,
    opacity     0.22s ease !important;
  box-shadow: 0 4px 20px {accent_glow} !important;
}}
.stButton > button::before,
.stDownloadButton > button::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.18) 0%, transparent 60%);
  pointer-events: none;
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
  transform: translateY(-2px) scale(1.04) !important;
  box-shadow: 0 10px 34px {accent_glow}, 0 2px 8px rgba(0,0,0,0.2) !important;
}}
.stButton > button:active,
.stDownloadButton > button:active {{
  transform: scale(0.96) !important;
  box-shadow: 0 2px 10px {accent_glow_sm} !important;
}}

/* ══════════════════════════════════════════════════════════
   INPUT FIELDS
══════════════════════════════════════════════════════════ */
div[data-testid="stTextInput"] input,
div[data-testid="stPasswordInput"] input {{
  background: {input_bg} !important;
  border: 1px solid {border_color} !important;
  border-radius: var(--radius-sm) !important;
  color: {text_primary} !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.92rem !important;
  padding: 0.65rem 0.95rem !important;
  transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
  backdrop-filter: blur(8px) !important;
}}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stPasswordInput"] input:focus {{
  border-color: {accent} !important;
  box-shadow: 0 0 0 3px {accent_glow_sm}, 0 0 0 1px {accent} !important;
  outline: none !important;
}}
div[data-testid="stTextInput"] label,
div[data-testid="stPasswordInput"] label {{
  color: {text_muted} !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  font-weight: 600 !important;
}}

/* ══════════════════════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════════════════════ */
div[data-testid="stDataFrame"] {{
  background: {bg_glass} !important;
  backdrop-filter: blur(16px) !important;
  border: 1px solid {border_color} !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  box-shadow: 0 4px 24px {accent_glow_sm};
  animation: reveal 0.5s cubic-bezier(.22,1,.36,1) both;
}}

/* ══════════════════════════════════════════════════════════
   CHARTS
══════════════════════════════════════════════════════════ */
div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {{
  background: {bg_glass} !important;
  backdrop-filter: blur(14px) !important;
  border: 1px solid {border_color} !important;
  border-radius: var(--radius-md) !important;
  padding: 1.2rem !important;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
  animation: reveal 0.5s cubic-bezier(.22,1,.36,1) both;
}}
div[data-testid="stVegaLiteChart"]:hover,
div[data-testid="stArrowVegaLiteChart"]:hover {{
  border-color: {border_hover} !important;
  box-shadow: 0 8px 32px {accent_glow} !important;
}}

/* ══════════════════════════════════════════════════════════
   ALERTS
══════════════════════════════════════════════════════════ */
div[data-testid="stAlert"] {{
  background: {bg_glass} !important;
  backdrop-filter: blur(12px) !important;
  border-radius: var(--radius-sm) !important;
  border-left-width: 3px !important;
  animation: reveal 0.4s cubic-bezier(.22,1,.36,1) both;
}}

/* ══════════════════════════════════════════════════════════
   FIX 2 — LOGIN FORM — remove default Streamlit text
══════════════════════════════════════════════════════════ */
div[data-testid="stForm"] > div:first-child > div:first-child p,
div[data-testid="stForm"] > div > div > div > p,
div[data-testid="stForm"] h2,
div[data-testid="stForm"] h3 {{
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}}
div[data-testid="stForm"] > div > div[data-testid="stMarkdownContainer"] {{
  display: none !important;
}}

/* Login form glass panel */
div[data-testid="stForm"] {{
  background: {login_bg} !important;
  backdrop-filter: blur(32px) saturate(1.8) !important;
  -webkit-backdrop-filter: blur(32px) saturate(1.8) !important;
  border: 1px solid {border_color} !important;
  border-radius: var(--radius-xl) !important;
  padding: 2.2rem 2rem !important;
  box-shadow:
    0 32px 80px rgba(0,0,0,0.30),
    0 8px 32px {accent_glow_sm},
    inset 0 1px 0 {shine} !important;
  animation: login-float 0.7s cubic-bezier(.22,1,.36,1) both;
  position: relative;
  overflow: hidden;
  max-width: 440px;
  margin: 0 auto !important;
}}
div[data-testid="stForm"]::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, {shine} 0%, transparent 55%);
  pointer-events: none;
  border-radius: inherit;
}}
div[data-testid="stForm"]::after {{
  content: '';
  position: absolute;
  top: 0; left: 15%; right: 15%;
  height: 1px;
  background: linear-gradient(90deg, transparent, {accent2}, transparent);
  opacity: 0.7;
}}
@keyframes login-float {{
  from {{ opacity: 0; transform: translateY(32px) scale(0.96); filter: blur(6px); }}
  to   {{ opacity: 1; transform: translateY(0)    scale(1);    filter: blur(0);   }}
}}

/* ══════════════════════════════════════════════════════════
   RADIO (sidebar nav)
══════════════════════════════════════════════════════════ */
div[data-testid="stRadio"] label {{
  padding: 0.48rem 0.75rem !important;
  border-radius: 8px !important;
  transition: background 0.2s ease, color 0.2s ease !important;
  font-size: 0.88rem !important;
  cursor: pointer !important;
}}
div[data-testid="stRadio"] label:hover {{
  background: {bg_surface2} !important;
}}

/* ══════════════════════════════════════════════════════════
   KEYFRAMES
══════════════════════════════════════════════════════════ */
@keyframes reveal {{
  from {{ opacity: 0; transform: translateY(20px) scale(0.98); filter: blur(4px); }}
  to   {{ opacity: 1; transform: translateY(0)    scale(1);    filter: blur(0);   }}
}}
@keyframes reveal-right {{
  from {{ opacity: 0; transform: translateX(-20px); filter: blur(3px); }}
  to   {{ opacity: 1; transform: translateX(0);     filter: blur(0);   }}
}}
@keyframes float-up-down {{
  0%, 100% {{ transform: translateY(0px);  }}
  50%       {{ transform: translateY(-8px); }}
}}
@keyframes glow-pulse {{
  0%, 100% {{ opacity: 0.6; }}
  50%       {{ opacity: 1.0; }}
}}
@keyframes badge-in {{
  from {{ opacity: 0; transform: translateY(-10px) scale(0.9); }}
  to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}

.d1 {{ animation-delay: 0.08s; }}
.d2 {{ animation-delay: 0.18s; }}
.d3 {{ animation-delay: 0.28s; }}
.d4 {{ animation-delay: 0.38s; }}

/* ══════════════════════════════════════════════════════════
   FIX 3 — PAGE TRANSITION — smooth post-login content reveal
══════════════════════════════════════════════════════════ */
section[data-testid="stMain"] {{
  animation: page-enter 0.55s cubic-bezier(.22,1,.36,1) both !important;
}}
@keyframes page-enter {{
  from {{
    opacity: 0;
    transform: translateY(12px);
    filter: blur(3px);
  }}
  to {{
    opacity: 1;
    transform: translateY(0);
    filter: blur(0);
  }}
}}
section[data-testid="stSidebar"] {{
  animation: sidebar-enter 0.45s cubic-bezier(.22,1,.36,1) 0.05s both !important;
}}
@keyframes sidebar-enter {{
  from {{ opacity: 0; transform: translateX(-10px); }}
  to   {{ opacity: 1; transform: translateX(0); }}
}}

/* ══════════════════════════════════════════════════════════
   HERO SECTION
══════════════════════════════════════════════════════════ */
.hero-wrap {{
  text-align: center;
  padding: 4rem 1rem 3rem;
  position: relative;
  animation: reveal 0.8s cubic-bezier(.22,1,.36,1) both;
}}
.hero-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  background: {bg_surface};
  border: 1px solid {border_color};
  border-radius: 99px;
  padding: 0.32rem 1.1rem;
  font-size: 0.70rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: {accent} !important;
  font-weight: 600;
  margin-bottom: 1.5rem;
  backdrop-filter: blur(14px);
  animation: badge-in 0.7s cubic-bezier(.22,1,.36,1) 0.2s both;
}}
.hero-badge-dot {{
  width: 6px; height: 6px;
  background: {accent};
  border-radius: 50%;
  box-shadow: 0 0 8px {accent};
  animation: glow-pulse 2s ease-in-out infinite;
}}
.hero-title {{
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.6rem, 5.5vw, 4.4rem);
  font-weight: 800;
  background: {hero_grad};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.08;
  letter-spacing: -2px;
  margin-bottom: 1.1rem;
  text-shadow: none;
  animation: reveal 0.7s cubic-bezier(.22,1,.36,1) 0.15s both;
}}
.hero-sub {{
  font-size: clamp(0.88rem, 1.8vw, 1.08rem);
  color: {text_secondary};
  max-width: 520px;
  margin: 0 auto 0.5rem;
  line-height: 1.8;
  animation: reveal 0.7s cubic-bezier(.22,1,.36,1) 0.25s both;
}}
.hero-glow-blob {{
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 600px; height: 250px;
  background: radial-gradient(ellipse, {accent_glow} 0%, transparent 68%);
  pointer-events: none;
  z-index: -1;
  animation: glow-pulse 4s ease-in-out infinite;
}}

/* ══════════════════════════════════════════════════════════
   FEATURE CARDS
══════════════════════════════════════════════════════════ */
.feat-card {{
  background: {bg_glass};
  backdrop-filter: blur(22px) saturate(1.5);
  -webkit-backdrop-filter: blur(22px) saturate(1.5);
  border: 1px solid {border_color};
  border-radius: var(--radius-lg);
  padding: 1.9rem 1.7rem;
  height: 100%;
  position: relative;
  overflow: hidden;
  transition:
    transform    0.32s cubic-bezier(.34,1.56,.64,1),
    border-color 0.28s ease,
    box-shadow   0.28s ease;
  animation: reveal 0.65s cubic-bezier(.22,1,.36,1) both;
}}
.feat-card::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, {shine} 0%, transparent 55%);
  pointer-events: none;
  border-radius: inherit;
}}
.feat-card::after {{
  content: '';
  position: absolute;
  bottom: 0; left: 20%; right: 20%;
  height: 1px;
  background: linear-gradient(90deg, transparent, {border_color}, transparent);
}}
.feat-card:hover {{
  transform: translateY(-8px) scale(1.012);
  border-color: {border_hover};
  box-shadow: 0 20px 56px {accent_glow}, 0 6px 20px rgba(0,0,0,0.14);
}}
.feat-card:hover::after {{
  background: linear-gradient(90deg, transparent, {accent}, transparent);
}}
.feat-icon {{
  font-size: 2.4rem;
  display: block;
  margin-bottom: 1rem;
  filter: drop-shadow(0 0 14px {accent_glow});
  transition: transform 0.3s cubic-bezier(.34,1.56,.64,1), filter 0.3s ease;
}}
.feat-card:hover .feat-icon {{
  transform: scale(1.15) rotate(-5deg);
  filter: drop-shadow(0 0 22px {accent_glow});
}}
.feat-title {{
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: {text_primary} !important;
  margin-bottom: 0.55rem;
  letter-spacing: -0.3px;
}}
.feat-body {{
  color: {text_secondary};
  font-size: 0.86rem;
  line-height: 1.7;
}}

/* ══════════════════════════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════════════════════════ */
.sec-head {{
  display: flex;
  align-items: center;
  gap: 0.7rem;
  margin: 2.2rem 0 1.2rem;
  animation: reveal-right 0.5s cubic-bezier(.22,1,.36,1) both;
}}
.sec-head-bar {{
  width: 3px; height: 18px;
  background: linear-gradient(180deg, {accent}, {accent2});
  border-radius: 99px;
  box-shadow: 0 0 10px {accent_glow};
  flex-shrink: 0;
}}
.sec-head-text {{
  font-family: 'Syne', sans-serif;
  font-size: 0.70rem;
  font-weight: 700;
  color: {accent} !important;
  text-transform: uppercase;
  letter-spacing: 0.18em;
}}
.sec-head-line {{
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, {border_color} 0%, transparent 100%);
}}

/* ══════════════════════════════════════════════════════════
   PAYWALL CARD
══════════════════════════════════════════════════════════ */
.pw-card {{
  background: {paywall_bg};
  backdrop-filter: blur(30px) saturate(1.7);
  -webkit-backdrop-filter: blur(30px) saturate(1.7);
  border: 1px solid {border_color};
  border-radius: var(--radius-xl);
  padding: 3.5rem 2.8rem;
  text-align: center;
  max-width: 480px;
  margin: 5vh auto;
  box-shadow:
    0 32px 72px {accent_glow},
    0 12px 32px rgba(0,0,0,0.2),
    inset 0 1px 0 {shine};
  position: relative;
  overflow: hidden;
  animation: login-float 0.65s cubic-bezier(.22,1,.36,1) both;
}}
.pw-card::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, {shine} 0%, transparent 50%);
  pointer-events: none;
}}
.pw-card::after {{
  content: '';
  position: absolute;
  top: 0; left: 10%; right: 10%;
  height: 2px;
  background: linear-gradient(90deg, transparent, {accent}, {accent2}, transparent);
}}
.pw-icon {{
  font-size: 4rem;
  display: block;
  margin-bottom: 1rem;
  filter: drop-shadow(0 0 20px {accent_glow});
  animation: float-up-down 3.5s ease-in-out infinite;
}}
.pw-title {{
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 800;
  color: {text_primary} !important;
  margin-bottom: 0.8rem;
  letter-spacing: -0.5px;
}}
.pw-body {{
  color: {text_secondary};
  font-size: 0.92rem;
  line-height: 1.8;
}}
.pw-btn {{
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin-top: 1.6rem;
  background: linear-gradient(135deg, {btn_from}, {btn_to});
  color: #fff !important;
  padding: 0.55rem 1.8rem;
  border-radius: 99px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  box-shadow: 0 6px 24px {accent_glow};
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.pw-btn:hover {{
  transform: translateY(-2px) scale(1.04);
  box-shadow: 0 10px 32px {accent_glow};
}}

/* ══════════════════════════════════════════════════════════
   LOGIN BRANDING
══════════════════════════════════════════════════════════ */
.login-brand {{
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
  animation: reveal 0.8s cubic-bezier(.22,1,.36,1) 0.1s both;
}}
.login-logo {{
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.8rem, 5vw, 4rem);
  font-weight: 800;
  background: {hero_grad};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -2px;
  line-height: 1;
  display: inline-block;
  filter: drop-shadow(0 0 40px {accent_glow});
  animation: glow-pulse 3s ease-in-out infinite;
}}
.login-sub {{
  font-size: 1rem;
  color: {text_secondary};
  margin-top: 0.55rem;
  letter-spacing: 0.02em;
}}
.login-hint {{
  font-size: 0.75rem;
  color: {text_muted};
  margin-top: 0.3rem;
  letter-spacing: 0.06em;
}}
.login-divider {{
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin: 1.6rem auto;
  max-width: 440px;
}}
.login-divider-line {{
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, {border_color}, transparent);
}}
.login-divider-text {{
  font-size: 0.72rem;
  color: {text_muted};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  white-space: nowrap;
}}

/* ══════════════════════════════════════════════════════════
   HIDE STREAMLIT CHROME
══════════════════════════════════════════════════════════ */
#MainMenu, footer              {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# AUTH CONFIG
# ─────────────────────────────────────────────────────────────
AUTH_CONFIG: dict = {
    "credentials": {
        "usernames": {
            "rohit": {
                "first_name": "Rohit",
                "last_name": "Appadi",
                "email": "rohit@churniq.ai",
                "password": "rohit123",
                "logged_in": False,
            },
            "demo": {
                "first_name": "Demo",
                "last_name": "User",
                "email": "demo@churniq.ai",
                "password": "demo456",
                "logged_in": False,
            },
            "mahesh": {
                "first_name": "Mahesh",
                "last_name": "Panda",
                "email": "mahesh@churniq.ai",
                "password": "mahesh123",
                "logged_in": False,
            },
            "karan": {
                "first_name": "Karan",
                "last_name": "Pandit",
                "email": "karan@churniq.ai",
                "password": "karan123",
                "logged_in": False,
            },
            "areen": {
                "first_name": "Areen",
                "last_name": "Pal",
                "email": "areen@churniq.ai",
                "password": "areen123",
                "logged_in": False,
            },
            "devang": {
                "first_name": "Devang",
                "last_name": "Borude",
                "email": "devang@churniq.ai",
                "password": "devang123",
                "logged_in": False,
            },
        }
    },
    "cookie": {
        "name": "churniq_auth_cookie",
        "key": "churniq_super_secret_key_xyz_2024",
        "expiry_days": 1,
    },
}

PRO_USERS = set(
    k.lower() for k in AUTH_CONFIG["credentials"]["usernames"].keys()
)


# ─────────────────────────────────────────────────────────────
# ML HELPERS  (untouched)
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
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def sec_header(label: str) -> None:
    st.markdown(
        f'<div class="sec-head">'
        f'<span class="sec-head-bar"></span>'
        f'<span class="sec-head-text">{label}</span>'
        f'<span class="sec-head-line"></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(name: str, authenticator) -> str:
    with st.sidebar:
        st.markdown('<div class="sb-brand">⚡ ChurnIQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-tag">Customer Intelligence Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

        # Theme toggle — Fix 1 ensures it is always visible
        col_a, col_b = st.columns([3, 1])
        with col_a:
            lbl = "🌙 Dark mode" if is_dark() else "☀️ Light mode"
            st.markdown(f'<span style="font-size:0.82rem">{lbl}</span>', unsafe_allow_html=True)
        with col_b:
            toggled = st.toggle(
                "", value=st.session_state["dark_mode"],
                key="theme_toggle", label_visibility="collapsed"
            )
            if toggled != st.session_state["dark_mode"]:
                st.session_state["dark_mode"] = toggled
                st.rerun()

        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

        first = name.split()[0] if name else name
        st.markdown(
            f'<div class="sb-user">'
            f'<span class="sb-dot"></span>'
            f'<span><strong>{first}</strong></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["🏠  Home", "📂  Upload & Predict", "📊  Results"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)
        authenticator.logout("🚪  Logout", location="sidebar", key="sidebar_logout")

    return page.split("  ", 1)[-1].strip()


# ─────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────

def page_home(name: str) -> None:
    first = name.split()[0] if name else "there"
    accent_name = '#38bdf8' if is_dark() else '#0284c7'

    st.markdown(f"""
    <div class="hero-wrap">
      <div class="hero-glow-blob"></div>
      <div>
        <span class="hero-badge">
          <span class="hero-badge-dot"></span>
          AI-Powered &nbsp;·&nbsp; Real-Time Predictions &nbsp;·&nbsp; Enterprise Grade
        </span>
      </div>
      <div class="hero-title">Predict Churn.<br>Retain Revenue.</div>
      <div class="hero-sub">
        Welcome back,
        <strong style="-webkit-text-fill-color:{accent_name}">{first}</strong> —
        ChurnIQ transforms raw customer data into precision risk intelligence in seconds.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    cards = [
        ("📂", "Upload CSV",
         "Drop the IBM Telco CSV — ChurnIQ handles BOM cleaning, encoding, column alignment, and scaling automatically."),
        ("🤖", "ML Prediction",
         "A trained ensemble model scores every customer with a calibrated churn probability and risk tier."),
        ("📊", "Act on Insights",
         "Download risk-segmented results and focus retention budget precisely where revenue is most at stake."),
    ]
    for i, (col, (icon, title, body)) in enumerate(zip([c1, c2, c3], cards)):
        with col:
            st.markdown(
                f'<div class="feat-card d{i+1}">'
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
    if username not in PRO_USERS:
        st.markdown("""
        <div class="pw-card">
          <span class="pw-icon">🔒</span>
          <div class="pw-title">Pro Feature</div>
          <div class="pw-body">
            Churn predictions are available on the <strong>Pro Plan</strong>.<br><br>
            Upgrade to unlock batch CSV predictions, risk segmentation,
            revenue-at-risk analytics, and priority support.
          </div>
          <span class="pw-btn">⚡ Upgrade to Pro</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    model, scaler, feature_columns = load_model()

    st.markdown("""
    <div style="animation: reveal 0.55s cubic-bezier(.22,1,.36,1) both;">
      <h2 style="margin-bottom:0.3rem">📂 Upload Customer Data</h2>
      <p style="margin-top:0;font-size:0.92rem">
        Upload the <strong>original IBM Telco Customer Churn</strong> CSV.
        Must include a
        <code style="background:rgba(56,189,248,0.1);padding:0.1rem 0.4rem;border-radius:4px">Churn Value</code>
        column.
      </p>
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
    <div style="animation: reveal 0.55s cubic-bezier(.22,1,.36,1) both;">
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

    sec_header("Summary Metrics")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Total Customers",  f"{total_customers:,}")
    c2.metric("High Risk 🔴",     f"{high_risk_count:,}")
    c3.metric("Medium Risk 🟡",   f"{med_risk_count:,}")
    c4.metric(
        "Revenue at Risk 💸",
        f"${revenue_at_risk:,.0f}" if revenue_at_risk is not None else "N/A",
        help="Sum of Monthly Charges for High Risk customers." if rev_col else "Monthly Charges column not found.",
    )

    sec_header("Risk Distribution")
    dist = (
        result_df["Risk Segment"]
        .value_counts()
        .reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
        .reset_index()
    )
    dist.columns = ["Risk Segment", "Count"]
    st.bar_chart(dist.set_index("Risk Segment"), color="#0ea5e9", use_container_width=True)

    sec_header("Predictions Preview — first 20 rows")
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

    sec_header("Export")
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Full Predictions CSV",
        data=csv_bytes,
        file_name="churniq_predictions.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# LOGIN BRANDING  (Fix 2 — renders ABOVE form, no default text)
# ─────────────────────────────────────────────────────────────

def _render_login_branding() -> None:
    dark  = is_dark()
    grad  = "linear-gradient(135deg,#e0f2fe 0%,#38bdf8 45%,#818cf8 100%)" if dark else "linear-gradient(135deg,#0c1929 0%,#0284c7 50%,#38bdf8 100%)"
    sub   = "#6d9ab5" if dark else "#3a6080"
    hint  = "#2e4a63" if dark else "#9ab8cc"
    sep   = "rgba(56,189,248,0.12)" if dark else "rgba(14,165,233,0.14)"

    st.markdown(f"""
    <div class="login-brand">
      <div class="login-logo">⚡ ChurnIQ</div>
      <div class="login-sub" style="color:{sub}">Customer Intelligence Platform</div>
      <div class="login-hint" style="color:{hint}">
        ML-powered churn prediction &nbsp;·&nbsp; Sign in to continue
      </div>
    </div>
    <div class="login-divider">
      <span class="login-divider-line"
            style="background:linear-gradient(90deg,transparent,{sep},transparent)"></span>
      <span class="login-divider-text" style="color:{hint}">Secure Login</span>
      <span class="login-divider-line"
            style="background:linear-gradient(90deg,{sep},transparent)"></span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # CSS always loads first — Fix 3 page-enter animation is defined here
    inject_css()

    authenticator = stauth.Authenticate(
        AUTH_CONFIG["credentials"],
        AUTH_CONFIG["cookie"]["name"],
        AUTH_CONFIG["cookie"]["key"],
        AUTH_CONFIG["cookie"]["expiry_days"],
    )

    auth_status = st.session_state.get("authentication_status")

    # ── Pre-auth: show branding above the form (Fix 2) ──
    if auth_status is not True:
        # Reset the post-login flag so transition fires fresh on next login
        st.session_state["_post_login_rendered"] = False
        _render_login_branding()

    authenticator.login(location="main")

    # Re-read session after the login widget has rendered
    auth_status = st.session_state.get("authentication_status")
    name        = st.session_state.get("name", "")
    username    = st.session_state.get("username", "")

    if auth_status is False:
        st.error("❌ Incorrect username or password. Please try again.")
        return

    if auth_status is None:
        return

    # ── Authenticated ──────────────────────────────────────
    # Fix 3: fire a lightweight JS repaint nudge only on the very
    # first render after login to guarantee the page-enter animation
    # triggers cleanly without a stale cached frame.
    if not st.session_state.get("_post_login_rendered"):
        st.session_state["_post_login_rendered"] = True
        st.markdown(
            """
            <script>
              (function() {
                var main = document.querySelector('section[data-testid="stMain"]');
                if (main) {
                  main.style.animation = 'none';
                  requestAnimationFrame(function() {
                    main.style.removeProperty('animation');
                  });
                }
              })();
            </script>
            """,
            unsafe_allow_html=True,
        )

    page = render_sidebar(name, authenticator)

    if page == "Home":
        page_home(name)
    elif page == "Upload & Predict":
        page_upload(username)
    elif page == "Results":
        page_results()


if __name__ == "__main__":
    main()
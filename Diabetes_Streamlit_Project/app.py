
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_diabetes


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Progression Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD FINAL MODEL
# =========================================================

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    """Load the pickled model. Only the Prediction page needs it, so a
    failure here (e.g. a pickle written by a different scikit-learn
    version) must not take down the whole app."""

    try:
        return joblib.load(BASE_DIR / "diabetes_model.pkl"), None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# =========================================================
# LOAD DATASET
# =========================================================

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame


# =========================================================
# SCORES (read from the saved model comparison)
# =========================================================

results_df = pd.read_csv(BASE_DIR / "model_results.csv")


def r2_of(model_name):
    row = results_df[results_df["Model"] == model_name]
    return float(row["R² Score"].iloc[0]) * 100


BASELINE_R2 = r2_of("Linear Regression")
TUNED_R2 = r2_of("Gradient Boosting (Tuned)")


# =========================================================
# IMAGE HELPERS
# =========================================================
# Drop your own artwork into an "assets" folder next to app.py to
# replace the banner / sidebar illustration:
#     assets/hero.jpg        -> top banner photo
#     assets/sidebar.png     -> sidebar illustration
#     assets/objective.png   -> project-objective illustration
# If a file is missing, the drawn SVG fallback below is used, so the app
# never depends on the network to look right.

ASSETS_DIR = BASE_DIR / "assets"


def svg_css(svg):
    """Wrap raw SVG markup as a base64 CSS url() value."""

    data = base64.b64encode(svg.strip().encode()).decode()

    return f'url("data:image/svg+xml;base64,{data}")'


def image_css(stem, fallback_svg=None):
    """Return (css_image, is_photo) for assets/<stem>.<ext>, else the drawn SVG.

    is_photo drives the sizing: a user photo should fill its panel (cover),
    while the drawn artwork should be fitted whole (contain).
    """

    for ext in ("jpg", "jpeg", "png", "webp"):

        path = ASSETS_DIR / f"{stem}.{ext}"

        if path.exists():

            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            data = base64.b64encode(path.read_bytes()).decode()

            return f'url("data:image/{mime};base64,{data}")', True

    if fallback_svg:
        return svg_css(fallback_svg), False

    return "", False


# --- drawn fallback artwork (no network needed) -----------------------

HERO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 380">
  <defs>
    <linearGradient id="hm" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#F472B6"/><stop offset="1" stop-color="#DC2626"/>
    </linearGradient>
    <linearGradient id="hd" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2B3A63"/><stop offset="1" stop-color="#151E38"/>
    </linearGradient>
  </defs>
  <path d="M470 96c0-26 21-45 45-45s45 19 45 47c0 44-56 79-90 105-34-26-90-61-90-105
           0-28 21-47 45-47s45 19 45 45z" fill="url(#hm)" opacity="0.9"/>
  <rect x="92" y="120" width="168" height="228" rx="26" fill="url(#hd)"
        stroke="#4C5C8A" stroke-width="3"/>
  <rect x="116" y="150" width="120" height="78" rx="12" fill="#0B1220" stroke="#33456F"/>
  <text x="176" y="200" font-family="monospace" font-size="42" font-weight="700"
        fill="#67E8F9" text-anchor="middle">105</text>
  <text x="176" y="220" font-family="sans-serif" font-size="15"
        fill="#7DA0C4" text-anchor="middle">mg/dL</text>
  <g fill="#3D4E7A">
    <rect x="116" y="248" width="52" height="26" rx="8"/>
    <rect x="184" y="248" width="52" height="26" rx="8"/>
    <rect x="116" y="286" width="52" height="26" rx="8"/>
    <rect x="184" y="286" width="52" height="26" rx="8"/>
  </g>
  <path d="M282 262h40l16-34 22 66 18-42 14 22h46" fill="none" stroke="#34D399"
        stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M470 236c14 18 24 30 24 42a24 24 0 0 1-48 0c0-12 10-24 24-42z"
        fill="#38BDF8" opacity="0.85"/>
</svg>
"""

SIDEBAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 260">
  <defs>
    <linearGradient id="sm" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#F472B6"/><stop offset="1" stop-color="#E11D48"/>
    </linearGradient>
    <linearGradient id="sd" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#33305F"/><stop offset="1" stop-color="#1B1B3A"/>
    </linearGradient>
  </defs>
  <rect x="34" y="52" width="112" height="156" rx="20" fill="url(#sd)"
        stroke="#5B4E9C" stroke-width="3"/>
  <rect x="52" y="74" width="76" height="52" rx="10" fill="#0D1226" stroke="#4A3F82"/>
  <text x="90" y="106" font-family="monospace" font-size="26" font-weight="700"
        fill="#67E8F9" text-anchor="middle">105</text>
  <text x="90" y="122" font-family="sans-serif" font-size="11"
        fill="#8B93C9" text-anchor="middle">mg/dL</text>
  <g fill="#463C7C">
    <rect x="52" y="142" width="34" height="18" rx="6"/>
    <rect x="96" y="142" width="34" height="18" rx="6"/>
    <rect x="52" y="170" width="34" height="18" rx="6"/>
    <rect x="96" y="170" width="34" height="18" rx="6"/>
  </g>
  <path d="M212 96c0-17 13-29 29-29s29 12 29 30c0 28-36 51-58 68-22-17-58-40-58-68
           0-18 13-30 29-30s29 12 29 29z" fill="url(#sm)"/>
  <path d="M196 128h18l9-19 13 38 10-24 8 13h22" fill="none" stroke="#FFFFFF"
        stroke-width="4" stroke-linecap="round" stroke-linejoin="round"
        opacity="0.9"/>
</svg>
"""

OBJECTIVE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 190">
  <rect x="30" y="26" width="128" height="140" rx="18" fill="#14263C"
        stroke="#2E4E76" stroke-width="3"/>
  <rect x="50" y="46" width="88" height="42" rx="9" fill="#0A1424" stroke="#2B4A72"/>
  <text x="94" y="74" font-family="monospace" font-size="22" font-weight="700"
        fill="#67E8F9" text-anchor="middle">104</text>
  <g fill="#3A5A85">
    <rect x="50" y="102" width="88" height="9" rx="4"/>
    <rect x="50" y="120" width="66" height="9" rx="4"/>
    <rect x="50" y="138" width="44" height="9" rx="4"/>
  </g>
  <rect x="182" y="18" width="118" height="154" rx="14" fill="#E8EEF7"/>
  <rect x="212" y="10" width="58" height="22" rx="7" fill="#9FB4CE"/>
  <g fill="#8FA6C2">
    <rect x="200" y="56" width="82" height="9" rx="4"/>
    <rect x="200" y="78" width="82" height="9" rx="4"/>
    <rect x="200" y="100" width="56" height="9" rx="4"/>
  </g>
  <path d="M318 44l30-12 30 12v34c0 26-18 44-30 50-12-6-30-24-30-50z"
        fill="#38BDF8" transform="translate(-16,14)"/>
  <path d="M316 96l10 11 20-22" fill="none" stroke="#FFFFFF" stroke-width="6"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


HERO_IMAGE, HERO_IS_PHOTO = image_css("hero", HERO_SVG)

HERO_SIZE = "cover" if HERO_IS_PHOTO else "auto 82%"
HERO_POS = "center right" if HERO_IS_PHOTO else "right 44px center"

SIDEBAR_IMAGE, SIDEBAR_IS_PHOTO = image_css("sidebar", SIDEBAR_SVG)
OBJECTIVE_IMAGE, OBJECTIVE_IS_PHOTO = image_css("objective", OBJECTIVE_SVG)

SIDEBAR_SIZE = "cover" if SIDEBAR_IS_PHOTO else "78%"
OBJECTIVE_SIZE = "cover" if OBJECTIVE_IS_PHOTO else "contain"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
<style>

/* ---------- global ---------- */

/* colours are set here as well as in .streamlit/config.toml, so the dark
   design still holds if the app is launched from another directory */
.stApp {{
    background: #0B1020;
    color: #E6EAF3;
}}

section.main > div {{
    padding-top: 1rem;
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1400px;
}}

.card {{
    background: #101827;
    border: 1px solid #1F2A44;
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
}}


/* ---------- hero banner ---------- */

.hero {{
    position: relative;
    border-radius: 18px;
    border: 1px solid #1F2A44;
    overflow: hidden;
    padding: 42px 44px 38px 44px;
    margin-bottom: 18px;
    background-image:
        linear-gradient(
            100deg,
            #05070F 0%,
            rgba(5, 7, 15, 0.94) 34%,
            rgba(5, 7, 15, 0.45) 58%,
            rgba(5, 7, 15, 0.10) 100%
        ),
        {HERO_IMAGE or "linear-gradient(120deg, #2A1B4D 0%, #123049 100%)"},
        linear-gradient(120deg, #2A1B4D 0%, #123049 100%);
    background-size: cover, {HERO_SIZE}, cover;
    background-position: center, {HERO_POS}, center;
    background-repeat: no-repeat;
}}

.hero-title {{
    font-size: 52px;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #3B82F6 0%, #A855F7 55%, #E879F9 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}}

.hero-sub {{
    font-size: 19px;
    color: #C7D0E0;
    margin-bottom: 18px;
}}

.hero-tag {{
    display: inline-block;
    padding: 10px 20px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    color: #FCD34D;
    background: rgba(120, 88, 12, 0.35);
    border: 1px solid rgba(252, 211, 77, 0.35);
}}


/* ---------- section headings ---------- */

.section-title {{
    font-size: 30px;
    font-weight: 800;
    color: #E6EAF3;
    margin-bottom: 6px;
}}

.section-body {{
    font-size: 17px;
    color: #C7D0E0;
}}

.accent-purple {{ color: #A78BFA; font-weight: 700; }}
.accent-teal   {{ color: #34D399; font-weight: 800; }}


/* ---------- stat cards ---------- */

.stat {{
    border-radius: 16px;
    padding: 22px 24px;
    height: 100%;
    display: flex;
    align-items: center;
    gap: 18px;
}}

.stat-icon {{ font-size: 40px; line-height: 1; }}

.stat-label {{
    font-size: 15px;
    color: #C7D0E0;
    margin-bottom: 2px;
}}

.stat-value {{
    font-size: 34px;
    font-weight: 800;
    line-height: 1.1;
}}

.stat-note {{
    font-size: 14px;
    color: #9AA6BC;
    margin-top: 2px;
}}

.stat-purple {{
    background: linear-gradient(135deg, #241B47 0%, #1A1733 100%);
    border: 1px solid #3B2E6B;
}}
.stat-purple .stat-value {{ color: #E9E4FF; }}

.stat-blue {{
    background: linear-gradient(135deg, #10233D 0%, #0E1B2E 100%);
    border: 1px solid #24456F;
}}
.stat-blue .stat-value {{ color: #DCEBFF; }}

.stat-gold {{
    background: linear-gradient(135deg, #2A2210 0%, #1C170B 100%);
    border: 1px solid #6B551C;
}}
.stat-gold .stat-value {{ font-size: 30px; color: #FBBF24; }}


/* ---------- objective card ---------- */

.objective {{
    display: flex;
    align-items: center;
    gap: 22px;
    background: linear-gradient(100deg, #0E1F2A 0%, #101827 60%, #14243A 100%);
    border: 1px solid #1F3A44;
    border-radius: 18px;
    padding: 26px 28px;
    margin-bottom: 18px;
}}

.objective-art {{
    width: 190px;
    min-width: 190px;
    height: 96px;
    border-radius: 14px;
    margin-left: auto;
    background-image:
        {OBJECTIVE_IMAGE or "linear-gradient(135deg, #1B3B57 0%, #0F2035 100%)"},
        linear-gradient(135deg, #1B3B57 0%, #0F2035 100%);
    background-size: {OBJECTIVE_SIZE}, cover;
    background-position: center, center;
    background-repeat: no-repeat;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 40px;
}}


/* ---------- tuning comparison ---------- */

.tuning {{
    display: flex;
    align-items: stretch;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #2A2140;
    margin-bottom: 18px;
}}

.tuning-side {{
    flex: 1;
    padding: 26px 30px;
}}

.tuning-before {{
    background: linear-gradient(100deg, #2B0F14 0%, #1A0C10 100%);
}}

.tuning-after {{
    background: linear-gradient(260deg, #0C2417 0%, #0D1A14 100%);
}}

.tuning-heading {{
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 8px;
}}

.tuning-before .tuning-heading {{ color: #F87171; }}
.tuning-after  .tuning-heading {{ color: #34D399; }}

.tuning-model {{
    font-size: 16px;
    color: #C7D0E0;
    margin-bottom: 6px;
}}

.tuning-score {{
    font-size: 22px;
    font-weight: 800;
}}

.tuning-before .tuning-score {{ color: #F87171; }}
.tuning-after  .tuning-score {{ color: #34D399; }}

.tuning-vs {{
    align-self: center;
    margin: 0 -26px;
    z-index: 2;
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: #12131C;
    border: 2px solid #6B551C;
    color: #E6EAF3;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.tuning-badge {{
    align-self: center;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 30px 0 10px;
    background: linear-gradient(260deg, #0C2417 0%, #0D1A14 100%);
    color: #E6EAF3;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.15;
}}

.tuning-badge .badge-icon {{ font-size: 40px; }}


/* ---------- sidebar ---------- */

section[data-testid="stSidebar"] {{
    background: #0A0F1C;
    border-right: 1px solid #1A2338;
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}

.side-title {{
    font-size: 26px;
    font-weight: 800;
    color: #F3F5FA;
    margin-bottom: 14px;
}}

/* nav radio -> pill menu */

section[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap: 6px;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    padding: 10px 14px;
    border-radius: 12px;
    width: 100%;
    transition: background 0.15s ease;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: #141D31;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(90deg, #4C2C8F 0%, #7C3AED 100%);
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    font-size: 17px;
    font-weight: 600;
    color: #C7D0E0;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
    color: #FFFFFF;
}}

/* hide the drawn radio circle (first child of the label's inner wrapper) */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div > div:first-child {{
    display: none;
}}

/* hide the radio dot but keep the input focusable / screen-reader visible */
section[data-testid="stSidebar"] div[role="radiogroup"] input {{
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
    margin: 0;
    pointer-events: none;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:focus-visible) {{
    outline: 2px solid #A78BFA;
    outline-offset: 2px;
}}

.side-art {{
    height: 210px;
    border-radius: 16px;
    margin: 8px 0 18px 0;
    border: 1px solid #2B2350;
    background-image:
        {SIDEBAR_IMAGE or "linear-gradient(150deg, #221A47 0%, #141B36 100%)"},
        linear-gradient(150deg, #221A47 0%, #141B36 100%);
    background-size: {SIDEBAR_SIZE}, cover;
    background-position: center, center;
    background-repeat: no-repeat;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 62px;
}}

.side-quote {{
    text-align: center;
    font-style: italic;
    font-size: 17px;
    color: #B7C2D6;
    line-height: 1.5;
}}

.side-heart {{
    text-align: center;
    font-size: 22px;
    margin-top: 14px;
}}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    '<div class="side-title">🩸 Navigation</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Explore Project",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🔮 Prediction",
        "📈 Model Performance"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown(
    '<div class="side-art"></div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="side-quote">'
    '"The best way to predict the future is to create it.<br>'
    'Stay healthy, stay ahead!"'
    '</div>'
    '<div class="side-heart">❤️</div>',
    unsafe_allow_html=True
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🩺 Diabetes Progression<br>Prediction</div>'
        '<div class="hero-sub">'
        'Smart ML Application for Predicting Diabetes Disease Progression'
        '</div>'
        '<div class="hero-tag">'
        '✨ From Baseline to Better Health – Predict. Prevent. Progress.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # WELCOME + PROJECT METRICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="card">'
        '<div class="section-title">👥 Welcome 👋 </div>'
        '<div class="section-body">'
        'This application uses a tuned '
        '<span class="accent-purple">Gradient Boosting Regression</span> '
        'model to predict the diabetes disease-progression score.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            '<div class="stat stat-purple">'
            '<div class="stat-icon">👥</div>'
            '<div>'
            '<div class="stat-label">Dataset Records</div>'
            f'<div class="stat-value">{df.shape[0]}</div>'
            '<div class="stat-note">Total Patients</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="stat stat-blue">'
            '<div class="stat-icon">📄</div>'
            '<div>'
            '<div class="stat-label">Input Features</div>'
            '<div class="stat-value">10</div>'
            '<div class="stat-note">Health Parameters</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            '<div class="stat stat-gold">'
            '<div class="stat-icon">🏆</div>'
            '<div>'
            '<div class="stat-label">Best Model (After Tuning)</div>'
            '<div class="stat-value">Gradient Boosting</div>'
            '<div class="stat-note">Highest Performance</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.write("")


    # -----------------------------------------------------
    # PROJECT OBJECTIVE
    # -----------------------------------------------------

    st.markdown(
        '<div class="objective">'
        '<div class="stat-icon">🎯</div>'
        '<div>'
        '<div class="accent-teal" style="font-size:26px;">Project Objective</div>'
        '<div class="section-body">'
        'Predict the quantitative disease-progression score using patient '
        'baseline features and help in early intervention.'
        '</div>'
        '</div>'
        '<div class="objective-art"></div>'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # BEFORE vs AFTER TUNING
    # -----------------------------------------------------

    st.markdown(
        '<div class="tuning">'

        '<div class="tuning-side tuning-before">'
        '<div class="tuning-heading">📊 Before Tuning</div>'
        '<div class="tuning-model">Linear Regression (Baseline Model)</div>'
        f'<div class="tuning-score">R² Score: {BASELINE_R2:.2f}%</div>'
        '</div>'

        '<div class="tuning-vs">VS</div>'

        '<div class="tuning-side tuning-after">'
        '<div class="tuning-heading">After Tuning 🚀</div>'
        '<div class="tuning-model">Gradient Boosting Regressor (Tuned Model)</div>'
        f'<div class="tuning-score">R² Score: {TUNED_R2:.2f}%</div>'
        '</div>'

        '<div class="tuning-badge">'
        '<span class="badge-icon">🏅</span>'
        '<span>Best<br>Model</span>'
        '</div>'

        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("👥 Records", df.shape[0])

    with col2:
        st.metric("🧬 Features", 10)

    with col3:
        st.metric("🎯 Target", "Disease Progression")

    st.divider()

    st.subheader("🔍 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("🧬 Feature Information")

    feature_info = pd.DataFrame({

        "Feature": [
            "age",
            "sex",
            "bmi",
            "bp",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6"
        ],

        "Display Name": [
            "👤 Age",
            "⚧️ Sex",
            "⚖️ Body Mass Index",
            "❤️ Blood Pressure",
            "🧪 Total Cholesterol",
            "🧬 LDL",
            "💙 HDL",
            "📊 Cholesterol / HDL",
            "🧪 Triglycerides",
            "🩸 Blood Sugar"
        ]
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 Diabetes Progression Prediction")

    model, model_error = load_model()

    if model_error:

        st.error(
            "🚫 The saved model could not be loaded, so predictions are "
            "disabled. This usually means `diabetes_model.pkl` was created "
            "with a different scikit-learn version than the one installed "
            f"here. Re-run the notebook to regenerate it.\n\n`{model_error}`"
        )

        st.stop()

    st.write(
        "Enter the feature values used by the trained "
        "machine learning model."
    )

    st.warning(
        "⚠️ The Scikit-learn Diabetes Dataset uses "
        "standardized feature values. Enter values using "
        "the same scaled format as the dataset."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👤 Patient Information")

        age = st.number_input(
            "👤 Age",
            value=0.0,
            format="%.5f"
        )

        sex = st.number_input(
            "⚧️ Sex",
            value=0.0,
            format="%.5f"
        )

        bmi = st.number_input(
            "⚖️ Body Mass Index",
            value=0.0,
            format="%.5f"
        )

        bp = st.number_input(
            "❤️ Blood Pressure",
            value=0.0,
            format="%.5f"
        )

        s1 = st.number_input(
            "🧪 Total Cholesterol (s1)",
            value=0.0,
            format="%.5f"
        )

    with col2:

        st.subheader("🧪 Blood Measurements")

        s2 = st.number_input(
            "🧬 LDL (s2)",
            value=0.0,
            format="%.5f"
        )

        s3 = st.number_input(
            "💙 HDL (s3)",
            value=0.0,
            format="%.5f"
        )

        s4 = st.number_input(
            "📊 Cholesterol / HDL (s4)",
            value=0.0,
            format="%.5f"
        )

        s5 = st.number_input(
            "🧪 Triglycerides (s5)",
            value=0.0,
            format="%.5f"
        )

        s6 = st.number_input(
            "🩸 Blood Sugar (s6)",
            value=0.0,
            format="%.5f"
        )


    st.divider()


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "🚀 Generate Prediction",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                age,
                sex,
                bmi,
                bp,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6
            ]],
            columns=[
                "age",
                "sex",
                "bmi",
                "bp",
                "s1",
                "s2",
                "s3",
                "s4",
                "s5",
                "s6"
            ]
        )

        prediction = model.predict(input_data)[0]

        st.success(
            "✅ Prediction Generated Successfully!"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🔮 Predicted Disease Progression",
                f"{prediction:.2f}"
            )

        with col2:

            st.metric(
                "🤖 Model Used",
                "Gradient Boosting"
            )

        st.info(
            "ℹ️ This application is intended for educational "
            "and machine-learning demonstration purposes "
            "and is not a medical diagnostic tool."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Comparison of the machine learning models evaluated "
        "during the project."
    )

    st.success(
        "🏆 Final model selected from the highest R² score."
    )

    display_results = results_df.copy()

    display_results["R² Percentage"] = (
        display_results["R² Score"] * 100
    ).round(2).astype(str) + "%"

    st.subheader("📊 Model Comparison")

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 R² Score Comparison")

    chart_data = (
        results_df
        .set_index("Model")["R² Score"]
        * 100
    )

    st.bar_chart(chart_data)

    st.subheader("💡 Model Evaluation")

    st.write(
        """
        **R² Score:** Higher values indicate better
        explanatory performance.

        **MAE:** Lower values indicate smaller average
        prediction errors.

        **RMSE:** Lower values indicate better control
        over larger prediction errors.
        """
    )

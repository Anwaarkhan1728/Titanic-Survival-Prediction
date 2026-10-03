# =========================================================
# 🚢 TITANIC SURVIVAL PREDICTION — STREAMLIT APP (FIXED)
# =========================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
    }
    .big-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #444;
        margin-top: -10px;
        margin-bottom: 25px;
    }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: 15px;
    }
    .survived   { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .not-survived { background: linear-gradient(135deg, #eb3349, #f45c43); }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #1e3c72;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72, #2a5298);
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 30px;
        border: none;
        width: 100%;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2a5298, #1e3c72);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if "Name" in data.columns:
        data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    else:
        data["Title"] = "Mr"

    rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
            "Rev", "Sir", "Jonkheer", "Dona"]
    data["Title"] = data["Title"].replace(rare, "Rare")
    data["Title"] = data["Title"].replace(["Mlle", "Ms"], "Miss")
    data["Title"] = data["Title"].replace("Mme", "Mrs")
    data["Title"] = data["Title"].fillna("Mr")

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"]    = (data["FamilySize"] == 1).astype(int)

    if "Cabin" in data.columns:
        data["HasCabin"] = data["Cabin"].notna().astype(int)
    else:
        data["HasCabin"] = 0

    data["FarePerPerson"] = data["Fare"] / data["FamilySize"]

    if data["Age"].isna().any():
        medians = {"Mr": 30, "Miss": 22, "Mrs": 35,
                   "Master": 5, "Rare": 45}
        data["Age"] = data.apply(
            lambda r: medians.get(r["Title"], 28) if pd.isna(r["Age"]) else r["Age"],
            axis=1
        )

    data["Fare"]     = data["Fare"].fillna(32.0)
    data["Embarked"] = data["Embarked"].fillna("S")

    return data

# =========================================================
# 4. LOAD MODEL — ROBUST VERSION
# =========================================================
MODEL_PATH = "titanic_best_model.pkl"
EXPECTED_SKLEARN = "1.6.1"

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None, f"❌ Model file `{path}` not found in repo."

    current = sklearn.__version__
    if current != EXPECTED_SKLEARN:
        return None, (
            f"⚠️ **scikit-learn version mismatch!**\n\n"
            f"- Model trained with: `scikit-learn=={EXPECTED_SKLEARN}`\n"
            f"- Current environment: `scikit-learn=={current}`\n\n"
            f"**Fix:** Update `requirements.txt` to add `scikit-learn=={EXPECTED_SKLEARN}` and redeploy."
        )

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except AttributeError as e:
        return None, (
            f"❌ **Pickle load failed (AttributeError):** {e}\n\n"
            f"Version mismatch. Expected `{EXPECTED_SKLEARN}`, got `{current}`."
        )
    except Exception as e:
        return None, f"❌ **Failed to load model:** {type(e).__name__}: {e}"


model, load_error = load_model(MODEL_PATH)

# =========================================================
# 5. HEADER
# =========================================================
st.markdown('<h1 class="big-title">🚢 Titanic Survival Predictor</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter passenger details below and the ML model '
            'will predict whether they would have survived the Titanic disaster.</p>',
            unsafe_allow_html=True)

if model is None:
    st.error(load_error)
    st.info(
        "📌 **Tip:** Ensure your `requirements.txt` contains:\n"
        "```\nscikit-learn==1.6.1\n```\n"
        "Then push to GitHub — Streamlit Cloud will auto-rebuild."
    )
    st.stop()

st.success(f"✅ Model loaded successfully! (scikit-learn {sklearn.__version__})")

# =========================================================
# 6. SIDEBAR INPUTS
# =========================================================
st.sidebar.header("⚙️ Passenger Details")
st.sidebar.markdown("Fill in the information below:")

with st.sidebar:
    pclass   = st.selectbox("🎫 Passenger Class", [1, 2, 3], index=2,
                            help="1 = First, 2 = Second, 3 = Third")
    sex      = st.radio("⚧ Sex", ["male", "female"], horizontal=True)
    title    = st.selectbox("👤 Title",
                            ["Mr", "Mrs", "Miss", "Master",
                             "Dr", "Rev", "Col", "Capt",
                             "Major", "Sir", "Lady"],
                            index=0)
    age      = st.slider("🎂 Age", 0, 90, 28)

    st.markdown("**👨‍👩‍👧 Family Aboard**")
    sibsp    = st.number_input("Siblings / Spouses", 0, 8, 0)
    parch    = st.number_input("Parents / Children", 0, 6, 0)

    fare     = st.number_input("💷 Fare (£)", 0.0, 600.0, 32.0, step=0.5)
    embarked = st.selectbox("⚓ Port of Embarkation",
                            ["S", "C", "Q"],
                            help="S=Southampton, C=Cherbourg, Q=Queenstown")
    has_cabin = st.checkbox("🛏️ Cabin recorded?", value=False)

    predict_btn = st.button("🔮 Predict Survival")

# =========================================================
# 7. MAIN AREA
# =========================================================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📋 Passenger Summary")
    summary = pd.DataFrame({
        "Feature": ["Passenger Class", "Sex", "Title", "Age",
                    "Siblings/Spouses", "Parents/Children",
                    "Fare (£)", "Embarked", "Cabin Recorded"],
        "Value":   [pclass, sex, title, age, sibsp, parch,
                    fare, embarked, "Yes" if has_cabin else "No"]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# =========================================================
# 8. PREDICTION
# =========================================================
if predict_btn:
    row = pd.DataFrame([{
        "Name":     f"Doe, {title}. John",
        "Pclass":   pclass,
        "Sex":      sex,
        "Age":      float(age),
        "SibSp":    sibsp,
        "Parch":    parch,
        "Fare":     float(fare),
        "Embarked": embarked,
        "Cabin":    "C85" if has_cabin else None,
    }])

    row_fe = engineer_features(row)

    try:
        pred  = int(model.predict(row_fe)[0])
        proba = model.predict_proba(row_fe)[0]
        prob_survived = float(proba[1])
        prob_died     = float(proba[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    with col_right:
        st.markdown("### 🎯 Prediction Result")

        if pred == 1:
            st.markdown(
                f'<div class="result-card survived">'
                f'✅ SURVIVED<br>'
                f'<span style="font-size:1.1rem">Confidence: {prob_survived*100:.2f}%</span>'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-card not-survived">'
                f'❌ DID NOT SURVIVE<br>'
                f'<span style="font-size:1.1rem">Confidence: {prob_died*100:.2f}%</span>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("#### 📊 Probabilities")
        st.metric("Survival Probability", f"{prob_survived*100:.2f}%")
        st.progress(prob_survived)

        st.metric("Death Probability", f"{prob_died*100:.2f}%")
        st.progress(prob_died)

    with st.expander("🔍 View Engineered Features Used by Model"):
        st.dataframe(row_fe.T.rename(columns={0: "Value"}),
                     use_container_width=True)

    st.markdown("### 📈 Survival Probability Gauge")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([""], [prob_survived], color="#38ef7d", edgecolor="black")
    ax.barh([""], [1 - prob_survived], left=[prob_survived],
            color="#eb3349", edgecolor="black")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(f"Survival: {prob_survived*100:.2f}%   |   "
                 f"Death: {prob_died*100:.2f}%")
    ax.text(prob_survived / 2, 0, f"{prob_survived*100:.1f}%",
            ha="center", va="center", color="white", fontweight="bold")
    ax.text(prob_survived + (1 - prob_survived) / 2, 0,
            f"{prob_died*100:.1f}%",
            ha="center", va="center", color="white", fontweight="bold")
    st.pyplot(fig)

else:
    with col_right:
        st.markdown("### 🎯 Prediction Result")
        st.info("👈 Fill in the passenger details in the sidebar and "
                "click **🔮 Predict Survival** to see the result.")

# =========================================================
# 9. FOOTER
# =========================================================
st.divider()
st.markdown(
    "<div style='text-align:center; color:#666; padding:10px;'>"
    "🚢 <b>Titanic Survival Predictor</b> • Built with Streamlit + scikit-learn • "
    "Model: Logistic Regression (best of 3 classifiers)"
    "</div>",
    unsafe_allow_html=True
)

# ======================================================================================
# 1. IMPORTS
# ======================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
from PIL import Image
from streamlit_lottie import st_lottie
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ======================================================================================
# 2. PAGE CONFIGURATION
# ======================================================================================
st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================================================
# 3. CUSTOM CSS FOR GLASSMORPHISM DESIGN
# ======================================================================================
st.markdown("""
<style>
    .stApp {
        background: url("https://www.transparenttextures.com/patterns/dark-matter.png");
        background-color: #0c111c;
    }
    .main .block-container { padding: 2rem 3rem; }
    .header-text { font-family: 'Garamond', serif; color: #e0e7ff; font-weight: 700; font-size: 3rem; text-align: center; }
    .subheader-text { color: #a5b4fc; font-size: 1.1rem; text-align: center; margin-bottom: 2.5rem; }
    .form-container { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 2rem; border-radius: 1rem; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton>button { background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.8rem 1.6rem; border-radius: 0.75rem; font-weight: bold; font-size: 1.1rem; border: none; }
    .stMetric { background-color: transparent; border-radius: 1rem; padding: 1rem; text-align: center; }
    .stMetric > label { font-weight: 600; color: #c7d2fe; }
    .stMetric > div > span { font-size: 2.5rem; color: #818cf8; font-weight: 700; }
    .stExpander { background: rgba(255, 255, 255, 0.05); border-radius: 0.5rem; border: 1px solid rgba(255, 255, 255, 0.1); }
    .footer { text-align: center; padding: 2rem; color: #9ca3af; }
    .footer a { color: #818cf8; text-decoration: none; font-weight: 600; margin: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ======================================================================================
# 4. LOAD ASSETS
# ======================================================================================
@st.cache_data

def load_assets():
    model_data = joblib.load("income_classifier_py313.pkl")
    with open("animation.json", "r") as f:
        lottie_json = json.load(f)
    return model_data, lottie_json

model_data, lottie_json = load_assets()
models = model_data["models"]
preprocessor = model_data["preprocessor"]
X_sample = model_data["X_sample"]

# ======================================================================================
# 5. SESSION STATE INIT
# ======================================================================================
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction_result = None
    st.session_state.selected_model = "Gradient Boosting"

# ======================================================================================
# 6. HEADER & FORM
# ======================================================================================
st.markdown('<p class="header-text">💰 Income Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Predict whether income exceeds $50K using machine learning.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    with st.form("income_form"):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #c7d2fe;'>Applicant Profile</h3>", unsafe_allow_html=True)

        age = st.slider("Age", 18, 80, 35)
        workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                               'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        education_num = st.slider("Education Number", 1, 16, 10)
        marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                         'Separated', 'Widowed', 'Married-spouse-absent'])
        occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                                 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                                 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                                 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
        relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
        race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        sex = st.selectbox("Sex", ['Female', 'Male'])
        capital_gain = st.number_input("Capital Gain", value=0)
        capital_loss = st.number_input("Capital Loss", value=0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                                                         'India', 'Puerto-Rico', 'Honduras', 'Jamaica', 'Italy'])
        model_choice = st.selectbox("Model to Use", list(models.keys()), index=list(models.keys()).index("Gradient Boosting"))

        submit = st.form_submit_button("🔍 Predict Income")
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================================
# 7. PREDICTION
# ======================================================================================
if submit:
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'educational-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])
    processed = preprocessor.transform(input_data)
    model = models[model_choice]
    proba = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    st.session_state.prediction_result = (prediction, proba)
    st.session_state.prediction_made = True
    st.session_state.selected_model = model_choice
    st.rerun()

# ======================================================================================
# 8. OUTPUT UI
# ======================================================================================
with col2:
    if not st.session_state.prediction_made:
        st_lottie(lottie_json, speed=1, height=300)
        st.info("Prediction will appear here once submitted.", icon="🔍")
    else:
        pred, proba = st.session_state.prediction_result
        label = ">50K" if pred == 1 else "<=50K"
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #c7d2fe;'>Prediction Result</h3>", unsafe_allow_html=True)
        st.metric("Predicted Income Class", label, delta=f"Probability: {proba:.2%}")
        st.success(f"Prediction complete using {st.session_state.selected_model}.", icon="✅")
        st.markdown('</div>', unsafe_allow_html=True)

        # SHAP EXPLAIN
        if st.session_state.selected_model == "Gradient Boosting":
            st.markdown("---")
            with st.expander("💡 SHAP Explanation"):
                explainer = shap.Explainer(models['Gradient Boosting'])
                shap_values = explainer(preprocessor.transform(X_sample.iloc[:100]))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(bbox_inches='tight')

# ======================================================================================
# 9. FOOTER
# ======================================================================================
st.markdown("""
---
<div class="footer">
    <p>Crafted with 🤖 by <b>Ayush Anand</b></p>
    <a href="https://github.com/Ayush03A" target="_blank">GitHub</a> |
    <a href="https://www.linkedin.com/in/3ayushanand/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

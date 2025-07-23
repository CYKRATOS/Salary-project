import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import json

# --------------------
# Page config
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# --------------------
# Load assets
@st.cache_resource
def load_assets():
    with open("logistic_only_model.pkl", "rb") as f:
        model_data = joblib.load(f)
    with open("animation.json", "r") as f:
        lottie_json = json.load(f)
    return model_data, lottie_json

model_data, lottie_json = load_assets()
model = model_data["model"]

# --------------------
# Apply Glassmorphism UI
st.markdown("""
    <style>
        .glass {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 1rem;
            padding: 2rem;
            backdrop-filter: blur(15px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --------------------
# Main UI
st.title("💼 Income Classification App")
st_lottie(lottie_json, height=200)

st.markdown("#### Predict if a person's income exceeds 50K/year based on their info.")
with st.container():
    with st.form("income_form"):
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        age = st.slider("Age", 18, 90, 30)
        workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                                               'Federal-gov', 'Local-gov', 'State-gov',
                                               'Without-pay', 'Never-worked'])
        education_num = st.slider("Education Number", 1, 16, 10)
        marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced',
                                                         'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'])
        occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service',
                                                 'Sales', 'Exec-managerial', 'Prof-specialty',
                                                 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                                 'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
                                                 'Protective-serv', 'Armed-Forces'])
        relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband',
                                                     'Not-in-family', 'Other-relative', 'Unmarried'])
        race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
                                     'Other', 'Black'])
        sex = st.selectbox("Sex", ['Male', 'Female'])
        capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
        capital_loss = st.number_input("Capital Loss", 0, 99999, 0)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines',
                                                         'Germany', 'Canada', 'Puerto-Rico', 'India',
                                                         'Cuba', 'England', 'Jamaica'])

        submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame([{
                'age': age,
                'workclass': workclass,
                'education-num': education_num,
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

            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            income = ">50K" if pred == 1 else "<=50K"
            st.success(f"📊 **Prediction:** Person is likely to earn **{income}**.")
            st.metric("Probability of >50K income", f"{proba*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("___")
st.caption("Built with ❤️ using Streamlit and Logistic Regression")

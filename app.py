import streamlit as st
import joblib
import pandas as pd
import time

# Load model and features
model = joblib.load('titanic_model.pkl')
features = joblib.load('feature_names.pkl')

# Page Configuration
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Sidebar
st.sidebar.title("📋 About")
st.sidebar.info(
    """
    This app predicts whether a passenger would survive the Titanic disaster based on their information.
    
    - Powered by **Machine Learning**
    """
)
st.sidebar.markdown("💡 Tip: Adjust the inputs and click **Predict** to see the result!")

# Title
st.title("🚢 Titanic Survival Prediction App")

# Create layout with columns for better alignment
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎫 Passenger Class", [1, 2, 3])
    sex = st.selectbox("👤 Sex", ["male", "female"])
    embarked = st.selectbox("🛳️ Port of Embarkation", ["S", "C", "Q"])
    fare = st.number_input("💰 Fare Paid", 0.0, 600.0, 50.0)

with col2:
    age = st.slider("🎂 Age", 0, 80, 25)
    sibsp = st.number_input("🧍‍🤝‍🧍 Siblings/Spouses ", 0, 10, 0)
    parch = st.number_input("👨‍👩‍👧 Parents/Children ", 0, 10, 0)

# Derived features
pclass_str = {1: "First", 2: "Second", 3: "Third"}[pclass]
who = "man" if sex == "male" else "woman"
adult_male = 1 if sex == "male" and age >= 18 else 0
alone = 1 if sibsp == 0 and parch == 0 else 0

# Create input dict
input_dict = {
    'pclass': pclass,
    'age': age,
    'sibsp': sibsp,
    'parch': parch,
    'fare': fare,
    'adult_male': adult_male,
    'alone': alone,
    'sex_male': 1 if sex == "male" else 0,
    'embarked_Q': 1 if embarked == "Q" else 0,
    'embarked_S': 1 if embarked == "S" else 0,
    'class_Second': 1 if pclass == 2 else 0,
    'class_Third': 1 if pclass == 3 else 0,
    'who_man': 1 if who == "man" else 0,
    'who_woman': 1 if who == "woman" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])[features]

# Predict Button
if st.button("🔍 Predict Survival"):
    with st.spinner("Analyzing passenger data..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.079)
            progress.progress(i + 1)
        time.sleep(1)

    prediction = model.predict(input_df)[0]
    result_text = "✅ **Survived**" if prediction == 1 else "❌ **Did not survive**"
    st.markdown(f"### 🎯 Prediction Result: {result_text}")

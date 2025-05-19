import streamlit as st
import joblib
import pandas as pd

# Load model and features
model = joblib.load('titanic_model.pkl')
features = joblib.load('feature_names.pkl')

st.title("🚢 Titanic Survival Prediction")

# User input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 50.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
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

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("✅ Survived" if prediction == 1 else "❌ Did not survive")

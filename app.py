import streamlit as st
import joblib
import pandas as pd
import time

# Set page config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Load model and features
model = joblib.load('titanic_model.pkl')
features = joblib.load('feature_names.pkl')

# --- Inject CSS for floating & sinking boat ---
boat_base_css = """
<style>
.boat-container {
    width: 100%%;
    height: 160px;
    overflow: hidden;
    position: relative;
    margin-bottom: 10px;
}

.boat {
    position: absolute;
    width: 120px;
    height: 60px;
    background-image: url('https://i.imgur.com/WP4E2tD.png');
    background-size: contain;
    background-repeat: no-repeat;
    animation: float 4s ease-in-out infinite;
    left: 50%%;
    transform: translateX(-50%%);
}

@keyframes float {
    0%% { top: 40px; }
    50%% { top: 20px; }
    100%% { top: 40px; }
}

.boat.sink {
    animation: sink 3s forwards;
}

@keyframes sink {
    0%% { top: 40px; opacity: 1; }
    50%% { top: 80px; opacity: 0.7; }
    100%% { top: 200px; opacity: 0; }
}
</style>
"""
st.markdown(boat_base_css, unsafe_allow_html=True)

# Title and Sidebar
st.title("🚢 Titanic Survival Prediction App")

st.sidebar.title("📋 About")
st.sidebar.info(
    """
    Predict whether a Titanic passenger would survive based on their data.
    
    Built with **Streamlit** + **ML Model**
    """
)

# Maintain prediction result to control animation
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Determine boat class (sinks only if predicted = 0)
boat_class = ""
if st.session_state.prediction_result is not None:
    boat_class = "sink" if st.session_state.prediction_result == 0 else ""

# Boat Animation
st.markdown(f"""
<div class="boat-container">
    <div class="boat {boat_class}"></div>
</div>
""", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("🎫 Passenger Class", [1, 2, 3])
    sex = st.selectbox("👤 Sex", ["male", "female"])
    embarked = st.selectbox("🛳️ Port of Embarkation", ["S", "C", "Q"])
    fare = st.number_input("💰 Fare Paid", 0.0, 600.0, 50.0)
with col2:
    age = st.slider("🎂 Age", 0, 80, 25)
    sibsp = st.number_input("🧍‍🤝‍🧍 Siblings/Spouses aboard", 0, 10, 0)
    parch = st.number_input("👨‍👩‍👧 Parents/Children aboard", 0, 10, 0)

# Derived Features
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

# Create DataFrame
input_df = pd.DataFrame([input_dict])[features]

# Prediction button
if st.button("🔍 Predict Survival"):
    with st.spinner("Analyzing passenger data..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    # Run model prediction
    prediction = model.predict(input_df)[0]
    st.session_state.prediction_result = prediction  # Save to session state

    result_text = "✅ **Survived**" if prediction == 1 else "❌ **Did not survive**"
    st.markdown(f"### 🎯 Prediction Result: {result_text}")

    # Summary of input
    st.markdown("---")
    st.subheader("🔎 Passenger Summary")
    st.markdown(f"""
    - **Class**: {pclass_str}
    - **Sex**: {sex}
    - **Age**: {age}
    - **Fare**: ${fare}
    - **Siblings/Spouses**: {sibsp}
    - **Parents/Children**: {parch}
    - **Embarked**: {embarked}
    - **Alone**: {"Yes" if alone else "No"}
    """)

    # Celebration
    if prediction == 1:
        st.balloons()

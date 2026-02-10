import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import base64

st.set_page_config(page_title="Stroke Risk Prediction", layout="centered")

# Load and encode background image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get the background image
bg_image = get_base64_image(Path(__file__).parent / "stroke_web.jpg")

# Add background with transparency
st.markdown(f"""
    <style>
    .stApp {{
        background-color: black;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 80%;
        max-width: 1200px;
        height: 80%;
        background-image: url("data:image/jpeg;base64,{bg_image}");
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.3;
        z-index: 0;
        pointer-events: none;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("Stroke Risk Prediction App")
st.write("Enter patient details to predict stroke risk.")

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_pickle(filename: str):
    candidates = [
        ROOT_DIR / "models" / filename,
        ROOT_DIR / filename,
        ROOT_DIR / "notebook" / filename,
        Path.cwd() / filename,
        Path.cwd() / "notebook" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("rb") as f:
                return pickle.load(f)

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find '{filename}'. Looked in:\n{searched}"
    )


try:
    model = _load_pickle("stroke_prediction_model.pkl")
    scaler = _load_pickle("scaler.pkl")
    feature_names = _load_pickle("feature_names.pkl")
except FileNotFoundError as e:
    st.error(
        "Missing model files.\n\n"
        "Expected these files in the project root or in the 'notebook' folder:\n"
        "- stroke_prediction_model.pkl\n"
        "- scaler.pkl\n"
        "- feature_names.pkl\n\n"
        f"Details: {e}"
    )
    st.stop()

st.header("Patient Information")

age = st.slider("Age", 0, 100, 50)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
avg_glucose_level = st.number_input(
    "Average Glucose Level (mg/dL)",
    min_value=50.0,
    max_value=300.0,
    value=100.0,
    help="Enter recent blood test value. Normal fasting range: 70–99 mg/dL"
)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Government Job", "Child (Not Employed)", "Retired"]
)
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# Map work_type to internal model values
if work_type == "Government Job":
    work_type = "Govt_job"
elif work_type == "Child (Not Employed)":
    work_type = "children"
elif work_type == "Retired":
    work_type = "Never_worked"

# Map Yes/No to 1/0 for model
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

if st.button("Predict Stroke Risk"):

    # Create base dataframe
    input_data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi
    }

    df_input = pd.DataFrame([input_data])

    # Feature Engineering

    # Age group
    if age <= 40:
        df_input['age_group'] = 0
    elif age <= 60:
        df_input['age_group'] = 1
    elif age <= 80:
        df_input['age_group'] = 2
    else:
        df_input['age_group'] = 3

    # BMI category
    if bmi < 18.5:
        df_input['bmi_category'] = 0
    elif bmi < 25:
        df_input['bmi_category'] = 1
    elif bmi < 30:
        df_input['bmi_category'] = 2
    else:
        df_input['bmi_category'] = 3

    # Risk score
    df_input['age_over_60'] = 1 if age > 60 else 0
    df_input['bmi_over_30'] = 1 if bmi > 30 else 0
    df_input['risk_score'] = (
        hypertension + heart_disease +
        df_input['age_over_60'] +
        df_input['bmi_over_30']
    )

    # Glucose category
    if avg_glucose_level < 140:
        df_input['glucose_category'] = 0
    elif avg_glucose_level < 200:
        df_input['glucose_category'] = 1
    else:
        df_input['glucose_category'] = 2

    # Add missing columns
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder columns
    df_input = df_input[feature_names]

    # Scale
    df_input_scaled = scaler.transform(df_input)

    # Predict
    prediction = model.predict(df_input_scaled)[0]
    probability = model.predict_proba(df_input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Stroke Risk ⚠️ (Probability: {probability:.2%})")
    else:
        st.success(f"Low Stroke Risk ✅ (Probability: {probability:.2%})")


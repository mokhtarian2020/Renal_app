import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('log_reg_model.pkl')
scaler = joblib.load('scale.pkl')

# Define input ranges for each category
mild_bun_range = (10, 50)
mild_creatinine_range = (2, 10)
mild_potassium_range = (3.5, 5.0)
mild_bp_systolic_range = (100, 140)
mild_bp_diastolic_range = (60, 90)

moderate_bun_range = (50, 70)
moderate_creatinine_range = (10, 15)
moderate_potassium_range = (5.1, 5.5)
moderate_bp_systolic_range = (140, 160)
moderate_bp_diastolic_range = (90, 100)

severe_bun_range = (70, 120)
severe_creatinine_range = (15, 25)
severe_potassium_range = (5.5, 7.0)
severe_bp_systolic_range = (160, 200)
severe_bp_diastolic_range = (100, 120)

st.title("Renal Dialysis Patient Classification")

# Add sliding inputs for user to adjust values
bun = st.slider("BUN (mg/dL)", 10, 120, 50)
creatinine = st.slider("Creatinine (mg/dL)", 2, 25, 10)
potassium = st.slider("Potassium (mEq/L)", 3.5, 7.0, 5.0)
bp_systolic = st.slider("Systolic Blood Pressure (mmHg)", 100, 200, 140)
bp_diastolic = st.slider("Diastolic Blood Pressure (mmHg)", 60, 120, 90)

# Prepare the input data
input_data = np.array([[bun, creatinine, potassium, bp_systolic, bp_diastolic]])

# Scale the input data
scaled_data = scaler.transform(input_data)

# Predict the stage using the model
prediction = model.predict(scaled_data)

# Mapping numeric output to stage names
stage_mapping = {0: 'mild', 1: 'moderate', 2: 'severe'}
predicted_stage = stage_mapping[int(prediction[0])]

# Display the prediction result
st.write(f"The predicted stage is: **{predicted_stage}**")

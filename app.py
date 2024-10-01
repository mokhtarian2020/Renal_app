import streamlit as st
import numpy as np
import joblib

# Load the logistic regression model and scaler from the files you uploaded
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict COPD stage using the model
def predict_copd_stage(features):
    # Scale the input features
    features_scaled = scaler.transform([features])
    
    # Make a prediction using the model
    prediction = model.predict(features_scaled)
    
    # Map the prediction to the COPD stages
    stage_map = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
    return stage_map[prediction[0]]

# Streamlit app title
st.title('COPD Patient Classification App')

# App subtitle
st.write('Adjust the patient data below to predict the COPD stage')

# COPD Ranges
ranges = {
    'FEV1': {'mild': [80, 100], 'moderate': [50, 79], 'severe': [0, 49]},
    'FVC': {'mild': [80, 120], 'moderate': [60, 79], 'severe': [30, 59]},
    'pO2': {'mild': [80, 100], 'moderate': [60, 79], 'severe': [40, 59]},
    'SpO2': {'mild': [94, 100], 'moderate': [88, 93], 'severe': [80, 87]},
    'pCO2': {'mild': [35, 45], 'moderate': [46, 55], 'severe': [56, 70]},
    'Lactate': [0.5, 2.2], 'Hemoglobin': [13, 17], 'HeartRate': [60, 100], 
    'BloodPressure': [(110, 70), (140, 90)]
}

# Sliders for patient input data based on the ranges
fev = st.slider('FEV1 (Forced Expiratory Volume)', min_value=0.0, max_value=100.0, value=50.0, step=0.1, help="Mild: 80-100, Moderate: 50-79, Severe: 0-49")
fv = st.slider('FVC (Forced Vital Capacity)', min_value=0.0, max_value=120.0, value=60.0, step=0.1, help="Mild: 80-120, Moderate: 60-79, Severe: 30-59")
po2 = st.slider('pO2 (Partial Pressure of Oxygen)', min_value=0.0, max_value=100.0, value=70.0, step=0.1, help="Mild: 80-100, Moderate: 60-79, Severe: 40-59")
spo2 = st.slider('SpO2 (Oxygen Saturation)', min_value=80.0, max_value=100.0, value=94.0, step=0.1, help="Mild: 94-100, Moderate: 88-93, Severe: 80-87")
pco2 = st.slider('pCO2 (Partial Pressure of Carbon Dioxide)', min_value=30.0, max_value=70.0, value=40.0, step=0.1, help="Mild: 35-45, Moderate: 46-55, Severe: 56-70")
lactate = st.slider('Lactate Level', min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Normal: 0.5-2.2")
hemoglobin = st.slider('Hemoglobin Level', min_value=0.0, max_value=20.0, value=13.0, step=0.1, help="Normal: 13-17")
heart_rate = st.slider('Heart Rate', min_value=40.0, max_value=200.0, value=80.0, step=1.0, help="Normal: 60-100")
bp_systolic = st.slider('Systolic Blood Pressure', min_value=70.0, max_value=200.0, value=120.0, step=1.0, help="Normal: 110-140")
bp_diastolic = st.slider('Diastolic Blood Pressure', min_value=40.0, max_value=120.0, value=80.0, step=1.0, help="Normal: 70-90")

# When the 'Predict' button is clicked, make a prediction
if st.button('Predict'):
    # Store the input data in a list
    features = [fev, fv, po2, spo2, pco2, lactate, hemoglobin, heart_rate, bp_systolic, bp_diastolic]
    
    # Check if all values are entered
    if any(f == 0 for f in features):
        st.write("Please fill in all the fields correctly.")
    else:
        # Get the prediction and display it
        result = predict_copd_stage(features)
        st.success(f'The predicted COPD stage is: {result}')

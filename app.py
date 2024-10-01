import streamlit as st
import numpy as np
import joblib

# Load the logistic regression model and scaler from the files you uploaded
model = joblib.load('log_reg_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict renal stage using the model
def predict_renal_stage(features):
    # Scale the input features
    features_scaled = scaler.transform([features])
    
    # Make a prediction using the model
    prediction = model.predict(features_scaled)
    
    # Map the prediction to the renal stages
    stage_map = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
    return stage_map[prediction[0]]

# Streamlit app title
st.title('Renal Dialysis Patient Classification App')

# App subtitle
st.write('Adjust the patient data below to predict the renal dialysis stage')

# Renal Ranges based on the data
ranges = {
    'BUN': {'mild': (10, 50), 'moderate': (50, 70), 'severe': (70, 120)},
    'Creatinine': {'mild': (2, 10), 'moderate': (10, 15), 'severe': (15, 25)},
    'Potassium': {'mild': (3.5, 5.0), 'moderate': (5.1, 5.5), 'severe': (5.5, 7.0)},
    'Systolic BP': {'mild': (100, 140), 'moderate': (140, 160), 'severe': (160, 200)},
    'Diastolic BP': {'mild': (60, 90), 'moderate': (90, 100), 'severe': (100, 120)},
}

# Sliders for patient input data based on the ranges
bun = st.slider('BUN (mg/dL)', min_value=10.0, max_value=120.0, value=50.0, step=0.1, help="Mild: 10-50, Moderate: 50-70, Severe: 70-120")
creatinine = st.slider('Creatinine (mg/dL)', min_value=2.0, max_value=25.0, value=10.0, step=0.1, help="Mild: 2-10, Moderate: 10-15, Severe: 15-25")
potassium = st.slider('Potassium (mEq/L)', min_value=3.5, max_value=7.0, value=5.0, step=0.1, help="Mild: 3.5-5.0, Moderate: 5.1-5.5, Severe: 5.5-7.0")
bp_systolic = st.slider('Systolic Blood Pressure (mmHg)', min_value=100.0, max_value=200.0, value=140.0, step=1.0, help="Mild: 100-140, Moderate: 140-160, Severe: 160-200")
bp_diastolic = st.slider('Diastolic Blood Pressure (mmHg)', min_value=60.0, max_value=120.0, value=90.0, step=1.0, help="Mild: 60-90, Moderate: 90-100, Severe: 100-120")

# When the 'Predict' button is clicked, make a prediction
if st.button('Predict'):
    # Store the input data in a list
    features = [bun, creatinine, potassium, bp_systolic, bp_diastolic]
    
    # Check if all values are entered
    if any(f == 0 for f in features):
        st.write("Please fill in all the fields correctly.")
    else:
        # Get the prediction and display it
        result = predict_renal_stage(features)
        st.success(f'The predicted renal dialysis stage is: {result}')

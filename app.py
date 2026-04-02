import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download, login
from google.colab import userdata
import os

# 1. Hugging Face Login and Model Loading
# Retrieve HF_TOKEN from Colab userdata secrets
# Get HF_TOKEN from Colab userdata secrets


hf_username = "rakesh1248" # IMPORTANT: Replace with your HF username
model_name = "random_forest_engine_condition_classifier"
repo_id = f"{hf_username}/{model_name}"
model_path_in_repo = "random_forest_model.joblib"
hf_token = userdata.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")

# Log in to Hugging Face
login(hf_token)
# Create a directory to store the downloaded model if it doesn't exist
model_dir = "./model_cache"
os.makedirs(model_dir, exist_ok=True)

@st.cache_resource
def load_model():
    try:
        # Download the model file from the Hugging Face Hub
        local_model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_path_in_repo,
            repo_type="model",
            local_dir=model_dir
        )
        # Load the model from the local path
        model = joblib.load(local_model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        st.stop()

loaded_model = load_model()

# 2. Streamlit Application Layout
st.set_page_config(layout="wide")
st.title("Engine Predictive Maintenance App")
st.write("Predict whether an engine requires maintenance based on sensor readings.")

# Sidebar for user inputs
st.sidebar.header("Engine Sensor Readings")

# Input widgets for numerical features
engine_rpm = st.sidebar.slider("Engine RPM", min_value=60, max_value=2300, value=750)
lub_oil_pressure = st.sidebar.slider("Lub Oil Pressure (bar/kPa)", min_value=0.0, max_value=8.0, value=3.5, step=0.1)
fuel_pressure = st.sidebar.slider("Fuel Pressure (bar/kPa)", min_value=0.0, max_value=22.0, value=6.0, step=0.1)
coolant_pressure = st.sidebar.slider("Coolant Pressure (bar/kPa)", min_value=0.0, max_value=8.0, value=2.0, step=0.1)
lub_oil_temp = st.sidebar.slider("Lub Oil Temperature (°C)", min_value=70.0, max_value=90.0, value=78.0, step=0.1)
coolant_temp = st.sidebar.slider("Coolant Temperature (°C)", min_value=60.0, max_value=200.0, value=80.0, step=0.1)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

st.subheader("Current Input Parameters:")
st.write(input_data)

# Prediction button
if st.button("Predict Engine Condition"):
    if loaded_model:
        prediction = loaded_model.predict(input_data)
        prediction_proba = loaded_model.predict_proba(input_data)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("Engine Condition: Faulty (Maintenance Recommended)")
        else:
            st.success("Engine Condition: Normal")

        st.write(f"Confidence (Normal): {prediction_proba[0][0]:.2f}")
        st.write(f"Confidence (Faulty): {prediction_proba[0][1]:.2f}")
    else:
        st.warning("Model not loaded. Please check the logs for errors.")

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import snapshot_download
import os

# --- Model Config --- #
model_repo_id_app = "rakesh1248/random_forest_engine_condition_classifier"
model_filename = "random_forest_model.joblib"

model_dir_app = "./model_cache"
os.makedirs(model_dir_app, exist_ok=True)

@st.cache_resource
def load_model_app():
    try:
        repo_path = snapshot_download(
            repo_id=model_repo_id_app,
            local_dir=model_dir_app
        )

        model_path = os.path.join(repo_path, model_filename)

        model = joblib.load(model_path)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

loaded_model_app = load_model_app()

# --- UI --- #
st.set_page_config(layout="wide")
st.title("Engine Predictive Maintenance App")

st.sidebar.header("Engine Sensor Readings")

engine_rpm = st.sidebar.slider("Engine RPM", 60, 2300, 750)
lub_oil_pressure = st.sidebar.slider("Lub Oil Pressure", 0.0, 8.0, 3.5, 0.1)
fuel_pressure = st.sidebar.slider("Fuel Pressure", 0.0, 22.0, 6.0, 0.1)
coolant_pressure = st.sidebar.slider("Coolant Pressure", 0.0, 8.0, 2.0, 0.1)
lub_oil_temp = st.sidebar.slider("Lub Oil Temperature", 70.0, 90.0, 78.0, 0.1)
coolant_temp = st.sidebar.slider("Coolant Temperature", 60.0, 200.0, 80.0, 0.1)

input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}])

st.write(input_data)

if st.button("Predict"):
    prediction = loaded_model_app.predict(input_data)
    proba = loaded_model_app.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("Faulty Engine")
    else:
        st.success("Normal Engine")

    st.write(f"Normal: {proba[0][0]:.2f}")
    st.write(f"Faulty: {proba[0][1]:.2f}")

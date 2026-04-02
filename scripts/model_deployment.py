import os
from huggingface_hub import HfApi, login

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("Successfully logged into Hugging Face Hub.")
else:
    raise ValueError("HF_TOKEN environment variable not set. Cannot proceed with Hugging Face operations.")

hf_username = "rakesh1248"
MODEL_REPO_ID = f"{hf_username}/random_forest_engine_condition_classifier"
SPACE_REPO_ID = f"{hf_username}/engine_predictive_maintenance_app"
LOCAL_APP_DIR = "engine_predictive_maintenance"

os.makedirs(LOCAL_APP_DIR, exist_ok=True)

print("\n--- Model Deployment (Hugging Face Spaces) ---")

# Define Dockerfile content
dockerfile_content = """
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

# Create a non-root user and switch to it for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Switch working directory to the user's app directory
WORKDIR $HOME/app

# Copy files again with ownership set to the new user
COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
"""
with open(os.path.join(LOCAL_APP_DIR, "Dockerfile"), "w") as f:
    f.write(dockerfile_content)
print("Dockerfile created.")

# Define app.py content (dynamically inject MODEL_REPO_ID)
app_py_content = f"""
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import snapshot_download
import os
import mlflow
import shutil

# Disable MLflow tracking to avoid creating new runs in the deployed app
mlflow.set_tracking_uri("file:///dev/null")

# Define Hugging Face details for model download
hf_username = "{hf_username}"
model_repo_name = "{model_repo_name}"
repo_id_model = f"{{hf_username}}/{{model_repo_name}}"
# Ensure HF_TOKEN is available from environment
hf_token = "{hf_token}"
if hf_token:
    login(token=hf_token)
else:
    st.error("HF_TOKEN environment variable not set. Please set it as a Space secret.")
    st.stop()

# Configuration for loading the model from Hugging Face Models
model_repo_id_app = "{MODEL_REPO_ID}" # Use the full repo_id
model_path_in_repo_app = "random_forest_model.joblib"

model_dir_app = "./model_cache"
os.makedirs(model_dir_app, exist_ok=True)

@st.cache_resource
def load_model_app():
    try:
        local_model_path = hf_hub_download(
            repo_id=model_repo_id_app,
            filename=model_path_in_repo_app,
            repo_type="model",
            local_dir=model_dir_app
        )
        model = joblib.load(local_model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {{e}}")
        st.stop()

loaded_model_app = load_model_app()

# Streamlit Application Layout
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
input_data = pd.DataFrame([{{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}}])

st.subheader("Current Input Parameters:")
st.write(input_data)

# Prediction button
if st.button("Predict Engine Condition"):
    if loaded_model_app:
        prediction = loaded_model_app.predict(input_data)
        prediction_proba = loaded_model_app.predict_proba(input_data)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("Engine Condition: Faulty (Maintenance Recommended)")
        else:
            st.success("Engine Condition: Normal")

        st.write(f"Confidence (Normal): {{prediction_proba[0][0]:.2f}}")
        st.write(f"Confidence (Faulty): {{prediction_proba[0][1]:.2f}}")
    else:
        st.warning("Model not loaded. Please check the logs for errors.")
"""
with open(os.path.join(LOCAL_APP_DIR, "app.py"), "w") as f:
    f.write(app_py_content)
print("app.py created.")

# Define requirements.txt content for the Streamlit app
requirements_content = """
streamlit
pandas
scikit-learn
joblib
huggingface_hub
datasets
"""
with open(os.path.join(LOCAL_APP_DIR, "requirements.txt"), "w") as f:
    f.write(requirements_content)
print("requirements.txt created for deployment.")

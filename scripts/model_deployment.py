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

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
\tPATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""
with open(os.path.join(LOCAL_APP_DIR, "Dockerfile"), "w") as f:
    f.write(dockerfile_content)
print("Dockerfile created.")

# Define app.py content (from cell v0TB-th1fGBM, with dynamic MODEL_REPO_ID and HF_TOKEN handling)
app_py_content = f"""
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import snapshot_download, login
import os

# --- Model Config ---
model_repo_id_app = \"{MODEL_REPO_ID}\" # Dynamically injected from the deployment script
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
        st.error(f\"Error loading model: {{e}}\")
        st.stop()

loaded_model_app = load_model_app()

# --- UI ---
st.set_page_config(layout="wide")
st.title("Engine Predictive Maintenance App")

st.sidebar.header("Engine Sensor Readings")

engine_rpm = st.sidebar.slider("Engine RPM", 60, 2300, 750)
lub_oil_pressure = st.sidebar.slider("Lub Oil Pressure", 0.0, 8.0, 3.5, 0.1)
fuel_pressure = st.sidebar.slider("Fuel Pressure", 0.0, 22.0, 6.0, 0.1)
coolant_pressure = st.sidebar.slider("Coolant Pressure", 0.0, 8.0, 2.0, 0.1)
lub_oil_temp = st.sidebar.slider("Lub Oil Temperature", 70.0, 90.0, 78.0, 0.1)
coolant_temp = st.sidebar.slider("Coolant Temperature", 60.0, 200.0, 80.0, 0.1)

input_data = pd.DataFrame([{{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': lub_oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp
}}])

st.write(input_data)

if st.button("Predict"):
    prediction = loaded_model_app.predict(input_data)
    proba = loaded_model_app.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("Faulty Engine")
    else:
        st.success("Normal Engine")

    st.write(f"Normal: {{proba[0][0]:.2f}}")
    st.write(f"Faulty: {{proba[0][1]:.2f}}")
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

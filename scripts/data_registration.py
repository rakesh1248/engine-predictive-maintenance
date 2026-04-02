import os
from huggingface_hub import HfApi, login
from datasets import load_dataset
import pandas as pd

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("Successfully logged into Hugging Face Hub.")
else:
    raise ValueError("HF_TOKEN environment variable not set. Cannot proceed with Hugging Face operations.")

hf_username = os.environ.get('HF_USERNAME')
DATASET_REPO_ID = f"{hf_username}/engine_predictive_maintenance"
LOCAL_DATA_DIR = "engine_predictive_maintenance/data"

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

print(f"\n--- Data Registration: Ensuring dataset repository '{DATASET_REPO_ID}' ---")

api = HfApi()
api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
print(f"Dataset repository '{DATASET_REPO_ID}' ensured to exist.")

# Optional: Upload a basic 'engine_data.csv' if it doesn't exist already to prime the dataset repo.
# In a real scenario, this data would come from an external source or be committed manually.
# This block ensures a base 'engine_data.csv' exists for the pipeline to start.
try:
    # Check if 'engine_data.csv' exists in the dataset repo
    # Using data_files={'train': 'engine_data.csv'} will check for this specific file.
    load_dataset(DATASET_REPO_ID, split='train', data_files={'train': 'engine_data.csv'})
    print("'engine_data.csv' already exists in the dataset repository.")
except Exception:
    print(f"'engine_data.csv' not found. Creating and uploading a dummy 'engine_data.csv' for initial setup.")
    # Fallback: Generate dummy data if no data can be loaded from HF
    data = {
        'Engine rpm': [750, 1200, 600, 900, 1500, 800, 700, 1100, 500, 1000],
        'Lub oil pressure': [3.5, 4.2, 2.8, 3.9, 4.5, 3.1, 2.5, 4.0, 2.0, 3.8],
        'Fuel pressure': [6.0, 8.5, 5.0, 7.2, 9.0, 5.5, 4.8, 8.0, 4.0, 7.0],
        'Coolant pressure': [2.0, 3.0, 1.5, 2.5, 3.2, 1.8, 1.2, 2.8, 1.0, 2.6],
        'lub oil temp': [78.0, 82.0, 75.0, 79.0, 83.0, 76.0, 74.0, 81.0, 73.0, 80.0],
        'Coolant temp': [80.0, 85.0, 76.0, 81.0, 86.0, 77.0, 75.0, 84.0, 72.0, 82.0],
        'Engine Condition': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    dummy_df = pd.DataFrame(data)
    dummy_csv_path = os.path.join(LOCAL_DATA_DIR, "engine_data.csv")
    os.makedirs(os.path.dirname(dummy_csv_path), exist_ok=True)
    dummy_df.to_csv(dummy_csv_path, index=False)
    api.upload_file(
        path_or_fileobj=dummy_csv_path,
        path_in_repo="engine_data.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset"
    )
    print("Dummy 'engine_data.csv' uploaded to the dataset repository.")

print("\nData Registration step complete.")

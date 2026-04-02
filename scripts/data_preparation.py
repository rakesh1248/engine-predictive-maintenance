import os
import pandas as pd
from huggingface_hub import HfApi, login
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

print("\n--- Data Preparation ---")

# Load the base dataset (assuming it was registered by data_registration.py)
df = None
try:
    print(f"Loading 'engine_data.csv' from {DATASET_REPO_ID}...")
    dataset = load_dataset(DATASET_REPO_ID, data_files={'train': 'engine_data.csv'}, split='train')
    df = dataset.to_pandas()
    print("Original 'engine_data.csv' loaded successfully from Hugging Face.")
except Exception as e:
    print(f"Error loading 'engine_data.csv' from Hugging Face: {e}")
    # Fallback to local dummy data if HF load fails, which should ideally be handled earlier by data_registration.py
    print("Attempting to load locally generated dummy data if available.")
    dummy_csv_path = os.path.join(LOCAL_DATA_DIR, "engine_data.csv")
    if os.path.exists(dummy_csv_path):
        df = pd.read_csv(dummy_csv_path)
        print("Loaded local dummy 'engine_data.csv'.")
    else:
        raise RuntimeError("No 'engine_data.csv' found locally or on Hugging Face. Data preparation aborted.")

# Data Cleaning
print("Checking for duplicate rows...")
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print(f"Removed {df.duplicated().sum()} duplicate rows.")
else:
    print("No duplicate rows found.")

print("Checking for constant columns...")
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
if constant_columns:
    df.drop(columns=constant_columns, axis=1, inplace=True)
    print(f"Constant columns removed: {constant_columns}")
else:
    print("No constant columns found.")

# Split data
X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
print(f"Data split into training ({train_df.shape[0]} rows) and testing ({test_df.shape[0]} rows) sets.")

# Save and Upload train/test data to Hugging Face
train_file_path = os.path.join(LOCAL_DATA_DIR, "train_data.csv")
test_file_path = os.path.join(LOCAL_DATA_DIR, "test_data.csv")
train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)
print("Training and testing data saved locally.")

api = HfApi()

api.upload_file(path_or_fileobj=train_file_path, path_in_repo="train_data.csv", repo_id=DATASET_REPO_ID, repo_type="dataset")
api.upload_file(path_or_fileobj=test_file_path, path_in_repo="test_data.csv", repo_id=DATASET_REPO_ID, repo_type="dataset")
print(f"Training and testing data uploaded to Hugging Face Datasets: {DATASET_REPO_ID}")

print("\nData Preparation step complete.")

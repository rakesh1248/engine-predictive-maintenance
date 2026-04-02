import os
import pandas as pd
import joblib
from huggingface_hub import HfApi, login
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("Successfully logged into Hugging Face Hub.")
else:
    raise ValueError("HF_TOKEN environment variable not set. Cannot proceed with Hugging Face operations.")

hf_username = os.environ.get('HF_USERNAME')
DATASET_REPO_ID = f"{hf_username}/engine_predictive_maintenance"
MODEL_REPO_ID = f"{hf_username}/random_forest_engine_condition_classifier"
LOCAL_MODEL_DIR = "model"

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print("\n--- Model Building with Experimentation Tracking ---")

# Load processed data
print(f"Loading train and test data from {DATASET_REPO_ID}...")
train_dataset = load_dataset(DATASET_REPO_ID, data_files={'train': 'train_data.csv'}, split='train')
test_dataset = load_dataset(DATASET_REPO_ID, data_files={'test': 'test_data.csv'}, split='test')
train_df = train_dataset.to_pandas()
test_df = test_dataset.to_pandas()
print("Train and test data loaded successfully.")

X_train_model = train_df.drop('Engine Condition', axis=1)
y_train_model = train_df['Engine Condition']
X_test_model = test_df.drop('Engine Condition', axis=1)
y_test_model = test_df['Engine Condition']

# Define and train the model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
print("Training Random Forest Classifier...")
model.fit(X_train_model, y_train_model)
print("Model training complete.")

# Evaluate model
y_pred = model.predict(X_test_model)
accuracy = accuracy_score(y_test_model, y_pred)
precision = precision_score(y_test_model, y_pred)
recall = recall_score(y_test_model, y_pred)
f1 = f1_score(y_test_model, y_pred)
print(f"Model Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

# Save and Upload Model to Hugging Face Models
model_file_path = os.path.join(LOCAL_MODEL_DIR, "random_forest_model.joblib")
joblib.dump(model, model_file_path)
print("Model saved locally.")

api = HfApi()

api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=model_file_path, path_in_repo="random_forest_model.joblib", repo_id=MODEL_REPO_ID, repo_type="model")
print(f"Model uploaded to Hugging Face Models: {MODEL_REPO_ID}")

print("\nModel Building with Experimentation Tracking step complete.")

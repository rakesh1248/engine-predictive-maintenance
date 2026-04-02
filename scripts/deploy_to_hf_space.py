import os
from huggingface_hub import HfApi, login

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("Successfully logged into Hugging Face Hub.")
else:
    raise ValueError("HF_TOKEN environment variable not set. Cannot proceed with Hugging Face operations.")

hf_username = os.environ.get('HF_USERNAME')
space_name = "engine_predictive_maintenance_app" # Your desired Hugging Face Space name
repo_id = f"{hf_username}/{space_name}"

api = HfApi()

# 1. Create a new Hugging Face Space repository (if it doesn't exist)
print(f"### Creating/Ensuring Hugging Face Space: {repo_id} ###\n")
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)
print(f"Hugging Face Space '{repo_id}' ensured to exist.\n")

# Define paths to the deployment files (assuming they are in engine_predictive_maintenance/)
deployment_files_dir = "engine_predictive_maintenance"
dockerfile_path = os.path.join(deployment_files_dir, "Dockerfile")
app_file_path = os.path.join(deployment_files_dir, "app.py")
requirements_file_path = os.path.join(deployment_files_dir, "requirements.txt")

# 2. Upload Dockerfile
print("### Uploading Dockerfile ###\n")
api.upload_file(
    path_or_fileobj=dockerfile_path,
    path_in_repo="Dockerfile",
    repo_id=repo_id,
    repo_type="space"
)
print("Dockerfile uploaded successfully!\n")

# 3. Upload app.py
print("### Uploading app.py ###\n")
api.upload_file(
    path_or_fileobj=app_file_path,
    path_in_repo="app.py",
    repo_id=repo_id,
    repo_type="space"
)
print("app.py uploaded successfully!\n")

# 4. Upload requirements.txt
print("### Uploading requirements.txt ###\n")
api.upload_file(
    path_or_fileobj=requirements_file_path,
    path_in_repo="requirements.txt",
    repo_id=repo_id,
    repo_type="space"
)
print("requirements.txt uploaded successfully!\n")

print(f"All deployment files uploaded to Hugging Face Space: https://huggingface.co/spaces/{repo_id}")

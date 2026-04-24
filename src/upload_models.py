"""
upload_models.py
--------------------
Uploads serialized model files from models/ to Hugging Face Hub
for use by the Streamlit app.

RUN ONCE AFTER PREPROCESS.PY HAS COMPLETED SUCCESSFULLY:
    python src/upload_models.py

Requirements
---------------
-   A Hugging Face account
-   A Hugging Face write-access token stored as an environment variable:
        export HF_TOKEN=your_token_here
-   The models/ folder must be populated by preprocess.py first
"""

import os
from huggingface_hub import login, create_repo, upload_file

#Config
HF_USERNAME = "Orrsoft95"

#Name of the Huggingface repo
HF_REPO_NAME = "MyTop5"
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

#Files to upload - MUST match outputs of preprocess.py
MODEL_FILES = [
    "anime_metadata.pkl",
    "content_feature_matrix.pkl",
    "anime_index_map.pkl",
    "svd_model.joblib",
    "anime_titles.pkl"
]

#Upload pickle files to Hugging Face
def main():
    #Authenticate using token from environment variable!
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set.\n"
            "Run: export HF_TOKEN=your_token_here"
        )
    login(token=hf_token)

    #Create repo if it doesn't exist yet
    create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        exist_ok=True, #Don't throw an error if repo already exists
        private=False #Ensure repo is public
    )

    print(f"Repository ready: https://huggingface.co/{HF_REPO_ID}")

    #Upload each model file
    for filename in MODEL_FILES:
        local_path = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(local_path):
            print(f"Skipping {filename} - file not found in models/")
            continue
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading {filename} ({file_size_mb}:.1f MB)...")

        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        print(f"{filename} -> DONE")
    
    print(f"\nAll models uploaded to: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    main()
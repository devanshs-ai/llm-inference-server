import os
from huggingface_hub import snapshot_download

# --- Configuration ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_DIR = "C:/Users/Devansh/holeeshit/llm-inference-server/venv/models/TinyLlama-1.1B-Chat"

def download_model():
    # Create the directory if it doesn't exist
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"Starting download: {MODEL_ID}")
    print(f"Saving to: {os.path.abspath(LOCAL_DIR)}")
    
    try:
        # snapshot_download is the cleanest way to get the full repo
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,  # VERY IMPORTANT for Windows stability
            revision="main"
        )
        print("\nSuccess! All model files are now local.")
        print(f"Total size is approx 2.5GB. Verify the 'models' folder exists.")
    except Exception as e:
        print(f"\nError during download: {e}")

if __name__ == "__main__":
    download_model()
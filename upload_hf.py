import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    print("error: HF_TOKEN not found in .env")
    exit(1)

api = HfApi()
repo_id = "nxsr85/semantic-dsp-mapper-lora"

print("creating repo...")
api.create_repo(repo_id=repo_id, token=token, exist_ok=True)

print("uploading weights...")
api.upload_folder(
    folder_path="lora_safe_audio_model",
    repo_id=repo_id,
    token=token
)

print("model is up.")
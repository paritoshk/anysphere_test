from huggingface_hub import snapshot_download
import os

def download_model():
    model_path = "/workspace/llama3finetune/model"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    auth_token = os.getenv("HF_AUTH_TOKEN")  # Replace with your token
    snapshot_download(model_name, local_dir=model_path, token=auth_token)
    print(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_model()
# download_fixed.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen1.5-0.5B-Chat",
    local_dir="../models/Qwen--Qwen1.5-0.5B-Chat",
    max_workers=2
)
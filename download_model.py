"""
Build-time model check for Render.

The backend now runs a float16 TFLite export of the classifier that is committed
to the repo at app/models/pneumonia_classifier_fp16.tflite (~48 MB), so there is
nothing to download anymore. This script is kept because Render's build command
(`python download_model.py && pip install -r requirements.txt`) still calls it.

If the TFLite file is missing (e.g. Git LFS pointer), it can optionally be fetched
from TFLITE_MODEL_URL.
"""
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "app", "models")
TFLITE_NAME = "pneumonia_classifier_fp16.tflite"
REQUIRED = [TFLITE_NAME, "pneumonia_head.npz", "pneumonia_meta.json"]


def is_valid(path, min_bytes):
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def download(url, target):
    print(f"[DOWNLOAD] {url} -> {target}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = target + ".part"
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, target)


def main():
    tflite_path = os.path.join(MODELS_DIR, TFLITE_NAME)
    if not is_valid(tflite_path, 1024 * 1024):
        url = os.getenv("TFLITE_MODEL_URL", "").strip()
        if url:
            os.makedirs(MODELS_DIR, exist_ok=True)
            download(url, tflite_path)
        else:
            print(f"[ERROR] {tflite_path} is missing and TFLITE_MODEL_URL is not set.")
            sys.exit(1)

    for name in REQUIRED:
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            print(f"[OK] {name} ({os.path.getsize(p) / (1024 * 1024):.2f} MB)")
        else:
            print(f"[WARN] {name} missing - Grad-CAM heatmaps may be unavailable.")
    print("[READY] Lightweight TFLite model bundle is present.")


if __name__ == "__main__":
    main()

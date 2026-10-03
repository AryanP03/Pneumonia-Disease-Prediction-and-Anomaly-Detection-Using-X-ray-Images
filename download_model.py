import os
import sys
import urllib.request

MODEL_FILENAME = "pneumonia_resnet_best_model_1.h5"
DEFAULT_MODEL_URL = "https://github.com/AryanP03/Pneumonia-Disease-Prediction-and-Anomaly-Detection-Using-X-ray-Images/releases/download/v1.0.0/pneumonia_resnet_best_model_1.h5"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATHS = [
    os.path.join(BASE_DIR, "app", MODEL_FILENAME),
    os.path.join(BASE_DIR, MODEL_FILENAME)
]

def check_existing_model():
    for path in TARGET_PATHS:
        if os.path.exists(path):
            size = os.path.getsize(path)
            # A real model file is ~226MB; Git LFS text pointers are ~130 bytes
            if size > 1024 * 1024:
                print(f"[OK] Model weights found at: {path} ({size / (1024 * 1024):.1f} MB)")
                return path
            else:
                print(f"[NOTICE] Found file at {path} but size is only {size} bytes (unpulled Git LFS pointer).")
    return None

def download_model(url):
    target = TARGET_PATHS[0]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp_target = target + ".part"
    print(f"[DOWNLOAD] Downloading model weights from:\n  {url}\nTo:\n  {target}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response, open(temp_target, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunk

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    print(f"\rProgress: {pct:.1f}% ({mb_down:.1f} / {mb_tot:.1f} MB)", end="", flush=True)

        print()
        if os.path.exists(target):
            os.remove(target)
        os.rename(temp_target, target)

        final_size = os.path.getsize(target)
        if final_size > 1024 * 1024:
            print(f"[SUCCESS] Download completed! Verified size: {final_size / (1024 * 1024):.1f} MB")
            return True
        else:
            print(f"[ERROR] Downloaded file is too small ({final_size} bytes).")
            return False
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        if os.path.exists(temp_target):
            try:
                os.remove(temp_target)
            except Exception:
                pass
        return False

def main():
    existing = check_existing_model()
    if existing:
        print("[READY] Model is already present and valid.")
        return

    url = os.getenv("MODEL_DOWNLOAD_URL", "").strip() or DEFAULT_MODEL_URL
    print(f"[INFO] Using model source: {url}")
    success = download_model(url)
    if not success:
        print("[CRITICAL] Could not download model weights.")
        sys.exit(1)

if __name__ == "__main__":
    main()

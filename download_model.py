import os
import sys
import urllib.request

# Name of the model required by the AI diagnostic backend
MODEL_FILENAME = "pneumonia_resnet_best_model_1.h5"

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
    print(f"[DOWNLOAD] Downloading model weights from:\n  {url}\nTo:\n  {target}")

    def progress_hook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100.0)
            mb_down = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            print(f"\rDownloading: {pct:.1f}% ({mb_down:.1f}/{mb_tot:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, target, reporthook=progress_hook)
        print()
        final_size = os.path.getsize(target)
        if final_size > 1024 * 1024:
            print(f"[SUCCESS] Download completed! Size: {final_size / (1024 * 1024):.1f} MB")
            return True
        else:
            print(f"[ERROR] Downloaded file is too small ({final_size} bytes). Verify the URL.")
            return False
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        return False

def main():
    existing = check_existing_model()
    if existing:
        print("[READY] Model is already present and valid.")
        return

    url = os.getenv("MODEL_DOWNLOAD_URL", "").strip()
    if not url:
        print("[INFO] MODEL_DOWNLOAD_URL is not set.")
        print("[INFO] If you are using Git LFS, ensure 'git lfs pull' has run.")
        print("[INFO] Otherwise, set MODEL_DOWNLOAD_URL in your Render Environment Variables to download on build.")
        return

    success = download_model(url)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

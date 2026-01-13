"""Download pre-trained model weights from Google Drive."""

import os
import sys
import subprocess
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

WEIGHTS_FILE_ID = '19bKLEmZx1rEEJONf0igz4O4FJaBQk-Hi'

REQUIRED_WEIGHTS = [
    "weights_rgb/best_pose_model.pth",
    "weights_rgbd/best_pose_model.pth",
    "yolov8n-seg.pt",
]


def download_weights():
    """Download and extract model weights from Google Drive."""
    
    # Check existing
    missing = [f for f in REQUIRED_WEIGHTS if not os.path.exists(f)]
    if not missing:
        print("All weights already exist.")
        return True
    
    print(f"Missing: {', '.join(missing)}")
    
    # Install gdown
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    
    # Download
    url = f'https://drive.google.com/uc?id={WEIGHTS_FILE_ID}'
    zip_path = "model_weights.zip"
    
    print("Downloading weights...")
    gdown.download(url, zip_path, quiet=False, fuzzy=True)
    
    if not os.path.exists(zip_path):
        print("Download failed.")
        return False
    
    # Extract
    print("Extracting...")
    os.makedirs("weights_rgb", exist_ok=True)
    os.makedirs("weights_rgbd", exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(".")
    
    os.remove(zip_path)
    
    # Verify
    still_missing = [f for f in REQUIRED_WEIGHTS if not os.path.exists(f)]
    if still_missing:
        print(f"Failed to extract: {', '.join(still_missing)}")
        return False
    
    print("Weights ready.")
    return True


if __name__ == "__main__":
    success = download_weights()
    sys.exit(0 if success else 1)

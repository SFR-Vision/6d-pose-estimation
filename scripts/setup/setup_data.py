"""Download LineMOD dataset from Google Drive."""

import os
import sys
import subprocess
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

FILE_ID = '1kxNSIQAs_KyOF0EM0OEWUbQHZNiS-mam'
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")


def download_dataset():
    """Download and extract LineMOD dataset."""
    
    # Check if already exists
    data_path = os.path.join(DATASETS_DIR, "Linemod_preprocessed", "data")
    if os.path.exists(data_path) and os.listdir(data_path):
        print(f"Dataset already exists at {data_path}")
        return True
    
    # Install gdown
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Download
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    zip_path = os.path.join(DATASETS_DIR, "Linemod_preprocessed.zip")
    
    print("Downloading dataset...")
    gdown.download(url, zip_path, quiet=False)
    
    if not os.path.exists(zip_path):
        print("Download failed.")
        return False
    
    # Extract
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATASETS_DIR)
        os.remove(zip_path)
        print("Dataset ready.")
        return True
    except zipfile.BadZipFile:
        print("Invalid zip file.")
        return False


if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
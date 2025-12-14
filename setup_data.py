import os
import sys
import subprocess
import zipfile

# ================= CONFIGURATION =================
# Google Drive File ID
FILE_ID = '1kxNSIQAs_KyOF0EM0OEWUbQHZNiS-mam'

# Where the data should live
DATASETS_DIR = "datasets"
# =================================================

def install(package):
    """Installs a package using pip inside the current environment."""
    print(f"🔧 Installing missing package: {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_and_extract():
    try:
        import gdown
    except ImportError:
        install("gdown")
        import gdown

    # 2. Check if data exists
    expected_path = os.path.join(DATASETS_DIR, "Linemod_preprocessed", "data")
    if os.path.exists(expected_path) and len(os.listdir(expected_path)) > 0:
        print(f"✅ Data already exists in {expected_path}")
        print("🚀 Skipping download.")
        return

    # 3. Prepare Folder
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    # 4. Download
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    output_zip = os.path.join(DATASETS_DIR, "Linemod_preprocessed.zip")
    
    print(f"⬇️ Downloading dataset from Google Drive...")
    gdown.download(url, output_zip, quiet=False)

    # 5. Unzip
    if os.path.exists(output_zip):
        print("📦 Unzipping...")
        try:
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(DATASETS_DIR)
            print("✅ Extraction complete!")
            
            # 6. Cleanup
            os.remove(output_zip)
            print("🗑️ Zip file removed.")
            
        except zipfile.BadZipFile:
            print("❌ Error: Downloaded file is not a valid zip.")
    else:
        print("❌ Error: Download failed.")

if __name__ == "__main__":
    download_and_extract()
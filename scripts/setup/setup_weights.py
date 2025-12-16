"""
Download and setup pre-trained weights from Google Drive.
Optional: Skip training and use pre-trained models.
"""
import os
import sys
import subprocess
import zipfile

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ================= CONFIGURATION =================
# Google Drive File ID for pre-trained weights
PRETRAINED_WEIGHTS_ID = '1Nx9w9tjLmpsbmzWogwUIB3QpJhUGW2D7'  # Update after uploading to Drive
# =================================================

def install(package):
    """Installs a package using pip."""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

def download_pretrained_weights():
    """Download pre-trained weights from Google Drive."""
    try:
        import gdown
    except ImportError:
        print("📦 Installing gdown for Google Drive downloads...")
        install("gdown")
        import gdown
    
    # Check if weights already exist
    weights_exist = (
        os.path.exists("weights_rgb/best_pose_model.pth") and 
        os.path.exists("weights_hybrid/best_pose_model.pth") and
        os.path.exists("runs/detect/linemod_yolo/weights/best.pt")
    )
    
    if weights_exist:
        print("✅ Pre-trained weights already exist!")
        print("   - RGB model: weights_rgb/best_pose_model.pth")
        print("   - Hybrid model: weights_hybrid/best_pose_model.pth")
        print("   - YOLO detector: runs/detect/linemod_yolo/weights/best.pt")
        return True
    
    print("⬇️  Downloading pre-trained weights from Google Drive...")
    
    url = f'https://drive.google.com/uc?id={PRETRAINED_WEIGHTS_ID}'
    output_zip = "pretrained_weights.zip"
    
    try:
        # Download with fuzzy matching (more reliable)
        gdown.download(url, output_zip, quiet=False, fuzzy=True)
        
        if not os.path.exists(output_zip):
            print("❌ Download failed - file not found!")
            print("💡 Please check the Google Drive link is publicly accessible")
            return False
        
        file_size = os.path.getsize(output_zip) / (1024 * 1024)  # MB
        print(f"✅ Downloaded {file_size:.1f} MB")
        
        print("📦 Extracting weights...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Move YOLO weights to correct location
        os.makedirs("runs/detect/linemod_yolo/weights", exist_ok=True)
        if os.path.exists("yolo_weights/best.pt"):
            import shutil
            shutil.move("yolo_weights/best.pt", "runs/detect/linemod_yolo/weights/best.pt")
            if os.path.exists("yolo_weights"):
                os.rmdir("yolo_weights")
        
        os.remove(output_zip)
        print("✅ Pre-trained weights loaded successfully!")
        print("   - RGB model: weights_rgb/best_pose_model.pth")
        print("   - Hybrid model: weights_hybrid/best_pose_model.pth")
        print("   - YOLO detector: runs/detect/linemod_yolo/weights/best.pt")
        
        # Verify all files exist
        missing = []
        if not os.path.exists("weights_rgb/best_pose_model.pth"):
            missing.append("RGB weights")
        if not os.path.exists("weights_hybrid/best_pose_model.pth"):
            missing.append("Hybrid weights")
        if not os.path.exists("runs/detect/linemod_yolo/weights/best.pt"):
            missing.append("YOLO weights")
        
        if missing:
            print(f"⚠️  Warning: Missing files: {', '.join(missing)}")
            print("💡 Zip file may be incomplete or have wrong structure")
            return False
        
        return True
            
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        print("💡 Will train from scratch instead.")
        return False

if __name__ == "__main__":
    download_pretrained_weights()

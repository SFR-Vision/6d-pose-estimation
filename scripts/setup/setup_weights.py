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
PRETRAINED_WEIGHTS_ID = '1huMuHCqqlgiJG0330An67qRepFrCZIMr'  # Update after uploading to Drive
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
        install("gdown")
        import gdown
    
    # Check if weights already exist
    if (os.path.exists("weights/best_pose_model.pth") and 
        os.path.exists("weights_rgbd/best_pose_model_rgbd.pth") and
        os.path.exists("yolo_weights/best.pt")):
        print("✅ Pre-trained weights already exist!")
        return True
    
    print("⬇️  Downloading pre-trained weights from Google Drive...")
    
    url = f'https://drive.google.com/uc?id={PRETRAINED_WEIGHTS_ID}'
    output_zip = "pretrained_weights.zip"
    
    try:
        gdown.download(url, output_zip, quiet=False)
        
        if os.path.exists(output_zip):
            print("📦 Extracting weights...")
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Move YOLO weights to correct location
            os.makedirs("runs/detect/linemod_yolo/weights", exist_ok=True)
            if os.path.exists("yolo_weights/best.pt"):
                os.rename("yolo_weights/best.pt", "runs/detect/linemod_yolo/weights/best.pt")
                os.rmdir("yolo_weights")
            
            os.remove(output_zip)
            print("✅ Pre-trained weights loaded successfully!")
            print("   - RGB model: weights/best_pose_model.pth")
            print("   - RGB-D model: weights_rgbd/best_pose_model_rgbd.pth")
            print("   - YOLO detector: runs/detect/linemod_yolo/weights/best.pt")
            return True
        else:
            print("❌ Download failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        print("💡 Will train from scratch instead.")
        return False

if __name__ == "__main__":
    download_pretrained_weights()

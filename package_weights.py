"""
Package pre-trained weights for Google Drive upload.
Creates a zip file with all trained model weights.
"""
import os
import zipfile
from pathlib import Path

def package_weights():
    """Package all trained weights into a single zip file."""
    
    output_zip = "pretrained_weights.zip"
    
    weights_to_package = [
        # RGB pose model
        ("weights/best_pose_model.pth", "weights/best_pose_model.pth"),
        
        # RGB-D pose model
        ("weights_rgbd/best_pose_model_rgbd.pth", "weights_rgbd/best_pose_model_rgbd.pth"),
        
        # YOLO detector
        ("runs/detect/linemod_yolo/weights/best.pt", "yolo_weights/best.pt"),
    ]
    
    print("📦 Packaging pre-trained weights...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, archive_name in weights_to_package:
            if os.path.exists(file_path):
                zipf.write(file_path, archive_name)
                print(f"  ✅ Added: {file_path}")
            else:
                print(f"  ⚠️  Missing: {file_path}")
    
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n✅ Created {output_zip} ({size_mb:.2f} MB)")
    print("\n📤 Next steps:")
    print("1. Upload pretrained_weights.zip to Google Drive")
    print("2. Right-click → Share → Get link → Copy link")
    print("3. Extract file ID from link: drive.google.com/file/d/FILE_ID/view")
    print("4. Update PRETRAINED_WEIGHTS_ID in setup_weights.py")

if __name__ == "__main__":
    package_weights()

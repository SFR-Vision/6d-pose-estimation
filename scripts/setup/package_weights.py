"""
Script to package all model weights into a single zip file for Google Drive upload.
Run this locally after training is complete.
"""

import os
import shutil
import zipfile

# Navigate up from scripts/setup/ to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Weights to include
WEIGHT_FOLDERS = [
    "weights_rgb",
    "weights_rgb_geometric", 
    "weights_rgbd",
    "weights_rgbd_geometric",
    "runs/detect/linemod_yolo/weights"
]

OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "pretrained_weights.zip")


def package_weights():
    print("Packaging model weights...")
    
    # Create temp directory
    temp_dir = os.path.join(PROJECT_ROOT, "_temp_weights")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Copy weight files
    for folder in WEIGHT_FOLDERS:
        src_path = os.path.join(PROJECT_ROOT, folder)
        if not os.path.exists(src_path):
            print(f"  Skipping {folder} (not found)")
            continue
            
        # Determine destination folder name
        if "yolo" in folder:
            dest_folder = "yolo_weights"
        else:
            dest_folder = folder.replace("/", "_")
        
        dest_path = os.path.join(temp_dir, dest_folder)
        os.makedirs(dest_path, exist_ok=True)
        
        # Copy only .pth and .pt files
        for file in os.listdir(src_path):
            if file.endswith(('.pth', '.pt')):
                shutil.copy2(os.path.join(src_path, file), dest_path)
                print(f"  Added: {dest_folder}/{file}")
    
    # Create zip
    print(f"\nCreating {OUTPUT_ZIP}...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    # Get file size
    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\nDone! Created: pretrained_weights.zip ({size_mb:.1f} MB)")
    print("\nNext steps:")
    print("1. Upload pretrained_weights.zip to Google Drive")
    print("2. Right-click -> Share -> Anyone with link -> Copy link")
    print("3. Extract the file ID from the link")
    print("4. Update colab_notebook.ipynb with the file ID")


if __name__ == "__main__":
    package_weights()

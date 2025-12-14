import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix for macOS/KMP issue

from ultralytics import YOLO
import torch

# ================= CONFIGURATION =================
DATA_YAML = os.path.join(PROJECT_ROOT, "datasets", "yolo_ready", "dataset.yaml")

# Checkpoint Paths
PROJECT_DIR = os.path.join(PROJECT_ROOT, "runs", "detect")
EXP_NAME = "linemod_yolo"
LAST_CKPT = os.path.join(PROJECT_DIR, EXP_NAME, "weights", "last.pt")

# Training Parameters
EPOCHS = 5 
BATCH_SIZE = 16 
IMG_SIZE = 640 
WORKERS = 4 
# =================================================

def train_detector():
    if not os.path.exists(DATA_YAML):
        print(f"❌ Error: Could not find {DATA_YAML}")
        return

    print(f"🚀 Starting YOLOv8 Training on {torch.cuda.get_device_name(0)}...")

    # --- AUTO-RESUME LOGIC ---
    if os.path.exists(LAST_CKPT):
        print(f"🔄 Found checkpoint: {LAST_CKPT}")
        print("   Resuming training from where it left off...")
        model = YOLO(LAST_CKPT)
        resume_training = True
    else:
        print("✨ No checkpoint found. Starting fresh training...")
        model = YOLO("yolov8n.pt") 
        resume_training = False
    # -------------------------

    # Train
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=EXP_NAME,
        exist_ok=True,       # Write to the same folder
        resume=resume_training, # Critical flag for resuming
        optimizer='auto',
        verbose=True,
        seed=42
    )
    
    print("\n✅ Training Complete!")
    print(f"   Best Model Saved: {os.path.join(PROJECT_DIR, EXP_NAME, 'weights', 'best.pt')}")

    # Validation
    print("\n📊 Validating Model...")
    metrics = model.val()
    print(f"   mAP@50: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    train_detector()
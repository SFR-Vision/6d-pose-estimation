"""Training script for YOLOv8 object detector."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from ultralytics import YOLO

# Configuration
DATA_YAML = os.path.join(PROJECT_ROOT, "datasets", "yolo_ready", "dataset.yaml")
PROJECT_DIR = os.path.join(PROJECT_ROOT, "runs", "detect")
EXP_NAME = "linemod_yolo"
LAST_CKPT = os.path.join(PROJECT_DIR, EXP_NAME, "weights", "last.pt")

EPOCHS = 5
BATCH_SIZE = 16
IMG_SIZE = 640
WORKERS = 4


def train_detector():
    if not os.path.exists(DATA_YAML):
        print(f"Error: Could not find {DATA_YAML}")
        return

    print(f"Training YOLOv8 on {torch.cuda.get_device_name(0)}")

    # Resume from checkpoint if available
    if os.path.exists(LAST_CKPT):
        print(f"Resuming from checkpoint: {LAST_CKPT}")
        model = YOLO(LAST_CKPT)
        resume_training = True
    else:
        print("Starting fresh training")
        model = YOLO("yolov8n.pt")
        resume_training = False

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=EXP_NAME,
        exist_ok=True,
        resume=resume_training,
        optimizer='auto',
        verbose=True,
        seed=42
    )
    
    print(f"\nTraining complete. Best model: {os.path.join(PROJECT_DIR, EXP_NAME, 'weights', 'best.pt')}")

    # Validation
    print("\nValidating model...")
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    train_detector()
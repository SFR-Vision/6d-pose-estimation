"""Train YOLO segmentation model on LineMOD dataset."""

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

# Configuration
DATA_YAML = os.path.join(PROJECT_ROOT, "datasets", "yolo_seg_ready", "dataset.yaml")
MODEL_NAME = "yolov8n-seg.pt"  # Nano model (fastest, pretrained on COCO)
EPOCHS = 50  # Sufficient for fine-tuning pretrained model
BATCH_SIZE = 48
IMG_SIZE = 640
DEVICE = 0  # GPU 0, or 'cpu'
SAVE_DIR = os.path.join(PROJECT_ROOT, "runs", "segment", "linemod_yolo_seg")

def train():
    """Train YOLO segmentation model."""
    
    print("=" * 60)
    print("Training YOLO Segmentation Model")
    print("=" * 60)
    print(f"Data: {DATA_YAML}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Device: {DEVICE}")
    print("=" * 60 + "\n")
    
    # Check if dataset exists
    if not os.path.exists(DATA_YAML):
        print(f"Dataset not found: {DATA_YAML}")
        print("Run: python scripts/setup/prepare_yolo_seg.py")
        sys.exit(1)
    
    # Load model
    model = YOLO(MODEL_NAME)
    
    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=os.path.dirname(SAVE_DIR),
        name=os.path.basename(SAVE_DIR),
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss weight
        cls=0.5,  # Classification loss weight
        dfl=1.5,  # Distribution focal loss weight
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,
        workers=4,
        verbose=True,
        seed=0,
        deterministic=True,
    )
    
    print("\n" + "-" * 40)
    print("Training Complete!")
    print(f"Best weights: {os.path.join(SAVE_DIR, 'weights', 'best.pt')}")
    print(f"Last weights: {os.path.join(SAVE_DIR, 'weights', 'last.pt')}")
    print("\nValidation results:")
    print(f"  Results saved to: {SAVE_DIR}")
    
    # Run validation on best model
    print("\nRunning Final Validation...")
    
    best_model = YOLO(os.path.join(SAVE_DIR, 'weights', 'best.pt'))
    metrics = best_model.val(data=DATA_YAML, split='test')
    
    print("\nTest Set Metrics:")
    print(f"  mAP50: {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")
    print(f"  Precision: {metrics.seg.mp:.4f}")
    print(f"  Recall: {metrics.seg.mr:.4f}")
    
    print(f"\nTraining plots saved to: {SAVE_DIR}")


if __name__ == "__main__":
    train()

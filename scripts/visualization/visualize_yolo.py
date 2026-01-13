"""YOLO detection visualization on LineMOD images."""

import os
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "segment", "linemod_yolo_seg", "weights", "best.pt")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
OBJECTS = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']


def visualize_results(num_samples=4):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    
    print(f"Loading {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Collect images from all objects
    all_images = []
    for obj_id in OBJECTS:
        rgb_dir = os.path.join(DATA_DIR, obj_id, "rgb")
        if os.path.exists(rgb_dir):
            for f in os.listdir(rgb_dir):
                if f.endswith(".png"):
                    all_images.append((obj_id, os.path.join(rgb_dir, f)))
    
    if not all_images:
        print("No images found")
        return
    
    selected = random.sample(all_images, min(len(all_images), num_samples))
    
    plt.figure(figsize=(15, 5))
    for i, (obj_id, img_path) in enumerate(selected):
        results = model(img_path, verbose=False)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(res_rgb)
        plt.title(f"Object {obj_id}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_results()

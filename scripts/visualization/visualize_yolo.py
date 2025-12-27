"""Visualization script for YOLO object detection results."""

import os
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Configuration
MODEL_PATH = os.path.join("runs", "detect", "linemod_yolo", "weights", "best.pt")
TEST_DIR = os.path.join("datasets", "yolo_ready", "images", "test")


def visualize_results(num_samples=4):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if not os.path.exists(TEST_DIR):
        print(f"Error: Test folder not found at {TEST_DIR}")
        return

    images = [f for f in os.listdir(TEST_DIR) if f.endswith(".png")]
    
    if len(images) == 0:
        print("No images found in test folder")
        return

    selected_images = random.sample(images, min(len(images), num_samples))
    
    print(f"Visualizing {len(selected_images)} random test images...")

    plt.figure(figsize=(15, 5))
    
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(TEST_DIR, img_name)
        
        results = model(img_path)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(res_rgb)
        plt.title(img_name.split("_")[0])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("Done")


if __name__ == "__main__":
    visualize_results()

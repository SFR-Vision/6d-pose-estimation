import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# --- FIX FOR WINDOWS KERNEL CRASH ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ------------------------------------

from ultralytics import YOLO
import cv2
import random
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# Path to your trained model
MODEL_PATH = os.path.join("runs", "detect", "linemod_yolo", "weights", "best.pt")

# Path to test images
TEST_DIR = os.path.join("datasets", "yolo_ready", "images", "test")
# =================================================

def visualize_results(num_samples=4):
    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Did the training finish successfully?")
        return

    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 2. Get list of test images
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test folder not found at {TEST_DIR}")
        return

    images = [f for f in os.listdir(TEST_DIR) if f.endswith(".png")]
    
    if len(images) == 0:
        print("⚠️ No images found in test folder!")
        return

    # 3. Pick random images
    selected_images = random.sample(images, min(len(images), num_samples))
    
    print(f"👀 Visualizing {len(selected_images)} random test images...")

    # 4. Plot
    plt.figure(figsize=(15, 5))
    
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(TEST_DIR, img_name)
        
        # Run Inference
        results = model(img_path)
        
        # YOLO has a built-in .plot() function that draws the boxes
        # It returns a BGR numpy array (OpenCV format)
        res_plotted = results[0].plot()
        
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Subplot
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(res_rgb)
        plt.title(img_name.split("_")[0]) # Show Object ID as title
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("✅ Done! Check the popup window.")

if __name__ == "__main__":
    visualize_results()

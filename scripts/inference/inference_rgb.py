"""Inference script for RGB pose estimation model."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from ultralytics import YOLO

from models.pose_net_rgb import PoseNetRGB
from utils.mesh_utils import load_mesh_corners
from utils.visualization import project_points, draw_3d_box, draw_axes
from utils.camera import DEFAULT_K

# Configuration
YOLO_PATH = os.path.join("runs", "detect", "linemod_yolo", "weights", "best.pt")
POSE_PATH = os.path.join("weights_rgb", "best_pose_model.pth")
MESH_DIR = os.path.join("datasets", "Linemod_preprocessed", "models")

CLASS_ID_TO_OBJ_NAME = {
    0: "01", 1: "02", 2: "04", 3: "05", 4: "06", 5: "08",
    6: "09", 7: "10", 8: "11", 9: "12", 10: "13", 11: "14", 12: "15",
}


def run_inference(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing: {img_path}")
    
    # Load models
    yolo = YOLO(YOLO_PATH)
    pose_net = PoseNetRGB(pretrained=False).to(device)
    ckpt = torch.load(POSE_PATH, map_location=device, weights_only=False)
    pose_net.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    pose_net.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_img = cv2.imread(img_path)
    if original_img is None:
        print("Image not found")
        return
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = rgb_img.shape
    
    viz_img = original_img.copy()
    K = DEFAULT_K

    # Detect objects
    results = yolo(img_path, verbose=False)
    if not results[0].boxes:
        print("No objects detected")
        return

    print(f"Found {len(results[0].boxes)} objects")

    # Process each detection
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        obj_id_str = CLASS_ID_TO_OBJ_NAME.get(cls_id, "01")
        
        # Prepare crop
        c_x_box, c_y_box = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        size = max(w, h) * 1.2
        new_x, new_y = int(c_x_box - size/2), int(c_y_box - size/2)
        new_size = int(size)
        
        pad_l = max(0, -new_x)
        pad_t = max(0, -new_y)
        pad_r = max(0, (new_x + new_size) - w_img)
        pad_b = max(0, (new_y + new_size) - h_img)
        padded_img = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        crop = padded_img[new_y + pad_t:new_y + pad_t + new_size, new_x + pad_l:new_x + pad_l + new_size]
        crop_resized = cv2.resize(crop, (224, 224))
        
        # Pose inference
        input_tensor = transform(crop_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_rot, pred_trans = pose_net(input_tensor)
        pred_rot = pred_rot.cpu().numpy()[0]
        pred_trans = pred_trans.cpu().numpy()[0]
        
        # Geometric correction
        pred_z = pred_trans[2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pred_x = (c_x_box - cx) * pred_z / fx
        pred_y = (c_y_box - cy) * pred_z / fy
        corrected_trans = np.array([pred_x, pred_y, pred_z])

        # Visualization
        corners_3d = load_mesh_corners(MESH_DIR, obj_id_str)
        if corners_3d is not None:
            box_2d = project_points(corners_3d, pred_rot, corrected_trans, K)
            draw_3d_box(viz_img, box_2d, color=(0, 255, 0), thickness=2)
            draw_axes(viz_img, pred_rot, corrected_trans, K, scale=0.1)
            
            cv2.putText(viz_img, f"{obj_id_str} ({conf:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add legend
    h_img, w_img = viz_img.shape[:2]
    cv2.putText(viz_img, "Axes: X=Red(Front) | Y=Green(Left) | Z=Blue(Top)", 
                (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(viz_img, "Green Box = 3D Pose Estimation", 
                (10, h_img - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title("RGB Model: 6D Pose Inference with YOLO Detection")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_dir = os.path.join("datasets", "yolo_ready", "images", "test")
    
    if len(sys.argv) > 1:
        TEST_IMG = sys.argv[1]
    else:
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.jpg')]
            if len(files) > 0:
                random_file = np.random.choice(files)
                TEST_IMG = os.path.join(test_dir, random_file)
                print(f"Selected: {random_file}")
            else:
                print(f"No images found in {test_dir}")
                sys.exit(1)
        else:
            print(f"Directory not found: {test_dir}")
            sys.exit(1)
        
    run_inference(TEST_IMG)
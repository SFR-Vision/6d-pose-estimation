"""Inference script for RGBD Geometric Model with YOLO Detection."""

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

from models.pose_net_rgbd_geometric import PoseNetRGBDGeometric
from utils.mesh_utils import load_mesh_corners
from utils.visualization import project_points, draw_3d_box, draw_axes
from utils.camera import DEFAULT_K

# Configuration
YOLO_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "linemod_yolo", "weights", "best.pt")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights_rgbd_geometric", "best_pose_model.pth")
MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
TEST_DIR = os.path.join(PROJECT_ROOT, "datasets", "yolo_ready", "images", "test")

CLASS_ID_TO_OBJ_NAME = {
    0: "01", 1: "02", 2: "04", 3: "05", 4: "06", 5: "08",
    6: "09", 7: "10", 8: "11", 9: "12", 10: "13", 11: "14", 12: "15",
}


def run_inference(img_path, depth_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RGBD-Geometric inference with YOLO on {device}")
    print(f"Processing: {img_path}")
    
    # Load models
    if not os.path.exists(YOLO_PATH):
        print(f"YOLO model not found: {YOLO_PATH}")
        return
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Pose weights not found: {WEIGHTS_PATH}")
        return
        
    yolo = YOLO(YOLO_PATH)
    model = PoseNetRGBDGeometric(pretrained=False).to(device)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Models loaded successfully")
    
    # Load RGB image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Image not found: {img_path}")
        return
    
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = rgb_img.shape
    viz_img = original_img.copy()
    K = DEFAULT_K.copy()
    
    # Load depth image - parse filename to find from dataset
    depth_img = None
    if depth_path and os.path.exists(depth_path):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        print(f"Loaded depth from: {depth_path}")
    
    if depth_img is None:
        # Try to parse filename: format is "XX_YYYY.png" where XX=object, YYYY=frame
        base_name = os.path.basename(img_path)
        if '_' in base_name:
            parts = base_name.replace('.png', '').replace('.jpg', '').split('_')
            if len(parts) >= 2:
                obj_id = parts[0]
                frame_id = parts[1]
                dataset_depth_path = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", 
                                                   "data", obj_id, "depth", f"{frame_id}.png")
                if os.path.exists(dataset_depth_path):
                    depth_img = cv2.imread(dataset_depth_path, cv2.IMREAD_UNCHANGED)
                    print(f"Loaded depth from dataset: {dataset_depth_path}")
    
    if depth_img is None:
        depth_img = np.zeros((h_img, w_img), dtype=np.uint16)
        print("Warning: No depth image found, using zeros")
    else:
        # Squeeze if depth has extra channel dimension
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]
    
    # YOLO detection
    results = yolo(img_path, verbose=False)
    if not results[0].boxes:
        print("No objects detected by YOLO")
        return
    
    print(f"YOLO detected {len(results[0].boxes)} objects")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_size = 224
    
    # Process each detection
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        obj_id_str = CLASS_ID_TO_OBJ_NAME.get(cls_id, "01")
        
        # Bbox center in original image
        c_x_box, c_y_box = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        size = max(w, h) * 1.2
        crop_x1, crop_y1 = int(c_x_box - size/2), int(c_y_box - size/2)
        crop_size = int(size)
        
        # Padding
        pad_l = max(0, -crop_x1)
        pad_t = max(0, -crop_y1)
        pad_r = max(0, (crop_x1 + crop_size) - w_img)
        pad_b = max(0, (crop_y1 + crop_size) - h_img)
        
        padded_rgb = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        padded_depth = cv2.copyMakeBorder(depth_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        
        # Adjust crop coordinates for padding
        adj_x1 = crop_x1 + pad_l
        adj_y1 = crop_y1 + pad_t
        
        crop_rgb = padded_rgb[adj_y1:adj_y1+crop_size, adj_x1:adj_x1+crop_size]
        crop_depth = padded_depth[adj_y1:adj_y1+crop_size, adj_x1:adj_x1+crop_size]
        

        crop_rgb_resized = cv2.resize(crop_rgb, (img_size, img_size))
        crop_depth_resized = cv2.resize(crop_depth.astype(np.float32), (img_size, img_size))
        
        # Scale factor for crop -> resize
        scale = img_size / crop_size
        
        # Bbox center in CROP coordinates (then scale to resized)
        center_in_padded_x = c_x_box + pad_l
        center_in_padded_y = c_y_box + pad_t
        center_in_crop_x = (center_in_padded_x - adj_x1) * scale
        center_in_crop_y = (center_in_padded_y - adj_y1) * scale
        center_in_crop = np.array([center_in_crop_x, center_in_crop_y], dtype=np.float32)
        center_in_crop = np.clip(center_in_crop, 0, img_size - 1)
        
        # Adjust camera intrinsics for crop and resize (matching dataset_rgbd.py)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # x1 in dataset = crop_x1 in inference (before padding adjustment)
        cx_crop = (cx + pad_l - crop_x1) * scale
        cy_crop = (cy + pad_t - crop_y1) * scale
        fx_crop = fx * scale
        fy_crop = fy * scale
        K_crop = np.array([
            [fx_crop, 0, cx_crop],
            [0, fy_crop, cy_crop],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Normalize depth for CNN input
        depth_meters = crop_depth_resized / 1000.0
        depth_min, depth_max = 0.1, 1.6
        depth_normalized = np.clip((depth_meters - depth_min) / (depth_max - depth_min), 0, 1)
        depth_normalized[depth_meters < 0.01] = 0
        
        # Prepare inputs
        input_rgb = transform(crop_rgb_resized).unsqueeze(0).to(device)
        # Depth normalized: add channel dim [1, 224, 224] -> [1, 1, 224, 224]
        input_depth = torch.tensor(depth_normalized[..., np.newaxis], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        # Depth raw: [224, 224] -> [1, 224, 224]
        input_depth_raw = torch.tensor(depth_meters, dtype=torch.float32).unsqueeze(0).to(device)
        bbox_center = torch.from_numpy(center_in_crop).float().unsqueeze(0).to(device)
        cam_matrix = torch.tensor(K_crop, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Pose inference
        with torch.no_grad():
            pred_quat, pred_trans = model(input_rgb, input_depth, input_depth_raw, bbox_center, cam_matrix)
        
        pred_quat = pred_quat.cpu().numpy()[0]  # (4,)
        pred_trans = pred_trans.cpu().numpy().flatten()  # (3,)
        

        # Visualization using ORIGINAL camera matrix K (not crop)
        corners = load_mesh_corners(MESH_DIR, obj_id_str)
        if corners is not None:
            box_2d = project_points(corners, pred_quat, pred_trans, K)
            draw_3d_box(viz_img, box_2d, (255, 255, 0), 2)
            draw_axes(viz_img, pred_quat, pred_trans, K, scale=0.1)
            
            cv2.putText(viz_img, f"{obj_id_str} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Add legend
    cv2.putText(viz_img, "Axes: X=Red(Front) | Y=Green(Left) | Z=Blue(Top)", 
                (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(viz_img, "Cyan Box = 3D Pose Estimation", 
                (10, h_img - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title("RGBD-Geometric Model: 6D Pose Inference with YOLO Detection")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMG = sys.argv[1]
        TEST_DEPTH = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        if os.path.exists(TEST_DIR):
            files = [f for f in os.listdir(TEST_DIR) if f.endswith('.png') or f.endswith('.jpg')]
            if len(files) > 0:
                random_file = np.random.choice(files)
                TEST_IMG = os.path.join(TEST_DIR, random_file)
                TEST_DEPTH = None
                print(f"Selected: {random_file}")
            else:
                print(f"No images found in {TEST_DIR}")
                sys.exit(1)
        else:
            print(f"Directory not found: {TEST_DIR}")
            sys.exit(1)
    
    run_inference(TEST_IMG, TEST_DEPTH)

"""Inference script for RGB Geometric Model with YOLO Detection."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from torchvision import transforms
from ultralytics import YOLO

from models.pose_net_rgb_geometric import PoseNetRGBGeometric
from utils.mesh_utils import load_mesh_corners
from utils.visualization import project_points, draw_3d_box, draw_axes
from utils.camera import DEFAULT_K
from utils.inference_utils import load_ground_truth, load_model_points, compute_add, parse_image_filename

# Configuration
YOLO_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "linemod_yolo", "weights", "best.pt")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights_rgb_geometric", "best_pose_model.pth")
MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
TEST_DIR = os.path.join(PROJECT_ROOT, "datasets", "yolo_ready", "images", "test")

CLASS_ID_TO_OBJ_NAME = {
    0: "01", 1: "02", 2: "04", 3: "05", 4: "06", 5: "08",
    6: "09", 7: "10", 8: "11", 9: "12", 10: "13", 11: "14", 12: "15",
}


def put_text_with_outline(img, text, org, color, scale=0.55, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_label_with_bg(img, text, x, y, color, scale=0.55, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(max(5, min(x, img.shape[1] - tw - 5)))
    y = int(max(th + 5, min(y, img.shape[0] - 5)))
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def run_inference(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RGB-Geometric inference with YOLO on {device}")
    print(f"Processing: {img_path}")
    
    # Load models
    if not os.path.exists(YOLO_PATH):
        print(f"YOLO model not found: {YOLO_PATH}")
        return
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Pose weights not found: {WEIGHTS_PATH}")
        return
        
    yolo = YOLO(YOLO_PATH)
    model = PoseNetRGBGeometric(pretrained=False).to(device)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Models loaded successfully")
    
    # Load image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Image not found: {img_path}")
        return
    
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = rgb_img.shape
    viz_img = original_img.copy()
    K = DEFAULT_K
    
    # YOLO detection
    results = yolo(img_path, verbose=False)
    if not results[0].boxes:
        print("No objects detected by YOLO")
        return
    
    print(f"YOLO detected {len(results[0].boxes)} objects")
    
    # Parse image filename to get GT info
    file_obj_id, frame_id = parse_image_filename(img_path)
    gt_rot, gt_trans = None, None
    if file_obj_id and frame_id:
        gt_rot, gt_trans = load_ground_truth(DATA_ROOT, file_obj_id, frame_id)
        if gt_rot is not None:
            print(f"Ground truth loaded for object {file_obj_id}, frame {frame_id}")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process each detection
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        obj_id_str = CLASS_ID_TO_OBJ_NAME.get(cls_id, "01")
        
        # Prepare crop with padding
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
        
        # Prepare inputs
        input_tensor = transform(crop_resized).unsqueeze(0).to(device)
        bbox_center = torch.tensor([[c_x_box, c_y_box]], dtype=torch.float32).to(device)
        cam_matrix = torch.tensor(K, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Pose inference
        with torch.no_grad():
            pred_quat, pred_trans = model(input_tensor, bbox_center, cam_matrix)
        
        pred_quat = pred_quat.cpu().numpy()[0]  # Quaternion (4,)
        pred_trans = pred_trans.cpu().numpy()[0]
        
        # Compute ADD metric if ground truth is available
        metrics = None
        if gt_rot is not None and gt_trans is not None and obj_id_str == file_obj_id:
            model_points = load_model_points(MESH_DIR, obj_id_str)
            if model_points is not None:
                metrics = compute_add(pred_quat, pred_trans, gt_rot, gt_trans, model_points)
                xyz = metrics['trans_xyz_mm']
                print(f"  Object {obj_id_str}:")
                print(f"    ADD:         {metrics['add_mm']:.1f}mm")
                print(f"    Trans Error: {metrics['trans_error_mm']:.1f}mm  (X:{xyz[0]:+.1f} Y:{xyz[1]:+.1f} Z:{xyz[2]:+.1f})")
                print(f"    Rot Error:   {metrics['rot_error_deg']:.1f}°")
                print(f"    Pred Trans:  [{metrics['pred_trans'][0]:.3f}, {metrics['pred_trans'][1]:.3f}, {metrics['pred_trans'][2]:.3f}] m")
                print(f"    GT Trans:    [{metrics['gt_trans'][0]:.3f}, {metrics['gt_trans'][1]:.3f}, {metrics['gt_trans'][2]:.3f}] m")
        
        # Visualization (project_points handles quaternion conversion)
        corners = load_mesh_corners(MESH_DIR, obj_id_str)
        if corners is not None:
            # Draw predicted 3D box (cyan, thick)
            box_2d = project_points(corners, pred_quat, pred_trans, K)
            draw_3d_box(viz_img, box_2d, (0, 255, 255), 2)
            draw_axes(viz_img, pred_quat, pred_trans, K, scale=0.1)
            
            # Draw ground truth 3D box (green, thin)
            if gt_rot is not None and gt_trans is not None:
                gt_quat = R.from_matrix(gt_rot).as_quat()  # Convert rotation matrix to quaternion
                gt_box_2d = project_points(corners, gt_quat, gt_trans, K)
                draw_3d_box(viz_img, gt_box_2d, (0, 255, 0), 1)  # Green, thin lines
            
            # Display label with ADD metric if available
            if metrics is not None:
                add_mm = metrics['add_mm']
                label = f"{obj_id_str} ADD:{add_mm:.0f}mm T:{metrics['trans_error_mm']:.0f}mm R:{metrics['rot_error_deg']:.0f}deg"
                # Color based on accuracy: green if < 20mm, yellow if < 50mm, red otherwise
                if add_mm < 20:
                    color = (0, 255, 0)  # Green - excellent
                elif add_mm < 50:
                    color = (0, 255, 255)  # Yellow - good
                else:
                    color = (0, 0, 255)  # Red - poor
            else:
                label = f"{obj_id_str} ({conf:.2f})"
                color = (0, 255, 255)
            
            # Draw label for this detection
            draw_label_with_bg(viz_img, label, x1, max(25, y1 - 10), color, scale=0.55, thickness=1)
    
    # Add legend (outside the detection loop)
    put_text_with_outline(viz_img, "Cyan=Predicted | Green=GroundTruth", 
                          (10, h_img - 70), (255, 255, 255), scale=0.6, thickness=2)
    put_text_with_outline(viz_img, "Axes: X=Red(Front) | Y=Green(Left) | Z=Blue(Top)", 
                          (10, h_img - 45), (255, 255, 255), scale=0.6, thickness=2)
    put_text_with_outline(viz_img, "ADD: Green - Excellent | Yellow - Good | Red - Poor", 
                          (10, h_img - 20), (0, 255, 255), scale=0.6, thickness=2)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title("RGB-Geometric Model: 6D Pose Inference with YOLO Detection")
    plt.axis("off")
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMG = sys.argv[1]
    else:
        if os.path.exists(TEST_DIR):
            files = [f for f in os.listdir(TEST_DIR) if f.endswith('.png') or f.endswith('.jpg')]
            if len(files) > 0:
                random_file = np.random.choice(files)
                TEST_IMG = os.path.join(TEST_DIR, random_file)
                print(f"Selected: {random_file}")
            else:
                print(f"No images found in {TEST_DIR}")
                sys.exit(1)
        else:
            print(f"Directory not found: {TEST_DIR}")
            sys.exit(1)
    
    run_inference(TEST_IMG)

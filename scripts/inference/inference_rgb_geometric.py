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


def load_ground_truth(obj_id_str, frame_id):
    """Load ground truth pose and bbox from LineMOD dataset."""
    gt_path = os.path.join(DATA_ROOT, obj_id_str, "gt.yml")
    if not os.path.exists(gt_path):
        return None, None, None
    
    with open(gt_path, 'r') as f:
        gts = yaml.safe_load(f)
    
    if frame_id not in gts:
        return None, None, None
    
    for anno in gts[frame_id]:
        if str(int(anno['obj_id'])).zfill(2) == obj_id_str:
            gt_rot = np.array(anno['cam_R_m2c']).reshape(3, 3)
            gt_trans = np.array(anno['cam_t_m2c']) / 1000.0  # mm to meters
            gt_bbox = anno.get('obj_bb', None)  # [x, y, w, h]
            return gt_rot, gt_trans, gt_bbox
    
    return None, None, None


def load_model_points(obj_id_str, num_points=500):
    """Load 3D model points for ADD computation."""
    ply_path = os.path.join(MESH_DIR, f"obj_{obj_id_str}.ply")
    if not os.path.exists(ply_path):
        return None
    
    verts = []
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    header_end = False
    for line in lines:
        if "end_header" in line:
            header_end = True
            continue
        if header_end:
            vals = line.strip().split()
            if len(vals) >= 3:
                verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    
    pts = np.array(verts) / 1000.0  # mm to meters
    
    # Filter outliers and downsample
    distances = np.linalg.norm(pts, axis=1)
    pts = pts[distances < 0.5]
    
    if len(pts) > num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    
    return pts


def compute_add(pred_quat, pred_trans, gt_rot, gt_trans, model_points):
    """Compute ADD (Average Distance of Model Points) and error breakdown.
    
    Returns:
        dict with 'add_mm', 'trans_error_mm', 'rot_error_deg', 'trans_xyz_mm'
    """
    # Convert quaternion to rotation matrix
    pred_R = R.from_quat([pred_quat[0], pred_quat[1], pred_quat[2], pred_quat[3]]).as_matrix()
    
    # Translation error breakdown (X, Y, Z in mm)
    trans_diff = (pred_trans - gt_trans) * 1000  # to mm
    trans_error = np.linalg.norm(trans_diff)
    
    # Rotation error (geodesic angle in degrees)
    R_diff = pred_R @ gt_rot.T
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_error_rad = np.arccos(cos_angle)
    rot_error_deg = np.degrees(rot_error_rad)
    
    # Transform model points
    gt_points = model_points @ gt_rot.T + gt_trans
    pred_points = model_points @ pred_R.T + pred_trans
    
    # Compute ADD
    add_dist = np.linalg.norm(pred_points - gt_points, axis=1).mean() * 1000
    
    return {
        'add_mm': add_dist,
        'trans_error_mm': trans_error,
        'trans_xyz_mm': trans_diff,  # [X, Y, Z] errors
        'rot_error_deg': rot_error_deg,
        'pred_trans': pred_trans,
        'gt_trans': gt_trans
    }


def parse_image_filename(img_path):
    """Parse object ID and frame ID from test image filename (e.g., '01_0219.png')."""
    filename = os.path.basename(img_path)
    parts = filename.replace('.png', '').replace('.jpg', '').split('_')
    if len(parts) >= 2:
        obj_id_str = parts[0]
        frame_id = int(parts[1])
        return obj_id_str, frame_id
    return None, None


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
    gt_rot, gt_trans, gt_bbox = None, None, None
    if file_obj_id and frame_id:
        gt_rot, gt_trans, gt_bbox = load_ground_truth(file_obj_id, frame_id)
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
            model_points = load_model_points(obj_id_str)
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
            
            cv2.putText(viz_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add legend
    cv2.putText(viz_img, "Cyan=Predicted | Green=GroundTruth", 
                (10, h_img - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(viz_img, "Axes: X=Red(Front) | Y=Green(Left) | Z=Blue(Top)", 
                (10, h_img - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(viz_img, "ADD: Green<20mm | Yellow<50mm | Red>50mm", 
                (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
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

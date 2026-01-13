"""Inference script for RGBD Model (5-channel) with YOLO Segmentation."""

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

from models.pose_net_rgbd import PoseNetRGBD
from utils.mesh_utils import load_mesh_corners
from utils.visualization import project_points, draw_3d_box, draw_axes
from utils.camera import DEFAULT_K
from utils.inference_utils import load_ground_truth, load_model_points, compute_add, parse_image_filename

# Configuration
YOLO_PATH = os.path.join(PROJECT_ROOT, "runs", "segment", "linemod_yolo_seg", "weights", "best.pt")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights_rgbd", "best_pose_model.pth")
MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
OBJECTS = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']

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
    model = PoseNetRGBD(pretrained=False).to(device)
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
    
    # Precompute depth in meters for center sampling
    depth_img_m = depth_img.astype(np.float32) / 1000.0

    # YOLO detection
    results = yolo(img_path, verbose=False)
    if not results[0].boxes:
        print("No objects detected by YOLO")
        return
    
    print(f"YOLO detected {len(results[0].boxes)} objects")
    
    # Parse image filename to get GT info
    file_obj_id, frame_id = parse_image_filename(img_path)
    print(f"Parsed: obj_id={file_obj_id}, frame_id={frame_id}")
    gt_rot, gt_trans = None, None
    if file_obj_id and frame_id is not None:
        gt_rot, gt_trans = load_ground_truth(DATA_ROOT, file_obj_id, frame_id)
        if gt_rot is not None:
            print(f"Ground truth loaded for object {file_obj_id}, frame {frame_id}")
        else:
            print(f"Ground truth not found for object {file_obj_id}, frame {frame_id}")
    
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
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
        crop_depth_resized = cv2.resize(crop_depth.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # Use original bbox center and K (consistent with dataset_rgbd.py)
        bbox_center_orig = np.array([c_x_box, c_y_box], dtype=np.float32)
        
        # Compute z_sensor from depth at center (matching dataset_rgbd.py)
        depth_meters = crop_depth_resized / 1000.0
        center = img_size // 2
        region = 5
        depth_center_region = depth_meters[center-region:center+region+1, center-region:center+region+1]
        valid_mask = depth_center_region > 0.01
        if valid_mask.sum() > 0:
            z_sensor = np.median(depth_center_region[valid_mask])
        else:
            z_sensor = 0.5  # fallback
        z_sensor = np.clip(z_sensor, 0.1, 2.0)
        
        # Normalize depth for CNN input (same as training)
        depth_min, depth_max = 0.1, 2.0
        depth_normalized = (depth_meters - depth_min) / (depth_max - depth_min)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_normalized[depth_meters < 0.01] = 0
        
        # Create mask from YOLO segmentation if available, otherwise use simple threshold
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            try:
                # Get mask for this detection
                mask_data = results[0].masks.data[results[0].boxes.cls == cls_id]
                if len(mask_data) > 0:
                    mask_full = mask_data[0].cpu().numpy()
                    mask_full = cv2.resize(mask_full, (w_img, h_img))
                else:
                    mask_full = np.ones((h_img, w_img), dtype=np.float32)
            except:
                mask_full = np.ones((h_img, w_img), dtype=np.float32)
        else:
            # Fallback: create mask from bounding box
            mask_full = np.zeros((h_img, w_img), dtype=np.float32)
            mask_full[y1:y2, x1:x2] = 1.0
        
        # Crop and resize mask
        padded_mask = cv2.copyMakeBorder(mask_full, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        crop_mask = padded_mask[adj_y1:adj_y1+crop_size, adj_x1:adj_x1+crop_size]
        crop_mask_resized = cv2.resize(crop_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensors (matching dataset_rgbd.py preprocessing)
        rgb_tensor = torch.from_numpy(crop_rgb_resized).permute(2, 0, 1).float() / 255.0
        rgb_tensor = transform(rgb_tensor)  # Normalize
        
        depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(crop_mask_resized).unsqueeze(0).float()  # (1, H, W)
        
        # Concatenate to 5-channel RGBDM tensor
        rgbdm = torch.cat([rgb_tensor, depth_tensor, mask_tensor], dim=0)  # (5, H, W)
        rgbdm = rgbdm.unsqueeze(0).to(device)  # (1, 5, H, W)
        
        z_sensor_tensor = torch.tensor([z_sensor], dtype=torch.float32).to(device)
        bbox_center = torch.from_numpy(bbox_center_orig).float().unsqueeze(0).to(device)
        cam_matrix = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Pose inference
        with torch.no_grad():
            pred_quat, pred_trans = model(rgbdm, z_sensor_tensor, bbox_center, cam_matrix)
        
        pred_quat = pred_quat.cpu().numpy()[0]  # (4,)
        pred_trans = pred_trans.cpu().numpy().flatten()  # (3,)
        
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
        
        # Visualization using ORIGINAL camera matrix K (not crop)
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
    plt.title("RGBD-Geometric Model: 6D Pose Inference with YOLO Detection")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMG = sys.argv[1]
        TEST_DEPTH = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Select random image from LineMOD
        obj_id = np.random.choice(OBJECTS)
        rgb_dir = os.path.join(DATA_ROOT, obj_id, "rgb")
        files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        frame = np.random.choice(files).replace('.png', '')
        TEST_IMG = os.path.join(rgb_dir, f"{frame}.png")
        TEST_DEPTH = os.path.join(DATA_ROOT, obj_id, "depth", f"{frame}.png")
        print(f"Selected: Object {obj_id}, Frame {frame}")
    
    run_inference(TEST_IMG, TEST_DEPTH)

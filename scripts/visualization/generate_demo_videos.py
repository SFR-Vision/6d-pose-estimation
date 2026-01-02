"""
Generate demo videos for each LineMOD object showing model predictions.

Creates 13 separate videos, one for each object, with:
- Original image
- Predicted 3D bounding box (cyan)
- Ground truth 3D bounding box (green, thin)
- XYZ axes
- Metrics overlay (ADD, Trans error, Rot error)

Usage:
    python scripts/visualization/generate_demo_videos.py
    python scripts/visualization/generate_demo_videos.py --model rgb_geometric
    python scripts/visualization/generate_demo_videos.py --object 05  # Single object
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import yaml
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.pose_net_rgb_geometric import PoseNetRGBGeometric
from models.pose_net_rgb import PoseNetRGB
from utils.visualization import project_points, draw_3d_box, draw_axes
from utils.mesh_utils import load_mesh_corners

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "demo_videos")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OBJ_IDS = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
OBJ_NAMES = {
    '01': 'ape', '02': 'benchvise', '04': 'camera', '05': 'can',
    '06': 'cat', '08': 'driller', '09': 'duck', '10': 'eggbox',
    '11': 'glue', '12': 'holepuncher', '13': 'iron', '14': 'lamp', '15': 'phone'
}

# Default camera intrinsics (LineMOD)
DEFAULT_K = np.array([
    [572.4114, 0, 325.2611],
    [0, 573.57043, 242.04899],
    [0, 0, 1]
], dtype=np.float32)


def load_model(model_type):
    """Load pose estimation model."""
    if model_type == 'rgb_geometric':
        model = PoseNetRGBGeometric(pretrained=False)
        weights_path = os.path.join(PROJECT_ROOT, "weights_rgb_geometric", "best_pose_model.pth")
    elif model_type == 'rgb':
        model = PoseNetRGB(pretrained=False)
        weights_path = os.path.join(PROJECT_ROOT, "weights_rgb", "best_pose_model.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print(f"Loaded {model_type} model from {weights_path}")
    return model


def load_ground_truth(obj_id_str, frame_id):
    """Load ground truth pose from LineMOD dataset."""
    gt_path = os.path.join(DATA_ROOT, obj_id_str, "gt.yml")
    info_path = os.path.join(DATA_ROOT, obj_id_str, "info.yml")
    
    if not os.path.exists(gt_path):
        return None, None, None, None
    
    with open(gt_path, 'r') as f:
        gts = yaml.safe_load(f)
    with open(info_path, 'r') as f:
        infos = yaml.safe_load(f)
    
    if frame_id not in gts:
        return None, None, None, None
    
    for anno in gts[frame_id]:
        if str(int(anno['obj_id'])).zfill(2) == obj_id_str:
            gt_rot = np.array(anno['cam_R_m2c']).reshape(3, 3)
            gt_trans = np.array(anno['cam_t_m2c']) / 1000.0  # mm to meters
            bbox = anno['obj_bb']
            cam_K = np.array(infos[frame_id]['cam_K']).reshape(3, 3).astype(np.float32)
            return gt_rot, gt_trans, bbox, cam_K
    
    return None, None, None, None


def compute_add(pred_quat, pred_trans, gt_rot, gt_trans, model_points):
    """Compute ADD metric."""
    pred_R = R.from_quat([pred_quat[0], pred_quat[1], pred_quat[2], pred_quat[3]]).as_matrix()
    
    gt_points = model_points @ gt_rot.T + gt_trans
    pred_points = model_points @ pred_R.T + pred_trans
    
    add_dist = np.linalg.norm(pred_points - gt_points, axis=1).mean() * 1000  # mm
    trans_error = np.linalg.norm((pred_trans - gt_trans) * 1000)  # mm
    
    # Rotation error
    R_diff = pred_R @ gt_rot.T
    cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    rot_error = np.degrees(np.arccos(cos_angle))
    
    return add_dist, trans_error, rot_error


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
    if len(pts) > num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    
    return pts


def process_frame(model, model_type, img_path, obj_id_str, frame_id, corners, model_points, transform):
    """Process a single frame and return visualization with metrics."""
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]
    
    # Load ground truth
    gt_rot, gt_trans, bbox, cam_K = load_ground_truth(obj_id_str, frame_id)
    if gt_rot is None:
        return None, None
    
    K = cam_K if cam_K is not None else DEFAULT_K
    
    # Prepare crop
    x, y, w, h = bbox
    c_x, c_y = x + w/2, y + h/2
    size = max(w, h) * 1.2
    x1, y1 = int(c_x - size/2), int(c_y - size/2)
    new_size = int(size)
    
    # Padding
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, (x1 + new_size) - w_img)
    pad_b = max(0, (y1 + new_size) - h_img)
    
    padded_img = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    crop = padded_img[y1 + pad_t:y1 + pad_t + new_size, x1 + pad_l:x1 + pad_l + new_size]
    crop_resized = cv2.resize(crop, (224, 224))
    
    # Prepare inputs
    input_tensor = transform(crop_resized).unsqueeze(0).to(DEVICE)
    bbox_center = torch.tensor([[c_x, c_y]], dtype=torch.float32).to(DEVICE)
    cam_matrix = torch.tensor(K, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        if model_type == 'rgb_geometric':
            pred_quat, pred_trans = model(input_tensor, bbox_center, cam_matrix)
        else:
            pred_quat, pred_trans = model(input_tensor)
    
    pred_quat = pred_quat.cpu().numpy()[0]
    pred_trans = pred_trans.cpu().numpy()[0]
    
    # Compute metrics
    add_mm, trans_err, rot_err = compute_add(pred_quat, pred_trans, gt_rot, gt_trans, model_points)
    metrics = {'add_mm': add_mm, 'trans_error_mm': trans_err, 'rot_error_deg': rot_err}
    
    # Visualization
    viz_img = img.copy()
    
    # Draw ground truth box (green, thin)
    gt_quat = R.from_matrix(gt_rot).as_quat()
    gt_box_2d = project_points(corners, gt_quat, gt_trans, K)
    draw_3d_box(viz_img, gt_box_2d, (0, 255, 0), 1)
    
    # Draw predicted box (cyan, thick)
    pred_box_2d = project_points(corners, pred_quat, pred_trans, K)
    draw_3d_box(viz_img, pred_box_2d, (0, 255, 255), 2)
    
    # Draw axes
    draw_axes(viz_img, pred_quat, pred_trans, K, scale=0.1)
    
    # Metrics overlay
    if add_mm < 20:
        color = (0, 255, 0)  # Green
    elif add_mm < 50:
        color = (0, 255, 255)  # Yellow
    else:
        color = (0, 0, 255)  # Red
    
    # Frame info
    cv2.putText(viz_img, f"Frame: {frame_id:04d}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(viz_img, f"ADD: {add_mm:.1f}mm", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(viz_img, f"Trans: {trans_err:.1f}mm  Rot: {rot_err:.1f}deg", (10, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Legend at bottom
    cv2.putText(viz_img, "Cyan=Predicted | Green=GroundTruth", 
                (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return viz_img, metrics


def generate_video_for_object(model, model_type, obj_id_str, output_path, fps=30):
    """Generate demo video for a single object."""
    obj_name = OBJ_NAMES.get(obj_id_str, obj_id_str)
    print(f"\nProcessing object {obj_id_str} ({obj_name})...")
    
    # Load mesh corners and model points
    corners = load_mesh_corners(MESH_DIR, obj_id_str)
    model_points = load_model_points(obj_id_str)
    
    if corners is None or model_points is None:
        print(f"  Skipping - mesh not found")
        return
    
    # Get all RGB images for this object
    rgb_dir = os.path.join(DATA_ROOT, obj_id_str, "rgb")
    if not os.path.exists(rgb_dir):
        print(f"  Skipping - RGB folder not found")
        return
    
    images = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    print(f"  Found {len(images)} frames")
    
    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Video writer
    first_img = cv2.imread(os.path.join(rgb_dir, images[0]))
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Stats
    total_add = 0
    count = 0
    
    # Process frames
    for img_name in tqdm(images, desc=f"  Object {obj_id_str}"):
        frame_id = int(img_name.split('.')[0])
        img_path = os.path.join(rgb_dir, img_name)
        
        viz_img, metrics = process_frame(
            model, model_type, img_path, obj_id_str, frame_id, 
            corners, model_points, transform
        )
        
        if viz_img is not None:
            video_writer.write(viz_img)
            total_add += metrics['add_mm']
            count += 1
    
    video_writer.release()
    
    avg_add = total_add / max(count, 1)
    print(f"  Saved: {output_path}")
    print(f"  Avg ADD: {avg_add:.1f}mm over {count} frames")


def main():
    parser = argparse.ArgumentParser(description="Generate demo videos for LineMOD objects")
    parser.add_argument('--model', type=str, default='rgb_geometric',
                        choices=['rgb', 'rgb_geometric'],
                        help='Model type to use')
    parser.add_argument('--object', type=str, default=None,
                        help='Single object ID to process (e.g., 05). Default: all objects')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS (default: 30)')
    args = parser.parse_args()
    
    print(f"Generating demo videos on {DEVICE}")
    print(f"Model: {args.model}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model = load_model(args.model)
    
    # Process objects
    objects_to_process = [args.object] if args.object else OBJ_IDS
    
    for obj_id in objects_to_process:
        obj_name = OBJ_NAMES.get(obj_id, obj_id)
        output_path = os.path.join(OUTPUT_DIR, f"obj_{obj_id}_{obj_name}.mp4")
        generate_video_for_object(model, args.model, obj_id, output_path, args.fps)
    
    print(f"\nAll videos saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

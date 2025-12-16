"""
Compare RGB-only (ResNet50) vs Hybrid (ResNet50 Rotation + Custom CNN + Pinhole Translation)
Visualizes predictions side-by-side with ground truth
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import yaml
import torch
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

from models.pose_net_rgb import PoseNet
from models.pose_net_hybrid import PoseNetHybrid

# ================= CONFIGURATION =================
YOLO_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "linemod_yolo", "weights", "best.pt")
RGB_WEIGHTS = os.path.join(PROJECT_ROOT, "weights_rgb", "best_pose_model.pth")
HYBRID_WEIGHTS = os.path.join(PROJECT_ROOT, "weights_hybrid", "best_pose_model.pth")
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "comparison_rgb_vs_hybrid")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("\n" + "="*70)
print("RGB-ONLY vs HYBRID MODEL COMPARISON")
print("="*70)
print(f"Device: {DEVICE}")
print(f"RGB Model: {RGB_WEIGHTS}")
print(f"Hybrid Model: {HYBRID_WEIGHTS}")
print("="*70 + "\n")

# ================= LOAD MODELS =================
print("📦 Loading models...")

# YOLO
yolo = YOLO(YOLO_PATH)
yolo.to(DEVICE)

# RGB-only model
rgb_model = PoseNet(pretrained=False).to(DEVICE)
rgb_ckpt = torch.load(RGB_WEIGHTS, map_location=DEVICE)
rgb_model.load_state_dict(rgb_ckpt['model_state_dict'] if 'model_state_dict' in rgb_ckpt else rgb_ckpt)
rgb_model.eval()

# Hybrid model
hybrid_model = PoseNetHybrid(pretrained=False).to(DEVICE)
hybrid_ckpt = torch.load(HYBRID_WEIGHTS, map_location=DEVICE)
hybrid_model.load_state_dict(hybrid_ckpt['model_state_dict'] if 'model_state_dict' in hybrid_ckpt else hybrid_ckpt)
hybrid_model.eval()

print("✅ Models loaded\n")

# ================= LOAD 3D MODELS =================
def load_ply(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    verts = []
    header_end = False
    for line in lines:
        if "end_header" in line:
            header_end = True
            continue
        if header_end:
            vals = line.strip().split()
            if len(vals) >= 3:
                verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(verts)

print("📦 Loading 3D models...")
models_3d = {}
for obj_id in range(1, 16):
    ply_path = os.path.join(MODEL_DIR, f"obj_{obj_id:02d}.ply")
    if os.path.exists(ply_path):
        pts = load_ply(ply_path)
        pts = pts / 1000.0  # mm to meters
        # Filter outliers
        distances = np.linalg.norm(pts, axis=1)
        valid_mask = distances < 0.5
        pts = pts[valid_mask]
        models_3d[obj_id - 1] = pts
print(f"✅ Loaded {len(models_3d)} 3D models\n")

# ================= HELPER FUNCTIONS =================
def compute_add(pred_rot, pred_trans, gt_rot, gt_trans, model_points):
    """Compute ADD error"""
    pred_points = (pred_rot @ model_points.T).T + pred_trans
    gt_points = (gt_rot @ model_points.T).T + gt_trans
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return np.mean(distances) * 1000  # meters to mm

def load_mesh_corners(obj_id):
    """Load 3D bounding box corners from mesh using percentile-based filtering"""
    obj_id_str = str(obj_id + 1).zfill(2)
    ply_name = f"obj_{obj_id_str}.ply"
    path = os.path.join(MODEL_DIR, ply_name)
    
    if not os.path.exists(path):
        return np.array([[-0.05]*3, [0.05]*3])
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    verts = []
    header_end = False
    for line in lines:
        if "end_header" in line:
            header_end = True
            continue
        if header_end:
            vals = line.strip().split()
            if len(vals) >= 3:
                verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    
    # mm -> meters
    verts = np.array(verts) / 1000.0
    
    # Filter outliers
    distances = np.linalg.norm(verts, axis=1)
    valid_mask = distances < 0.3  # 30cm radius
    verts_clean = verts[valid_mask] if len(verts[valid_mask]) > 0 else verts
    
    # Use percentiles for robust bbox
    min_pt = np.percentile(verts_clean, 1, axis=0)
    max_pt = np.percentile(verts_clean, 99, axis=0)
    
    # 8 corners of 3D bounding box
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]]
    ])
    
    return corners

def project_points(points_3d, rot_mat, trans, cam_K):
    """Project 3D points to 2D using camera intrinsics"""
    # Transform to camera frame
    points_cam = (rot_mat @ points_3d.T).T + trans
    
    # Project to image plane
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    
    x_2d = (points_cam[:, 0] / points_cam[:, 2]) * fx + cx
    y_2d = (points_cam[:, 1] / points_cam[:, 2]) * fy + cy
    
    return np.stack([x_2d, y_2d], axis=1).astype(np.int32)

def draw_3d_box(img, points_2d, color, thickness=2, style='solid'):
    """Draw 3D bounding box with optional dashed lines"""
    # 12 edges of a 3D box
    lines = [(0,1), (1,2), (2,3), (3,0),  # Bottom face
             (4,5), (5,6), (6,7), (7,4),  # Top face
             (0,4), (1,5), (2,6), (3,7)]  # Vertical edges
    
    for s, e in lines:
        pt1 = tuple(points_2d[s])
        pt2 = tuple(points_2d[e])
        
        if style == 'dashed':
            draw_dashed_line(img, pt1, pt2, color, thickness)
        else:
            cv2.line(img, pt1, pt2, color, thickness)
    
    return img

def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=8):
    """Draw a dashed line"""
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    dashes = int(dist / dash_length)
    
    for i in range(dashes):
        if i % 2 == 0:
            start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                    int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes))
            end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i+1) / dashes),
                  int(pt1[1] + (pt2[1] - pt1[1]) * (i+1) / dashes))
            cv2.line(img, start, end, color, thickness)

# ================= PROCESS TEST IMAGES =================
print("🔍 Processing test images...")

obj_folders = sorted([f for f in os.listdir(DATA_ROOT) if f.isdigit()])
total_rgb_error = []
total_hybrid_error = []

for obj_folder in obj_folders:
    obj_id = int(obj_folder) - 1
    
    if obj_id not in models_3d:
        continue
    
    base_path = os.path.join(DATA_ROOT, obj_folder)
    gt_path = os.path.join(base_path, 'gt.yml')
    info_path = os.path.join(base_path, 'info.yml')
    rgb_path = os.path.join(base_path, 'rgb')
    depth_path = os.path.join(base_path, 'depth')
    
    if not os.path.exists(gt_path) or not os.path.exists(info_path):
        continue
    
    with open(gt_path, 'r') as f:
        gts = yaml.safe_load(f)
    
    with open(info_path, 'r') as f:
        infos = yaml.safe_load(f)
    
    # Get one test image (cycle 9)
    images = sorted([img for img in os.listdir(rgb_path) if img.endswith(".png")])
    test_images = [img for i, img in enumerate(images) if i % 10 == 9]
    
    if not test_images:
        continue
    
    img_name = test_images[0]
    frame_id = int(img_name.split('.')[0])
    
    if frame_id not in gts or frame_id not in infos:
        continue
    
    # Load image and depth
    img = cv2.imread(os.path.join(rgb_path, img_name))
    depth_img = cv2.imread(os.path.join(depth_path, img_name), cv2.IMREAD_UNCHANGED)
    
    if img is None or depth_img is None:
        continue
    
    # Get ground truth for this object
    gt_anno = None
    for anno in gts[frame_id]:
        if str(int(anno['obj_id'])).zfill(2) == obj_folder:
            gt_anno = anno
            break
    
    if gt_anno is None:
        continue
    
    gt_rot = np.array(gt_anno['cam_R_m2c']).reshape(3, 3)
    gt_trans = np.array(gt_anno['cam_t_m2c']) / 1000.0  # mm to meters
    cam_K = np.array(infos[frame_id]['cam_K']).reshape(3, 3)
    
    # YOLO detection
    results = yolo.predict(img, verbose=False)
    detections = results[0].boxes
    
    if len(detections) == 0:
        print(f"⚠️  Object {obj_folder}: No YOLO detection")
        continue
    
    # Use first detection
    bbox = detections.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = bbox
    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
    
    # Crop for RGB model
    c_x, c_y = x + w/2, y + h/2
    size = max(w, h) * 1.2
    x1_crop = int(c_x - size/2)
    y1_crop = int(c_y - size/2)
    
    h_img, w_img = img.shape[:2]
    pad_l = max(0, -x1_crop)
    pad_t = max(0, -y1_crop)
    pad_r = max(0, (x1_crop + int(size)) - w_img)
    pad_b = max(0, (y1_crop + int(size)) - h_img)
    
    img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    depth_padded = cv2.copyMakeBorder(depth_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    
    x1_crop += pad_l
    y1_crop += pad_t
    
    rgb_crop = img_padded[y1_crop:y1_crop+int(size), x1_crop:x1_crop+int(size)]
    depth_crop = depth_padded[y1_crop:y1_crop+int(size), x1_crop:x1_crop+int(size)]
    
    rgb_crop = cv2.resize(rgb_crop, (224, 224))
    depth_crop = cv2.resize(depth_crop, (224, 224))
    
    # Preprocess RGB
    rgb_input = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_input = (rgb_input - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    rgb_tensor = torch.from_numpy(rgb_input).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    # Bbox center and camera matrix for hybrid model
    bbox_center = torch.tensor([[c_x, c_y]], dtype=torch.float32).to(DEVICE)
    cam_matrix = torch.from_numpy(cam_K).unsqueeze(0).float().to(DEVICE)
    
    # ===== RGB MODEL PREDICTION =====
    with torch.no_grad():
        pred_rot_rgb, pred_trans_rgb = rgb_model(rgb_tensor)
    
    # Both models output [x, y, z, w] (scipy default format)
    quat_rgb = pred_rot_rgb[0].cpu().numpy()
    pred_rot_mat_rgb = R.from_quat(quat_rgb).as_matrix()
    pred_trans_rgb = pred_trans_rgb[0].cpu().numpy()
    
    # ===== HYBRID MODEL PREDICTION (RGB-ONLY INPUT) =====
    with torch.no_grad():
        pred_rot_hybrid, pred_trans_hybrid = hybrid_model(rgb_tensor, bbox_center, cam_matrix)
    
    # Hybrid also outputs [x, y, z, w] (scipy default)
    quat_hybrid = pred_rot_hybrid[0].cpu().numpy()
    pred_rot_mat_hybrid = R.from_quat(quat_hybrid).as_matrix()
    pred_trans_hybrid = pred_trans_hybrid[0].cpu().numpy()
    
    # ===== COMPUTE ERRORS =====
    model_pts = models_3d[obj_id]
    
    add_error_rgb = compute_add(pred_rot_mat_rgb, pred_trans_rgb, gt_rot, gt_trans, model_pts)
    add_error_hybrid = compute_add(pred_rot_mat_hybrid, pred_trans_hybrid, gt_rot, gt_trans, model_pts)
    
    total_rgb_error.append(add_error_rgb)
    total_hybrid_error.append(add_error_hybrid)
    
    print(f"Object {obj_folder}: RGB={add_error_rgb:.1f}mm | Hybrid={add_error_hybrid:.1f}mm")
    
    # ===== VISUALIZE WITH 3D BOUNDING BOXES =====
    # Load 3D bbox corners
    corners_3d = load_mesh_corners(obj_id)
    
    # Project 3D corners to 2D
    gt_box_2d = project_points(corners_3d, gt_rot, gt_trans, cam_K)
    rgb_box_2d = project_points(corners_3d, pred_rot_mat_rgb, pred_trans_rgb, cam_K)
    hybrid_box_2d = project_points(corners_3d, pred_rot_mat_hybrid, pred_trans_hybrid, cam_K)
    
    # Create three versions
    img_gt = img.copy()
    img_rgb = img.copy()
    img_hybrid = img.copy()
    
    # Draw 3D boxes
    draw_3d_box(img_gt, gt_box_2d, (0, 255, 0), thickness=3)  # Green - GT
    draw_3d_box(img_rgb, rgb_box_2d, (255, 255, 0), thickness=3)  # Yellow - RGB prediction
    draw_3d_box(img_rgb, gt_box_2d, (0, 255, 0), thickness=2, style='dashed')  # Green dashed GT
    draw_3d_box(img_hybrid, hybrid_box_2d, (255, 0, 255), thickness=3)  # Magenta - Hybrid prediction
    draw_3d_box(img_hybrid, gt_box_2d, (0, 255, 0), thickness=2, style='dashed')  # Green dashed GT
    
    # Add labels
    cv2.putText(img_gt, f"Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_rgb, f"RGB: {add_error_rgb:.1f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_hybrid, f"Hybrid: {add_error_hybrid:.1f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Concatenate horizontally
    combined = np.hstack([img_gt, img_rgb, img_hybrid])
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f"obj_{obj_folder}_comparison.jpg")
    cv2.imwrite(output_path, combined)

# ================= SUMMARY =================
# Convert to numpy arrays
rgb_errors = np.array(total_rgb_error)
hybrid_errors = np.array(total_hybrid_error)

# Define thresholds for ADD-S accuracy (standard LineMOD thresholds)
thresholds = [20, 30, 50, 100]  # mm

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)
print(f"\n📊 Average ADD Error:")
print(f"   RGB Model:    {np.mean(rgb_errors):.2f} mm")
print(f"   Hybrid Model: {np.mean(hybrid_errors):.2f} mm")
print(f"   Improvement:  {np.mean(rgb_errors) - np.mean(hybrid_errors):.2f} mm ({(1 - np.mean(hybrid_errors)/np.mean(rgb_errors))*100:.1f}% better)")

print(f"\n🎯 ADD-S Accuracy (% of predictions below threshold):")
print(f"   {'Threshold':<12} {'RGB Model':<15} {'Hybrid Model':<15} {'Winner'}")
print(f"   {'-'*12} {'-'*15} {'-'*15} {'-'*10}")
for thresh in thresholds:
    rgb_acc = (rgb_errors < thresh).sum() / len(rgb_errors) * 100
    hybrid_acc = (hybrid_errors < thresh).sum() / len(hybrid_errors) * 100
    winner = "🏆 Hybrid" if hybrid_acc > rgb_acc else "🏆 RGB" if rgb_acc > hybrid_acc else "Tie"
    print(f"   < {thresh:3d} mm      {rgb_acc:5.1f}%          {hybrid_acc:5.1f}%          {winner}")

print(f"\n📈 Error Distribution:")
print(f"   {'Metric':<15} {'RGB Model':<15} {'Hybrid Model'}")
print(f"   {'-'*15} {'-'*15} {'-'*15}")
print(f"   {'Median':<15} {np.median(rgb_errors):>7.2f} mm    {np.median(hybrid_errors):>7.2f} mm")
print(f"   {'Std Dev':<15} {np.std(rgb_errors):>7.2f} mm    {np.std(hybrid_errors):>7.2f} mm")
print(f"   {'Min':<15} {np.min(rgb_errors):>7.2f} mm    {np.min(hybrid_errors):>7.2f} mm")
print(f"   {'Max':<15} {np.max(rgb_errors):>7.2f} mm    {np.max(hybrid_errors):>7.2f} mm")

print(f"\n💾 Visualizations saved to: {OUTPUT_DIR}")
print("="*70)

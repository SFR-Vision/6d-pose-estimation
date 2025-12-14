import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import sys

# Custom Modules
try:
    from models.pose_net import PoseNet
except ImportError:
    print("❌ Error: Could not import PoseNet.")
    sys.exit(1)

# ================= CONFIGURATION =================
YOLO_PATH = os.path.join("runs", "detect", "linemod_yolo", "weights", "best.pt")
POSE_PATH = os.path.join("weights", "best_pose_model.pth")
TEST_IMG_DIR = os.path.join("datasets", "yolo_ready", "images", "test")
ORIG_DATA_DIR = os.path.join("datasets", "Linemod_preprocessed", "data")
MESH_DIR = os.path.join("datasets", "Linemod_preprocessed", "models")
# =================================================

def load_mesh_data(obj_id_str):
    """
    Loads mesh and calculates a TIGHT box using percentiles to ignore outliers.
    """
    ply_name = f"obj_{obj_id_str}.ply"
    path = os.path.join(MESH_DIR, ply_name)
    
    if not os.path.exists(path):
        print(f"⚠️ Warning: Mesh {ply_name} not found.")
        return None, None

    with open(path, 'r') as f:
        lines = f.readlines()
        
    verts = []
    header_end = False
    for line in lines:
        if "end_header" in line: header_end = True; continue
        if header_end:
            vals = line.strip().split()
            if len(vals) >= 3:
                verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    
    # 1. MM -> Meters
    verts = np.array(verts) / 1000.0
    
    # 2. FILTER: Remove points > 30cm away from center (Hard Logic)
    # This matches your intuition that objects can't be huge
    distances = np.linalg.norm(verts, axis=1)
    valid_mask = distances < 0.3 
    verts_clean = verts[valid_mask]
    
    if len(verts_clean) == 0: verts_clean = verts

    # 3. STATISTICAL BOX (The Fix)
    # Instead of min() and max(), we take the 1st and 99th percentile.
    # This ignores the remaining 2% of noise points that expand the box.
    min_pt = np.percentile(verts_clean, 1, axis=0)
    max_pt = np.percentile(verts_clean, 99, axis=0)
    
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]], # 0
        [max_pt[0], min_pt[1], min_pt[2]], # 1
        [max_pt[0], max_pt[1], min_pt[2]], # 2
        [min_pt[0], max_pt[1], min_pt[2]], # 3
        [min_pt[0], min_pt[1], max_pt[2]], # 4
        [max_pt[0], min_pt[1], max_pt[2]], # 5
        [max_pt[0], max_pt[1], max_pt[2]], # 6
        [min_pt[0], max_pt[1], max_pt[2]]  # 7
    ])
    
    # Downsample for visualization
    if len(verts_clean) > 500:
        idx = np.random.choice(len(verts_clean), 500, replace=False)
        verts_vis = verts_clean[idx]
    else:
        verts_vis = verts_clean
        
    return corners, verts_vis

def get_gt_and_K(obj_id_str, frame_id):
    gt_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "gt.yml")
    info_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "info.yml")
    
    r_mat, t, K = None, None, None
    
    # Load K
    if os.path.exists(info_path):
        with open(info_path, 'r') as f: infos = yaml.safe_load(f)
        k_list = infos[frame_id]['cam_K'] if frame_id in infos else infos[list(infos.keys())[0]]['cam_K']
        K = np.array(k_list).reshape(3, 3)
    else:
        K = np.array([[572.4, 0, 325.2], [0, 573.5, 242.0], [0, 0, 1]])

    # Load GT
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f: gts = yaml.safe_load(f)
        if frame_id in gts:
            for anno in gts[frame_id]:
                if str(int(anno['obj_id'])).zfill(2) == obj_id_str:
                    t = np.array(anno['cam_t_m2c']) / 1000.0
                    r_mat = np.array(anno['cam_R_m2c']).reshape(3, 3)
                    break
    return r_mat, t, K

def project_points(points_3d, r_vec, t_vec, K):
    if r_vec.shape == (4,): r_mat = R.from_quat(r_vec).as_matrix()
    else: r_mat = r_vec
    
    p_cam = (r_mat @ points_3d.T).T + t_vec
    z = p_cam[:, 2].copy()
    z[z==0] = 0.001
    
    p_2d = np.zeros((points_3d.shape[0], 2))
    p_2d[:, 0] = (p_cam[:, 0] * K[0, 0] / z) + K[0, 2]
    p_2d[:, 1] = (p_cam[:, 1] * K[1, 1] / z) + K[1, 2]
    return p_2d.astype(int)

def draw_box(img, pts_2d, color, thickness=2):
    # Draw edges of the box
    lines = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for s, e in lines:
        cv2.line(img, tuple(pts_2d[s]), tuple(pts_2d[e]), color, thickness)
    return img

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Running Geometry-Corrected Visualization...")
    
    yolo = YOLO(YOLO_PATH)
    pose_net = PoseNet(pretrained=False).to(device)
    ckpt = torch.load(POSE_PATH, map_location=device)
    if 'model_state_dict' in ckpt: pose_net.load_state_dict(ckpt['model_state_dict'])
    else: pose_net.load_state_dict(ckpt)
    pose_net.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".png")]
    img_name = np.random.choice(img_files)
    print(f"👀 Analyzing: {img_name}")
    
    try:
        parts = img_name.split('_')
        obj_id_str = parts[0]
        frame_id = int(parts[1].split('.')[0])
    except: obj_id_str, frame_id = "01", 0
    
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    original_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = rgb_img.shape
    
    # YOLO
    results = yolo(img_path, verbose=False)
    if not results[0].boxes: print("No YOLO box"); return
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Calculate Center of YOLO Box (u, v)
    c_x_box = (x1 + x2) / 2
    c_y_box = (y1 + y2) / 2
    
    # Prepare Crop
    w, h = x2-x1, y2-y1
    size = max(w, h) * 1.2
    new_x = int(c_x_box - size/2)
    new_y = int(c_y_box - size/2)
    new_size = int(size)
    
    pad_l = max(0, -new_x); pad_t = max(0, -new_y)
    pad_r = max(0, (new_x + new_size) - w_img)
    pad_b = max(0, (new_y + new_size) - h_img)
    padded_img = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    crop = padded_img[new_y+pad_t:new_y+pad_t+new_size, new_x+pad_l:new_x+pad_l+new_size]
    crop_resized = cv2.resize(crop, (224, 224))
    
    input_tensor = transform(crop_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_rot, pred_trans = pose_net(input_tensor)
    pred_rot = pred_rot.cpu().numpy()[0]
    pred_trans = pred_trans.cpu().numpy()[0]

    # === 🔥 GEOMETRIC CORRECTION 🔥 ===
    # Use GT K for back-projection
    _, _, K = get_gt_and_K(obj_id_str, frame_id)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # We trust the model's Z (Depth), but NOT its X/Y.
    # We calculate X/Y using the center of the YOLO box.
    pred_z = pred_trans[2] 
    
    # Math: x = (u - cx) * z / fx
    pred_x = (c_x_box - cx) * pred_z / fx
    pred_y = (c_y_box - cy) * pred_z / fy
    
    corrected_trans = np.array([pred_x, pred_y, pred_z])
    
    print(f"   Original Pred: {pred_trans}")
    print(f"   Corrected T:   {corrected_trans}")
    # ==================================

    # === DRAWING ===
    viz_img = original_img.copy()
    box_corners_3d, mesh_points_3d = load_mesh_data(obj_id_str)
    
    if box_corners_3d is not None:
        gt_r, gt_t, _ = get_gt_and_K(obj_id_str, frame_id)
        
        # 1. GROUND TRUTH (Green)
        if gt_r is not None:
            box_2d = project_points(box_corners_3d, gt_r, gt_t, K)
            draw_box(viz_img, box_2d, (0, 255, 0), 2)
        
        # 2. PREDICTION (Yellow) - Using CORRECTED TRANS
        # Draw Box
        box_2d_pred = project_points(box_corners_3d, pred_rot, corrected_trans, K)
        draw_box(viz_img, box_2d_pred, (0, 255, 255), 2)
        
        # Draw Axes (Optional: Red/Green/Blue lines to show rotation)
        center_pt = project_points(np.array([[0,0,0]]), pred_rot, corrected_trans, K)[0]
        # X-axis (Red)
        x_pt = project_points(np.array([[0.1,0,0]]), pred_rot, corrected_trans, K)[0]
        cv2.line(viz_img, tuple(center_pt), tuple(x_pt), (255,0,0), 3)

    # YOLO Box (Red)
    cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Obj {obj_id_str}: Corrected Inference\nGreen = Truth | Yellow = Pred", fontsize=14)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize()

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '...'))
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

def load_mesh_corners(obj_id_str):
    """
    Loads the mesh and calculates the EXACT 3D Bounding Box corners.
    Does not assume the object is centered at 0,0,0.
    """
    ply_name = f"obj_{obj_id_str}.ply"
    path = os.path.join(MESH_DIR, ply_name)
    
    if not os.path.exists(path):
        print(f"⚠️ Mesh {ply_name} not found. Using generic box.")
        return np.array([[-0.05]*3, [0.05]*3]), None # Fallback

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
    
    # 2. Filter Outliers (Clean Visualization)
    distances = np.linalg.norm(verts, axis=1)
    valid_mask = distances < 0.5 
    verts_clean = verts[valid_mask]
    if len(verts_clean) == 0: verts_clean = verts

    # 3. Calculate Exact Min/Max (The Fix)
    min_pt = np.min(verts_clean, axis=0)
    max_pt = np.max(verts_clean, axis=0)
    
    # Create the 8 corners from these min/max values
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
    
    return corners, verts_clean

def get_ground_truth_and_K(obj_id_str, frame_id):
    # Load GT
    gt_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "gt.yml")
    info_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "info.yml")
    
    r_mat, t, K = None, None, None
    
    # Load K (Camera Matrix)
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            infos = yaml.safe_load(f)
            # Use specific frame K if available, else first frame
            if frame_id in infos: k_list = infos[frame_id]['cam_K']
            else: k_list = infos[list(infos.keys())[0]]['cam_K']
            K = np.array(k_list).reshape(3, 3)
    else:
        # Default LineMOD K
        K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])

    # Load Pose
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            gts = yaml.safe_load(f)
        if frame_id in gts:
            for anno in gts[frame_id]:
                if str(int(anno['obj_id'])).zfill(2) == obj_id_str:
                    t = np.array(anno['cam_t_m2c']) / 1000.0 # mm -> m
                    r_mat = np.array(anno['cam_R_m2c']).reshape(3, 3)
                    break
    
    return r_mat, t, K

def project_points(points_3d, r_vec, t_vec, K):
    if r_vec.shape == (4,): r_mat = R.from_quat(r_vec).as_matrix()
    else: r_mat = r_vec
    
    p_cam = (r_mat @ points_3d.T).T + t_vec
    
    # Avoid division by zero
    z = p_cam[:, 2]
    z[z == 0] = 1e-5
    
    p_2d = np.zeros((points_3d.shape[0], 2))
    p_2d[:, 0] = (p_cam[:, 0] * K[0, 0] / z) + K[0, 2]
    p_2d[:, 1] = (p_cam[:, 1] * K[1, 1] / z) + K[1, 2]
    
    return p_2d.astype(int)

def draw_3d_box(img, points_2d, color=(0, 255, 0)):
    # Standard cube connections
    # 0-3 (Base), 4-7 (Top)
    lines = [(0,1), (1,2), (2,3), (3,0), 
             (4,5), (5,6), (6,7), (7,4), 
             (0,4), (1,5), (2,6), (3,7)]
             
    for s, e in lines:
        pt1 = tuple(points_2d[s])
        pt2 = tuple(points_2d[e])
        # Simple clipping check
        if 0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0]:
            img = cv2.line(img, pt1, pt2, color, 2)
    return img

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Visualization V4 started on {device}...")
    
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
    if not img_files: print("❌ No images found."); return
    
    img_name = np.random.choice(img_files)
    print(f"👀 Analyzing: {img_name}")
    
    # Parse Filename
    try:
        parts = img_name.split('_')
        obj_id_str = parts[0]
        frame_id = int(parts[1].split('.')[0])
    except:
        obj_id_str, frame_id = "01", 0
    
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    original_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # YOLO
    results = yolo(img_path, verbose=False)
    if len(results[0].boxes) == 0: print("❌ YOLO failed."); return
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Crop
    w, h = x2-x1, y2-y1
    size = max(w, h) * 1.2
    c_x, c_y = x1 + w/2, y1 + h/2
    new_x, new_y = int(c_x - size/2), int(c_y - size/2)
    new_size = int(size)
    
    # Pad & Crop
    h_img, w_img, _ = original_img.shape
    pad_l = max(0, -new_x); pad_t = max(0, -new_y)
    pad_r = max(0, (new_x + new_size) - w_img)
    pad_b = max(0, (new_y + new_size) - h_img)
    padded_img = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    crop = padded_img[new_y+pad_t:new_y+pad_t+new_size, new_x+pad_l:new_x+pad_l+new_size]
    crop_resized = cv2.resize(crop, (224, 224))
    
    # PoseNet Inference
    input_tensor = transform(crop_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_rot, pred_trans = pose_net(input_tensor)
    pred_rot = pred_rot.cpu().numpy()[0]
    pred_trans = pred_trans.cpu().numpy()[0]

    # === LOAD TRUE MESH CORNERS ===
    corners_3d, verts = load_mesh_corners(obj_id_str)
    
    # Load GT and K
    gt_r, gt_t, K = get_ground_truth_and_K(obj_id_str, frame_id)
    
    viz_img = original_img.copy()
    
    if gt_r is not None:
        print(f"   ✅ GT Found: T={gt_t}")
        box_2d_gt = project_points(corners_3d, gt_r, gt_t, K)
        viz_img = draw_3d_box(viz_img, box_2d_gt, color=(0, 255, 0)) # GREEN
    else:
        print("   ❌ GT Missing.")

    # Draw Pred
    print(f"   ✅ Pred: T={pred_trans}")
    box_2d_pred = project_points(corners_3d, pred_rot, pred_trans, K)
    viz_img = draw_3d_box(viz_img, box_2d_pred, color=(0, 255, 255)) # YELLOW
    
    # Draw YOLO
    cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Obj {obj_id_str} | Green=True Box, Yellow=Pred Box")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize()

import os
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

def load_mesh_points(obj_id_str):
    """Loads mesh points, converts to meters, samples 500 points."""
    ply_name = f"obj_{obj_id_str}.ply"
    path = os.path.join(MESH_DIR, ply_name)
    
    if not os.path.exists(path):
        print(f"⚠️ Mesh {ply_name} not found.")
        return None

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
    
    # 3. Downsample for speed/visibility (Take every 10th point)
    # or take fixed 500
    if len(verts_clean) > 500:
        idx = np.random.choice(len(verts_clean), 500, replace=False)
        verts_final = verts_clean[idx]
    else:
        verts_final = verts_clean
        
    return verts_final

def get_gt_and_K(obj_id_str, frame_id):
    gt_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "gt.yml")
    info_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "info.yml")
    
    r_mat, t, K = None, None, None
    
    # K
    if os.path.exists(info_path):
        with open(info_path, 'r') as f: infos = yaml.safe_load(f)
        k_list = infos[frame_id]['cam_K'] if frame_id in infos else infos[list(infos.keys())[0]]['cam_K']
        K = np.array(k_list).reshape(3, 3)
    else:
        K = np.array([[572.4, 0, 325.2], [0, 573.5, 242.0], [0, 0, 1]])

    # GT
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
    # Safe division
    z = p_cam[:, 2].copy()
    z[z==0] = 0.001
    
    p_2d = np.zeros((points_3d.shape[0], 2))
    p_2d[:, 0] = (p_cam[:, 0] * K[0, 0] / z) + K[0, 2]
    p_2d[:, 1] = (p_cam[:, 1] * K[1, 1] / z) + K[1, 2]
    return p_2d.astype(int)

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Visualization (Point Cloud Mode)...")
    
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

    # Pick Random Image
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
    
    # YOLO & Crop
    results = yolo(img_path, verbose=False)
    if not results[0].boxes: print("No YOLO box"); return
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    w, h = x2-x1, y2-y1
    size = max(w, h) * 1.2
    c_x, c_y = x1 + w/2, y1 + h/2
    new_x, new_y = int(c_x - size/2), int(c_y - size/2)
    new_size = int(size)
    
    # Pad Crop
    h_img, w_img, _ = original_img.shape
    pad_l = max(0, -new_x); pad_t = max(0, -new_y)
    pad_r = max(0, (new_x + new_size) - w_img)
    pad_b = max(0, (new_y + new_size) - h_img)
    padded_img = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    crop = padded_img[new_y+pad_t:new_y+pad_t+new_size, new_x+pad_l:new_x+pad_l+new_size]
    crop_resized = cv2.resize(crop, (224, 224))
    
    # PoseNet
    input_tensor = transform(crop_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_rot, pred_trans = pose_net(input_tensor)
    pred_rot = pred_rot.cpu().numpy()[0]
    pred_trans = pred_trans.cpu().numpy()[0]

    # Visualize Points
    verts = load_mesh_points(obj_id_str)
    viz_img = original_img.copy()
    
    if verts is not None:
        gt_r, gt_t, K = get_gt_and_K(obj_id_str, frame_id)
        
        # 1. GT Points (Green Dots)
        if gt_r is not None:
            pts_2d_gt = project_points(verts, gt_r, gt_t, K)
            for x, y in pts_2d_gt:
                if 0<=x<w_img and 0<=y<h_img:
                    cv2.circle(viz_img, (x, y), 2, (0, 255, 0), -1)

        # 2. Pred Points (Yellow Dots)
        pts_2d_pred = project_points(verts, pred_rot, pred_trans, K)
        for x, y in pts_2d_pred:
             if 0<=x<w_img and 0<=y<h_img:
                cv2.circle(viz_img, (x, y), 2, (255, 255, 0), -1)

    # YOLO Box
    cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title("Green Dots = True Shape | Yellow Dots = Pred Shape", fontsize=14)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize()
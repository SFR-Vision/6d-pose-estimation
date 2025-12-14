import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import sys

# Custom Modules
try:
    from models.pose_net_rgbd import PoseNetRGBD
except ImportError:
    print("❌ Error: Could not import PoseNetRGBD.")
    sys.exit(1)

# ================= CONFIGURATION =================
YOLO_PATH = os.path.join("runs", "detect", "linemod_yolo", "weights", "best.pt")
POSE_PATH = os.path.join("weights_rgbd", "best_pose_model_rgbd.pth")
MESH_DIR = os.path.join("datasets", "Linemod_preprocessed", "models")

# Standard Camera Matrix (calibrate for your camera!)
K = np.array([[572.4114, 0.0, 325.2611],
              [0.0, 573.57043, 242.04899],
              [0.0, 0.0, 1.0]])

# Map YOLO Class Indices to LineMOD Object IDs
CLASS_ID_TO_OBJ_NAME = {
    0: "01", 1: "02", 2: "04", 3: "05", 4: "06", 5: "08",
    6: "09", 7: "10", 8: "11", 9: "12", 10: "13", 11: "14", 12: "15",
}
# =================================================

def load_mesh_corners(obj_id_str):
    """Loads mesh and returns tight bounding box corners"""
    ply_name = f"obj_{obj_id_str}.ply"
    path = os.path.join(MESH_DIR, ply_name)
    
    if not os.path.exists(path):
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
    
    # Unit conversion and outlier removal
    verts = np.array(verts) / 1000.0
    distances = np.linalg.norm(verts, axis=1)
    verts_clean = verts[distances < 0.3]
    if len(verts_clean) == 0: verts_clean = verts

    # Statistical Box
    min_pt = np.percentile(verts_clean, 1, axis=0)
    max_pt = np.percentile(verts_clean, 99, axis=0)
    
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]]
    ])
    return corners

def project_points(points_3d, r_vec, t_vec, K):
    """Project 3D points to 2D"""
    if r_vec.shape == (4,): r_mat = R.from_quat(r_vec).as_matrix()
    else: r_mat = r_vec
    
    p_cam = (r_mat @ points_3d.T).T + t_vec
    z = p_cam[:, 2].copy()
    z[z==0] = 0.001
    
    p_2d = np.zeros((points_3d.shape[0], 2))
    p_2d[:, 0] = (p_cam[:, 0] * K[0, 0] / z) + K[0, 2]
    p_2d[:, 1] = (p_cam[:, 1] * K[1, 1] / z) + K[1, 2]
    return p_2d.astype(int)

def draw_box(img, pts_2d, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box"""
    lines = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for s, e in lines:
        cv2.line(img, tuple(pts_2d[s]), tuple(pts_2d[e]), color, thickness)
    return img

def run_inference(rgb_path, depth_path=None):
    """
    Run RGB-D pose inference on an image
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image (optional, will use dummy depth if None)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Processing: {rgb_path}")
    
    # 1. Load Models
    yolo = YOLO(YOLO_PATH)
    pose_net = PoseNetRGBD(pretrained=False).to(device)
    ckpt = torch.load(POSE_PATH, map_location=device)
    pose_net.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    pose_net.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Read RGB Image
    original_img = cv2.imread(rgb_path)
    if original_img is None: 
        print("❌ RGB image not found.")
        return
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = rgb_img.shape
    
    # 3. Read Depth Image (or create dummy)
    if depth_path and os.path.exists(depth_path):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        print("   ✅ Using provided depth image")
    else:
        # Create dummy depth (uniform at 0.5m)
        depth_img = np.ones((h_img, w_img), dtype=np.uint16) * 500  # 500mm
        print("   ⚠️ No depth provided, using dummy depth (0.5m)")
    
    viz_img = original_img.copy()

    # 4. Detect Objects
    results = yolo(rgb_path, verbose=False)
    if not results[0].boxes:
        print("   ⚠️ No objects detected.")
        return

    print(f"   Found {len(results[0].boxes)} objects.")

    # 5. Loop through detections
    for box in results[0].boxes:
        # A. YOLO Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # B. Get Mesh ID
        obj_id_str = CLASS_ID_TO_OBJ_NAME.get(cls_id, "01")
        
        # C. Crop RGB & Depth
        c_x_box, c_y_box = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2-x1, y2-y1
        size = max(w, h) * 1.2
        new_x, new_y = int(c_x_box - size/2), int(c_y_box - size/2)
        new_size = int(size)
        
        pad_l = max(0, -new_x); pad_t = max(0, -new_y)
        pad_r = max(0, (new_x + new_size) - w_img)
        pad_b = max(0, (new_y + new_size) - h_img)
        
        padded_rgb = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        padded_depth = cv2.copyMakeBorder(depth_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        
        crop_rgb = padded_rgb[new_y+pad_t:new_y+pad_t+new_size, new_x+pad_l:new_x+pad_l+new_size]
        crop_depth = padded_depth[new_y+pad_t:new_y+pad_t+new_size, new_x+pad_l:new_x+pad_l+new_size]
        
        crop_rgb_resized = cv2.resize(crop_rgb, (224, 224))
        crop_depth_resized = cv2.resize(crop_depth, (224, 224))
        
        # D. Normalize Depth
        crop_depth_normalized = crop_depth_resized.astype(np.float32) / 1000.0
        crop_depth_normalized = np.clip(crop_depth_normalized / 2.0, 0, 1)
        crop_depth_normalized = crop_depth_normalized[..., np.newaxis]
        
        # E. Pose Inference
        input_rgb = transform(crop_rgb_resized).unsqueeze(0).to(device)
        input_depth = torch.from_numpy(crop_depth_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            pred_rot, pred_trans = pose_net(input_rgb, input_depth)
        pred_rot = pred_rot.cpu().numpy()[0]
        pred_trans = pred_trans.cpu().numpy()[0]
        
        # F. Geometric Correction
        pred_z = pred_trans[2]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        pred_x = (c_x_box - cx) * pred_z / fx
        pred_y = (c_y_box - cy) * pred_z / fy
        corrected_trans = np.array([pred_x, pred_y, pred_z])

        # G. Visualization
        corners_3d = load_mesh_corners(obj_id_str)
        if corners_3d is not None:
            # Draw 3D Box
            box_2d = project_points(corners_3d, pred_rot, corrected_trans, K)
            draw_box(viz_img, box_2d, color=(0, 255, 0), thickness=2)
            
            # Draw Axes
            origin = project_points(np.array([[0,0,0]]), pred_rot, corrected_trans, K)[0]
            axis_x = project_points(np.array([[0.1,0,0]]), pred_rot, corrected_trans, K)[0]
            axis_y = project_points(np.array([[0,0.1,0]]), pred_rot, corrected_trans, K)[0]
            axis_z = project_points(np.array([[0,0,0.1]]), pred_rot, corrected_trans, K)[0]
            
            cv2.line(viz_img, tuple(origin), tuple(axis_x), (0,0,255), 2)  # X Red
            cv2.line(viz_img, tuple(origin), tuple(axis_y), (255,0,0), 2)  # Y Blue
            cv2.line(viz_img, tuple(origin), tuple(axis_z), (0,255,0), 2)  # Z Green
            
            # Label
            cv2.putText(viz_img, f"{obj_id_str} ({conf:.2f})", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"      Obj {obj_id_str}: Pose = {corrected_trans}, Conf = {conf:.2f}")

    # 6. Show Result
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title("RGB-D 6D Pose Inference", fontsize=16)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Auto-pick test image or use command line argument
    if len(sys.argv) > 1:
        RGB_IMG = sys.argv[1]
        DEPTH_IMG = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Auto-select from test directory
        test_dir = os.path.join("datasets", "yolo_ready", "images", "test")
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.jpg')]
            if len(files) > 0:
                random_file = np.random.choice(files)
                RGB_IMG = os.path.join(test_dir, random_file)
                
                # Try to find corresponding depth
                obj_id = random_file.split('_')[0]
                frame_id = random_file.split('_')[1].split('.')[0]
                DEPTH_IMG = os.path.join("datasets", "Linemod_preprocessed", "data", 
                                        obj_id, "depth", f"{frame_id}.png")
                
                print(f"🎲 Auto-selected: {random_file}")
            else:
                print(f"❌ No images found in {test_dir}")
                sys.exit(1)
        else:
            print(f"❌ Directory not found: {test_dir}")
            sys.exit(1)
        
    run_inference(RGB_IMG, DEPTH_IMG)

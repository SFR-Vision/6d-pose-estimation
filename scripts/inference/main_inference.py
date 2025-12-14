import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# Custom Modules
try:
    from models.pose_net import PoseNet
except ImportError:
    print("❌ Error: Could not import PoseNet.")
    sys.exit(1)

# ================= CONFIGURATION =================
YOLO_PATH = os.path.join("runs", "detect", "linemod_yolo", "weights", "best.pt")
POSE_PATH = os.path.join("weights", "best_pose_model.pth")
MESH_DIR = os.path.join("datasets", "Linemod_preprocessed", "models")

# Standard Camera Matrix (approximate for general use)
# If using a webcam, you should calibrate it and replace this!
K = np.array([[572.4114, 0.0, 325.2611],
              [0.0, 573.57043, 242.04899],
              [0.0, 0.0, 1.0]])

# Map YOLO Class Indices to LineMOD Object IDs (string names for .ply files)
# Update this list based on your specific class names/order in data.yaml!
# Assuming standard order: 0->01 (Ape), 1->02 (Benchvise), etc.
CLASS_ID_TO_OBJ_NAME = {
    0: "01", # Ape
    1: "02", # Benchvise
    2: "04", # Cam (Note: ID 03 is usually skipped in LineMOD)
    3: "05", # Can
    4: "06", # Cat
    5: "08", # Drill
    6: "09", # Duck
    7: "10", # Eggbox
    8: "11", # Glue
    9: "12", # Holepuncher
    10: "13", # Iron
    11: "14", # Lamp
    12: "15", # Phone
}
# =================================================

def load_mesh_corners(obj_id_str):
    """Loads mesh and returns the TIGHT bounding box corners."""
    ply_name = f"obj_{obj_id_str}.ply"
    path = os.path.join(MESH_DIR, ply_name)
    
    # Cache checking could be added here for speed
    if not os.path.exists(path):
        return None # Mesh missing

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

    # Statistical Box (1st to 99th percentile)
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
    lines = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for s, e in lines:
        cv2.line(img, tuple(pts_2d[s]), tuple(pts_2d[e]), color, thickness)
    return img

def run_inference(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Processing: {img_path}")
    
    # 1. Load Models
    yolo = YOLO(YOLO_PATH)
    pose_net = PoseNet(pretrained=False).to(device)
    ckpt = torch.load(POSE_PATH, map_location=device)
    pose_net.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    pose_net.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Read Image
    original_img = cv2.imread(img_path)
    if original_img is None: print("❌ Image not found."); return
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = rgb_img.shape
    
    viz_img = original_img.copy()

    # 3. Detect Objects
    results = yolo(img_path, verbose=False)
    if not results[0].boxes:
        print("   ⚠️ No objects detected.")
        return

    print(f"   found {len(results[0].boxes)} objects.")

    # 4. Loop through every object found
    for box in results[0].boxes:
        # A. YOLO Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # B. Get Mesh ID from Class ID
        obj_id_str = CLASS_ID_TO_OBJ_NAME.get(cls_id, "01") # Default to 01 if unknown
        
        # C. Crop & Prepare
        c_x_box, c_y_box = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2-x1, y2-y1
        size = max(w, h) * 1.2
        new_x, new_y = int(c_x_box - size/2), int(c_y_box - size/2)
        new_size = int(size)
        
        pad_l = max(0, -new_x); pad_t = max(0, -new_y)
        pad_r = max(0, (new_x + new_size) - w_img)
        pad_b = max(0, (new_y + new_size) - h_img)
        padded_img = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        crop = padded_img[new_y+pad_t:new_y+pad_t+new_size, new_x+pad_l:new_x+pad_l+new_size]
        crop_resized = cv2.resize(crop, (224, 224))
        
        # D. Pose Inference
        input_tensor = transform(crop_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_rot, pred_trans = pose_net(input_tensor)
        pred_rot = pred_rot.cpu().numpy()[0]
        pred_trans = pred_trans.cpu().numpy()[0]
        
        # E. Geometric Correction (Back-Projection)
        # Recalculate X/Y based on YOLO Center + Pred Z
        pred_z = pred_trans[2]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        pred_x = (c_x_box - cx) * pred_z / fx
        pred_y = (c_y_box - cy) * pred_z / fy
        corrected_trans = np.array([pred_x, pred_y, pred_z])

        # F. Visualization
        corners_3d = load_mesh_corners(obj_id_str)
        if corners_3d is not None:
            # Draw 3D Box
            box_2d = project_points(corners_3d, pred_rot, corrected_trans, K)
            draw_box(viz_img, box_2d, color=(0, 255, 0), thickness=2)
            
            # Draw Axes (X=Red, Y=Blue, Z=Green)
            origin = project_points(np.array([[0,0,0]]), pred_rot, corrected_trans, K)[0]
            axis_x = project_points(np.array([[0.1,0,0]]), pred_rot, corrected_trans, K)[0]
            axis_y = project_points(np.array([[0,0.1,0]]), pred_rot, corrected_trans, K)[0]
            axis_z = project_points(np.array([[0,0,0.1]]), pred_rot, corrected_trans, K)[0]
            
            cv2.line(viz_img, tuple(origin), tuple(axis_x), (0,0,255), 2) # X Red
            cv2.line(viz_img, tuple(origin), tuple(axis_y), (255,0,0), 2) # Y Blue
            cv2.line(viz_img, tuple(origin), tuple(axis_z), (0,255,0), 2) # Z Green
            
            # Label
            cv2.putText(viz_img, f"{obj_id_str} ({conf:.2f})", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Show Final Result
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
    plt.title("6D Pose Inference", fontsize=16)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # 1. Define the test directory
    test_dir = os.path.join("datasets", "yolo_ready", "images", "test")
    
    # 2. Check if user provided a specific image path via command line
    if len(sys.argv) > 1:
        TEST_IMG = sys.argv[1]
    else:
        # 3. If no argument, AUTO-PICK the first available image
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.jpg')]
            if len(files) > 0:
                # Pick a random one for variety every time you run it
                random_file = np.random.choice(files)
                TEST_IMG = os.path.join(test_dir, random_file)
                print(f"🎲 Auto-selected image: {random_file}")
            else:
                print(f"❌ No images found in {test_dir}")
                sys.exit(1)
        else:
            print(f"❌ Directory not found: {test_dir}")
            sys.exit(1)
        
    run_inference(TEST_IMG)
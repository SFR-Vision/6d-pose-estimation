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

from models.pose_net import PoseNet
from models.pose_net_rgbd import PoseNetRGBD

# ================= CONFIGURATION =================
YOLO_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "linemod_yolo", "weights", "best.pt")
RGB_POSE_PATH = os.path.join(PROJECT_ROOT, "weights", "best_pose_model.pth")
RGBD_POSE_PATH = os.path.join(PROJECT_ROOT, "weights_rgbd", "best_pose_model_rgbd.pth")
TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "yolo_ready", "images", "test")
ORIG_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "comparison_results")
# =================================================

# Object ID to Name Mapping
OBJECT_NAMES = {
    1: "Ape", 2: "Benchvise", 4: "Camera", 5: "Can", 6: "Cat",
    8: "Driller", 9: "Duck", 10: "Eggbox", 11: "Glue", 12: "Holepuncher",
    13: "Iron", 14: "Lamp", 15: "Phone"
}

def load_mesh_corners(obj_id_str):
    """Load 3D bounding box corners from mesh using percentile-based filtering"""
    ply_name = f"obj_{obj_id_str}.ply"
    path = os.path.join(MESH_DIR, ply_name)
    
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
    
    # Filter outliers (more aggressive than before)
    distances = np.linalg.norm(verts, axis=1)
    valid_mask = distances < 0.3  # Changed from 0.5 to 0.3 meters
    verts_clean = verts[valid_mask] if len(verts[valid_mask]) > 0 else verts
    
    # Use percentiles instead of min/max to avoid noise
    min_pt = np.percentile(verts_clean, 1, axis=0)
    max_pt = np.percentile(verts_clean, 99, axis=0)
    
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

def get_ground_truth_and_K(obj_id_str, frame_id):
    """Load ground truth pose and camera matrix"""
    gt_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "gt.yml")
    info_path = os.path.join(ORIG_DATA_DIR, obj_id_str, "info.yml")
    
    r_mat, t, K = None, None, None
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            infos = yaml.safe_load(f)
            if frame_id in infos:
                k_list = infos[frame_id]['cam_K']
            else:
                k_list = infos[list(infos.keys())[0]]['cam_K']
            K = np.array(k_list).reshape(3, 3)
    else:
        K = np.array([[572.4114, 0.0, 325.2611], 
                      [0.0, 573.57043, 242.04899], 
                      [0.0, 0.0, 1.0]])
    
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            gts = yaml.safe_load(f)
        if frame_id in gts:
            for anno in gts[frame_id]:
                if str(int(anno['obj_id'])).zfill(2) == obj_id_str:
                    t = np.array(anno['cam_t_m2c']) / 1000.0
                    r_mat = np.array(anno['cam_R_m2c']).reshape(3, 3)
                    break
    
    return r_mat, t, K

def project_points(points_3d, r_vec, t_vec, K):
    """Project 3D points to 2D image plane"""
    if r_vec.shape == (4,):
        r_mat = R.from_quat(r_vec).as_matrix()
    else:
        r_mat = r_vec
    
    p_cam = (r_mat @ points_3d.T).T + t_vec
    z = p_cam[:, 2]
    z[z == 0] = 1e-5
    
    p_2d = np.zeros((points_3d.shape[0], 2))
    p_2d[:, 0] = (p_cam[:, 0] * K[0, 0] / z) + K[0, 2]
    p_2d[:, 1] = (p_cam[:, 1] * K[1, 1] / z) + K[1, 2]
    
    return p_2d.astype(int)

def draw_3d_box(img, points_2d, color, thickness=2, style='solid'):
    """Draw 3D bounding box with optional dashed lines"""
    lines = [(0,1), (1,2), (2,3), (3,0), 
             (4,5), (5,6), (6,7), (7,4), 
             (0,4), (1,5), (2,6), (3,7)]
    
    for s, e in lines:
        pt1 = tuple(points_2d[s])
        pt2 = tuple(points_2d[e])
        
        if style == 'dashed':
            draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=8)
        else:
            cv2.line(img, pt1, pt2, color, thickness)
    
    return img

def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=10):
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

def calculate_add_error(pred_rot, pred_trans, gt_rot, gt_trans, corners_3d):
    """Calculate ADD metric (Average Distance of Model Points)"""
    if pred_rot.shape == (4,):
        pred_r_mat = R.from_quat(pred_rot).as_matrix()
    else:
        pred_r_mat = pred_rot
    
    gt_points = (gt_rot @ corners_3d.T).T + gt_trans
    pred_points = (pred_r_mat @ corners_3d.T).T + pred_trans
    
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return np.mean(distances) * 1000  # Convert to mm

def process_image(yolo, rgb_model, rgbd_model, img_path, device, transform):
    """Process single image and return predictions"""
    # Parse filename
    img_name = os.path.basename(img_path)
    try:
        parts = img_name.split('_')
        obj_id_str = parts[0]
        frame_id = int(parts[1].split('.')[0])
    except:
        return None
    
    # Load images
    original_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Construct correct depth path: datasets/Linemod_preprocessed/data/{obj_id}/depth/{frame_id}.png
    depth_path = os.path.join(ORIG_DATA_DIR, obj_id_str, 'depth', f"{frame_id:04d}.png")
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return None
    
    # YOLO detection
    results = yolo(img_path, verbose=False)
    if len(results[0].boxes) == 0:
        return None
    
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Crop and prepare input
    w, h = x2-x1, y2-y1
    size = max(w, h) * 1.2
    c_x, c_y = x1 + w/2, y1 + h/2
    new_x, new_y = int(c_x - size/2), int(c_y - size/2)
    new_size = int(size)
    
    h_img, w_img = original_img.shape[:2]
    pad_l = max(0, -new_x)
    pad_t = max(0, -new_y)
    pad_r = max(0, (new_x + new_size) - w_img)
    pad_b = max(0, (new_y + new_size) - h_img)
    
    padded_rgb = cv2.copyMakeBorder(rgb_img, pad_t, pad_b, pad_l, pad_r, 
                                     cv2.BORDER_CONSTANT, value=0)
    padded_depth = cv2.copyMakeBorder(depth_img, pad_t, pad_b, pad_l, pad_r,
                                       cv2.BORDER_CONSTANT, value=0)
    
    rgb_crop = padded_rgb[new_y+pad_t:new_y+pad_t+new_size, 
                          new_x+pad_l:new_x+pad_l+new_size]
    depth_crop = padded_depth[new_y+pad_t:new_y+pad_t+new_size,
                              new_x+pad_l:new_x+pad_l+new_size]
    
    rgb_crop = cv2.resize(rgb_crop, (224, 224))
    depth_crop = cv2.resize(depth_crop, (224, 224))
    
    # Process depth
    depth_crop = depth_crop.astype(np.float32)
    if depth_crop.max() > 0:
        depth_crop = cv2.bilateralFilter(depth_crop, 5, 75, 75)
    depth_crop = depth_crop / 1000.0
    depth_crop = np.clip(depth_crop / 1.5, 0, 1)
    depth_crop = depth_crop[..., np.newaxis]
    
    # Prepare tensors
    rgb_tensor = transform(rgb_crop).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(depth_crop).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Predictions
    with torch.no_grad():
        rgb_rot, rgb_trans = rgb_model(rgb_tensor)
        rgbd_rot, rgbd_trans = rgbd_model(rgb_tensor, depth_tensor)
    
    rgb_rot = rgb_rot.cpu().numpy()[0]
    rgb_trans = rgb_trans.cpu().numpy()[0]
    rgbd_rot = rgbd_rot.cpu().numpy()[0]
    rgbd_trans = rgbd_trans.cpu().numpy()[0]
    
    # Ground truth
    gt_r, gt_t, K = get_ground_truth_and_K(obj_id_str, frame_id)
    if gt_r is None:
        return None
    
    corners_3d = load_mesh_corners(obj_id_str)
    
    # Calculate errors
    rgb_error = calculate_add_error(rgb_rot, rgb_trans, gt_r, gt_t, corners_3d)
    rgbd_error = calculate_add_error(rgbd_rot, rgbd_trans, gt_r, gt_t, corners_3d)
    
    return {
        'img': original_img.copy(),
        'obj_id': int(obj_id_str),
        'frame_id': frame_id,
        'gt_rot': gt_r,
        'gt_trans': gt_t,
        'rgb_rot': rgb_rot,
        'rgb_trans': rgb_trans,
        'rgbd_rot': rgbd_rot,
        'rgbd_trans': rgbd_trans,
        'corners_3d': corners_3d,
        'K': K,
        'rgb_error': rgb_error,
        'rgbd_error': rgbd_error
    }

def visualize_comparison(data, output_path):
    """Create comparison visualization"""
    img = data['img']
    obj_name = OBJECT_NAMES.get(data['obj_id'], f"Object {data['obj_id']}")
    
    # Create three images: GT, RGB, RGB-D
    img_gt = img.copy()
    img_rgb = img.copy()
    img_rgbd = img.copy()
    
    # Project and draw boxes
    gt_box = project_points(data['corners_3d'], data['gt_rot'], data['gt_trans'], data['K'])
    rgb_box = project_points(data['corners_3d'], data['rgb_rot'], data['rgb_trans'], data['K'])
    rgbd_box = project_points(data['corners_3d'], data['rgbd_rot'], data['rgbd_trans'], data['K'])
    
    draw_3d_box(img_gt, gt_box, (0, 255, 0), thickness=3)  # Green
    draw_3d_box(img_rgb, rgb_box, (255, 255, 0), thickness=3)  # Yellow
    draw_3d_box(img_rgb, gt_box, (0, 255, 0), thickness=2, style='dashed')  # Green dashed
    draw_3d_box(img_rgbd, rgbd_box, (255, 165, 0), thickness=3)  # Orange
    draw_3d_box(img_rgbd, gt_box, (0, 255, 0), thickness=2, style='dashed')  # Green dashed
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ground Truth
    axes[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold', color='green')
    axes[0].axis('off')
    
    # RGB-only
    axes[1].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'RGB-only Model\nADD Error: {data["rgb_error"]:.2f} mm', 
                      fontsize=14, fontweight='bold', color='#FFD700')
    axes[1].axis('off')
    
    # RGB-D
    axes[2].imshow(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))
    improvement = ((data['rgb_error'] - data['rgbd_error']) / data['rgb_error']) * 100
    axes[2].set_title(f'RGB-D Model\nADD Error: {data["rgbd_error"]:.2f} mm ({improvement:+.1f}%)', 
                      fontsize=14, fontweight='bold', color='#FFA500')
    axes[2].axis('off')
    
    # Overall title
    fig.suptitle(f'{obj_name} (ID: {data["obj_id"]:02d}) - Frame {data["frame_id"]}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 RGB vs RGB-D Comparison started on {device}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load models
    print("📦 Loading models...")
    yolo = YOLO(YOLO_PATH)
    
    rgb_model = PoseNet(pretrained=False).to(device)
    rgb_ckpt = torch.load(RGB_POSE_PATH, map_location=device)
    if 'model_state_dict' in rgb_ckpt:
        rgb_model.load_state_dict(rgb_ckpt['model_state_dict'])
    else:
        rgb_model.load_state_dict(rgb_ckpt)
    rgb_model.eval()
    
    rgbd_model = PoseNetRGBD(pretrained=False).to(device)
    rgbd_ckpt = torch.load(RGBD_POSE_PATH, map_location=device)
    if 'model_state_dict' in rgbd_ckpt:
        rgbd_model.load_state_dict(rgbd_ckpt['model_state_dict'])
    else:
        rgbd_model.load_state_dict(rgbd_ckpt)
    rgbd_model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process test images - one random image per object class
    print("🖼️  Processing test images (one per object class)...")
    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".png")]
    
    # Group images by object ID
    obj_images = {}
    for img_file in img_files:
        try:
            obj_id = img_file.split('_')[0]
            if obj_id not in obj_images:
                obj_images[obj_id] = []
            obj_images[obj_id].append(img_file)
        except:
            continue
    
    # Select one random image per object
    selected_images = []
    for obj_id in sorted(obj_images.keys()):
        random_img = np.random.choice(obj_images[obj_id])
        selected_images.append(random_img)
    
    print(f"   Selected {len(selected_images)} images (one per object class)")
    
    rgb_errors = []
    rgbd_errors = []
    processed = 0
    
    for img_file in selected_images:
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        data = process_image(yolo, rgb_model, rgbd_model, img_path, device, transform)
        
        if data is None:
            print(f"   ⚠️ {img_file}: Failed to process")
            continue
        
        # Save visualization
        output_path = os.path.join(OUTPUT_DIR, f"comparison_{img_file}")
        visualize_comparison(data, output_path)
        
        rgb_errors.append(data['rgb_error'])
        rgbd_errors.append(data['rgbd_error'])
        processed += 1
        
        obj_name = OBJECT_NAMES.get(data['obj_id'], f"Object {data['obj_id']}")
        print(f"   ✅ {obj_name}: RGB={data['rgb_error']:.2f}mm, RGB-D={data['rgbd_error']:.2f}mm")
    
    # Summary statistics
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Processed: {processed} images")
    print(f"\nRGB-only Model:")
    print(f"   Mean ADD Error: {np.mean(rgb_errors):.2f} mm")
    print(f"   Std Dev: {np.std(rgb_errors):.2f} mm")
    print(f"   Min/Max: {np.min(rgb_errors):.2f} / {np.max(rgb_errors):.2f} mm")
    print(f"\nRGB-D Model:")
    print(f"   Mean ADD Error: {np.mean(rgbd_errors):.2f} mm")
    print(f"   Std Dev: {np.std(rgbd_errors):.2f} mm")
    print(f"   Min/Max: {np.min(rgbd_errors):.2f} / {np.max(rgbd_errors):.2f} mm")
    print(f"\nImprovement:")
    improvement = ((np.mean(rgb_errors) - np.mean(rgbd_errors)) / np.mean(rgb_errors)) * 100
    print(f"   {improvement:.1f}% reduction in ADD error")
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()

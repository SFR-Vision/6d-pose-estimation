"""Visual comparison of all 4 pose estimation models."""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from torchvision import transforms

from utils.mesh_utils import load_mesh_corners
from utils.visualization import project_points, draw_3d_box

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")

WEIGHTS = {
    'RGB': os.path.join(PROJECT_ROOT, "weights_rgb", "best_pose_model.pth"),
    'RGB-Geo': os.path.join(PROJECT_ROOT, "weights_rgb_geometric", "best_pose_model.pth"),
    'RGBD': os.path.join(PROJECT_ROOT, "weights_rgbd", "best_pose_model.pth"),
    'RGBD-Geo': os.path.join(PROJECT_ROOT, "weights_rgbd_geometric", "best_pose_model.pth"),
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_models():
    """Load all available models."""
    models = {}
    
    for name, path in WEIGHTS.items():
        if not os.path.exists(path):
            print(f"{name}: Weights not found")
            continue
            
        try:
            if name == 'RGB':
                from models.pose_net_rgb import PoseNetRGB
                model = PoseNetRGB(pretrained=False)
            elif name == 'RGB-Geo':
                from models.pose_net_rgb_geometric import PoseNetRGBGeometric
                model = PoseNetRGBGeometric(pretrained=False)
            elif name == 'RGBD':
                from models.pose_net_rgbd import PoseNetRGBD
                model = PoseNetRGBD(pretrained=False)
            elif name == 'RGBD-Geo':
                from models.pose_net_rgbd_geometric import PoseNetRGBDGeometric
                model = PoseNetRGBDGeometric(pretrained=False)
            
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE).eval()
            models[name] = model
            print(f"{name}: Loaded")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    return models


def get_sample_data(obj_id_str, frame_id):
    """Load sample image, depth, GT pose, and camera matrix."""
    obj_path = os.path.join(DATA_ROOT, obj_id_str)
    
    rgb_path = os.path.join(obj_path, "rgb", f"{frame_id:04d}.png")
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth_path = os.path.join(obj_path, "depth", f"{frame_id:04d}.png")
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint16)
    
    gt_path = os.path.join(obj_path, "gt.yml")
    info_path = os.path.join(obj_path, "info.yml")
    
    with open(gt_path, 'r') as f:
        gts = yaml.safe_load(f)
    with open(info_path, 'r') as f:
        infos = yaml.safe_load(f)
    
    gt_rot, gt_trans, K = None, None, None
    
    if frame_id in gts:
        for anno in gts[frame_id]:
            if str(int(anno['obj_id'])).zfill(2) == obj_id_str:
                gt_rot = np.array(anno['cam_R_m2c']).reshape(3, 3)
                gt_trans = np.array(anno['cam_t_m2c']) / 1000.0
                break
    
    if frame_id in infos:
        K = np.array(infos[frame_id]['cam_K']).reshape(3, 3)
    
    bbox = gts[frame_id][0]['obj_bb'] if frame_id in gts else [0, 0, 100, 100]
    
    return rgb, depth, gt_rot, gt_trans, K, bbox


def prepare_crop(rgb, depth, bbox, img_size=224):
    """Prepare cropped and resized inputs."""
    x, y, w, h = bbox
    cx, cy = x + w/2, y + h/2  # Original bbox center in full image
    size = max(w, h) * 1.2
    
    x1 = int(cx - size/2)
    y1 = int(cy - size/2)
    
    h_img, w_img = rgb.shape[:2]
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, (x1 + int(size)) - w_img)
    pad_b = max(0, (y1 + int(size)) - h_img)
    
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        rgb = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        depth = cv2.copyMakeBorder(depth, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        x1 += pad_l
        y1 += pad_t
    
    rgb_crop = rgb[y1:y1+int(size), x1:x1+int(size)]
    depth_crop = depth[y1:y1+int(size), x1:x1+int(size)]
    
    rgb_crop = cv2.resize(rgb_crop, (img_size, img_size))
    depth_crop = cv2.resize(depth_crop, (img_size, img_size))
    
    scale = img_size / size
    bbox_center_crop = np.array([img_size/2, img_size/2], dtype=np.float32)
    bbox_center_orig = np.array([cx, cy], dtype=np.float32)  # Keep original center
    
    fx, fy = 572.4, 573.5
    cx_orig, cy_orig = 325.2, 242.0
    cx_crop = (cx_orig + pad_l - x1) * scale
    cy_crop = (cy_orig + pad_t - y1) * scale
    K_crop = np.array([
        [fx * scale, 0, cx_crop],
        [0, fy * scale, cy_crop],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Return original K as well for geometric correction
    K_orig = np.array([
        [fx, 0, cx_orig],
        [0, fy, cy_orig],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return rgb_crop, depth_crop, bbox_center_crop, bbox_center_orig, K_crop, K_orig, pad_l, pad_t, x1, y1, size


def run_inference(models, rgb_crop, depth_crop, bbox_center_crop, bbox_center_orig, K_crop, K_orig):
    """Run inference with all models and correct translations to original image space."""
    predictions = {}
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    rgb_tensor = transform(rgb_crop).unsqueeze(0).to(DEVICE)
    
    # Depth processing
    depth_meters = depth_crop.astype(np.float32) / 1000.0
    depth_min, depth_max = 0.1, 1.6
    depth_normalized = np.clip((depth_meters - depth_min) / (depth_max - depth_min), 0, 1)
    depth_normalized[depth_meters < 0.01] = 0
    depth_tensor = torch.from_numpy(depth_normalized[..., np.newaxis]).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    depth_raw_tensor = torch.from_numpy(depth_meters).float().unsqueeze(0).to(DEVICE)
    
    bbox_center_tensor = torch.from_numpy(bbox_center_crop).float().unsqueeze(0).to(DEVICE)
    K_crop_tensor = torch.from_numpy(K_crop).float().unsqueeze(0).to(DEVICE)
    
    # Original camera parameters for geometric correction
    fx_orig, fy_orig = K_orig[0, 0], K_orig[1, 1]
    cx_orig, cy_orig = K_orig[0, 2], K_orig[1, 2]
    
    with torch.no_grad():
        for name, model in models.items():
            try:
                if name == 'RGB':
                    pred_rot, pred_trans = model(rgb_tensor)
                elif name == 'RGB-Geo':
                    pred_rot, pred_trans = model(rgb_tensor, bbox_center_tensor, K_crop_tensor)
                elif name == 'RGBD':
                    pred_rot, pred_trans = model(rgb_tensor, depth_tensor)
                elif name == 'RGBD-Geo':
                    # Use K_crop and bbox_center_crop - matching training setup
                    pred_rot, pred_trans = model(rgb_tensor, depth_tensor, depth_raw_tensor, bbox_center_tensor, K_crop_tensor)
                
                pred_rot_np = pred_rot.cpu().numpy()[0]
                pred_trans_np = pred_trans.cpu().numpy().flatten()
                
                # Apply geometric correction: re-compute X,Y using original bbox center
                # The predicted Z is correct; we just need to project to original image space
                pred_z = pred_trans_np[2]
                corrected_x = (bbox_center_orig[0] - cx_orig) * pred_z / fx_orig
                corrected_y = (bbox_center_orig[1] - cy_orig) * pred_z / fy_orig
                corrected_trans = np.array([corrected_x, corrected_y, pred_z])
                
                predictions[name] = (pred_rot_np, corrected_trans)
            except Exception as e:
                print(f"{name} inference failed: {e}")
    
    return predictions


def visualize_comparison(num_samples=5):
    """Create visual comparison for random samples, saving each as a separate PNG."""
    print("Loading models...")
    models = load_models()
    
    if not models:
        print("No models loaded")
        return
    
    obj_folders = sorted([f for f in os.listdir(DATA_ROOT) if f.isdigit()])
    
    colors = {
        'GT': (0, 255, 0),
        'RGB': (255, 165, 0),
        'RGB-Geo': (0, 255, 255),
        'RGBD': (255, 0, 255),
        'RGBD-Geo': (255, 255, 0),
    }
    
    # Create results directory
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    for sample_idx in range(num_samples):
        obj_id_str = np.random.choice(obj_folders)
        gt_path = os.path.join(DATA_ROOT, obj_id_str, "gt.yml")
        with open(gt_path, 'r') as f:
            gts = yaml.safe_load(f)
        frame_ids = list(gts.keys())
        frame_id = np.random.choice(frame_ids)
        
        print(f"Sample {sample_idx+1}: Object {obj_id_str}, Frame {frame_id}")
        
        rgb, depth, gt_rot, gt_trans, K, bbox = get_sample_data(obj_id_str, frame_id)
        corners = load_mesh_corners(MESH_DIR, obj_id_str)
        
        if gt_rot is None or corners is None:
            print(f"  Skipping - missing GT or corners")
            continue
        
        rgb_crop, depth_crop, bbox_center_crop, bbox_center_orig, K_crop, K_orig, *_ = prepare_crop(rgb, depth, bbox)
        predictions = run_inference(models, rgb_crop, depth_crop, bbox_center_crop, bbox_center_orig, K_crop, K_orig)
        
        # Create figure for this sample - one row with GT + all models
        num_cols = len(models) + 1
        fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
        
        # Ground truth
        img_gt = rgb.copy()
        gt_pts = project_points(corners, gt_rot, gt_trans, K)
        draw_3d_box(img_gt, gt_pts, colors['GT'], 3)
        axes[0].imshow(img_gt)
        axes[0].set_title(f"Ground Truth\nObj {obj_id_str}", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Model predictions
        for i, (name, (pred_rot, pred_trans)) in enumerate(predictions.items()):
            img_pred = rgb.copy()
            draw_3d_box(img_pred, gt_pts, colors['GT'], 2)  # GT in green (thinner)
            pred_pts = project_points(corners, pred_rot, pred_trans, K)
            draw_3d_box(img_pred, pred_pts, colors.get(name, (255, 0, 0)), 3)
            
            axes[i+1].imshow(img_pred)
            axes[i+1].set_title(f"{name}\n(Green=GT)", fontsize=14, fontweight='bold')
            axes[i+1].axis('off')
        
        plt.suptitle(f"Model Comparison - Sample {sample_idx+1}: Object {obj_id_str}, Frame {frame_id}", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save individual file
        save_path = os.path.join(results_dir, f"comparison_sample_{sample_idx+1}_obj{obj_id_str}_frame{frame_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        
        plt.show()
        plt.close(fig)
    
    print(f"\nAll {num_samples} comparison images saved to: {results_dir}")


if __name__ == "__main__":
    visualize_comparison(num_samples=5)

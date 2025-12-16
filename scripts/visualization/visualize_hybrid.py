"""
Visualize Hybrid model predictions on test set
Similar to visualize_rgb.py but for hybrid model
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data.dataset_hybrid import LineMODDataset
from models.pose_net_hybrid import PoseNet
from models.loss import ADDLoss
import yaml
from scipy.spatial.transform import Rotation as R

def project_3d_points(points_3d, rotation_matrix, translation, camera_matrix):
    """Project 3D points to 2D using camera parameters"""
    # Transform points
    points_cam = (rotation_matrix @ points_3d.T).T + translation
    
    # Project to image
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    points_2d = np.zeros((len(points_3d), 2))
    points_2d[:, 0] = (points_cam[:, 0] / points_cam[:, 2]) * fx + cx
    points_2d[:, 1] = (points_cam[:, 1] / points_cam[:, 2]) * fy + cy
    
    return points_2d.astype(int)

def create_3d_bbox(diameter):
    """Create 3D bounding box points"""
    r = diameter / 2
    corners = np.array([
        [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],  # Back face
        [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]       # Front face
    ])
    return corners

def draw_bbox(image, corners_2d, color=(0, 255, 0)):
    """Draw 3D bounding box on image"""
    # Draw back face
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, 2)
    # Draw front face
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[4 + (i+1)%4]), color, 2)
    # Draw connecting lines
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, 2)

def visualize_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = PoseNet().to(device)
    checkpoint_path = os.path.join(PROJECT_ROOT, "weights_hybrid", "best_pose_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded Hybrid model from {checkpoint_path}")
    
    # Load test data
    test_dataset = LineMODDataset(
        data_dir=os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed"),
        split='test',
        augment_bbox=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Load object diameters
    with open(os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models", "models_info.yml"), 'r') as f:
        models_info = yaml.safe_load(f)
    
    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, "inference_results", "hybrid")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize loss for ADD computation
    add_loss_fn = ADDLoss(
        models_dir=os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models"),
        rot_weight=0.0,
        trans_weight=0.0
    )
    
    print(f"📊 Evaluating on {len(test_dataset)} test samples...")
    
    total_add = 0
    num_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # Visualize first 10 samples
                break
            
            rgb, gt_quat, gt_trans, obj_id, bbox_center, camera_matrix = batch
            rgb = rgb.to(device)
            bbox_center = bbox_center.to(device)
            camera_matrix = camera_matrix.to(device)
            
            # Predict pose
            pred_quat, pred_trans = model(rgb, bbox_center, camera_matrix)
            
            # Compute ADD error
            add_error = add_loss_fn(pred_quat, pred_trans, gt_quat.to(device), 
                                   gt_trans.to(device), obj_id.to(device)).item()
            total_add += add_error
            num_samples += 1
            
            # Convert to numpy for visualization
            rgb_np = (rgb[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            
            # Convert predictions to rotation matrix
            pred_quat_np = pred_quat[0].cpu().numpy()
            pred_rot = R.from_quat(pred_quat_np).as_matrix()
            pred_trans_np = pred_trans[0].cpu().numpy()
            
            # Convert ground truth
            gt_quat_np = gt_quat[0].cpu().numpy()
            gt_rot = R.from_quat(gt_quat_np).as_matrix()
            gt_trans_np = gt_trans[0].cpu().numpy()
            
            # Get object info
            obj_idx = obj_id[0].item()
            diameter = models_info[obj_idx]['diameter'] / 1000.0  # mm to m
            
            # Create 3D bbox
            bbox_3d = create_3d_bbox(diameter)
            
            # Get camera matrix
            cam_mat = camera_matrix[0].cpu().numpy()
            K = np.array([
                [cam_mat[0], 0, cam_mat[2]],
                [0, cam_mat[1], cam_mat[3]],
                [0, 0, 1]
            ])
            
            # Project ground truth
            gt_corners = project_3d_points(bbox_3d, gt_rot, gt_trans_np, K)
            draw_bbox(rgb_np, gt_corners, color=(0, 255, 0))  # Green
            
            # Project prediction
            pred_corners = project_3d_points(bbox_3d, pred_rot, pred_trans_np, K)
            draw_bbox(rgb_np, pred_corners, color=(255, 0, 255))  # Magenta
            
            # Add text
            cv2.putText(rgb_np, f"Object: {obj_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(rgb_np, f"ADD Error: {add_error*1000:.1f}mm", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(rgb_np, "Green: GT, Magenta: Pred", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save
            output_path = os.path.join(output_dir, f"obj_{obj_idx:02d}_sample_{i:04d}.jpg")
            cv2.imwrite(output_path, rgb_np)
            
            print(f"  Sample {i+1}: Object {obj_idx}, ADD Error: {add_error*1000:.2f}mm")
    
    avg_add = (total_add / num_samples) * 1000  # Convert to mm
    print(f"\n✅ Visualization complete!")
    print(f"📊 Average ADD Error: {avg_add:.2f}mm")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    visualize_hybrid()

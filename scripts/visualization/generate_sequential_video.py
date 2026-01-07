"""Generate sequential video with all frames showing GT vs predicted poses with ADD error."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

from models.pose_net_rgb import PoseNetRGB
from models.pose_net_rgbd import PoseNetRGBD
from utils.mesh_utils import load_mesh_corners
from utils.visualization import project_points, draw_3d_box
from utils.camera import DEFAULT_K


def load_model(model_type, device):
    """Load pose estimation model."""
    if model_type == "rgb":
        model = PoseNetRGB(pretrained=True)
        weights_path = os.path.join(PROJECT_ROOT, "weights_rgb", "best_pose_model.pth")
    elif model_type == "rgbd":
        model = PoseNetRGBD(pretrained=True)
        weights_path = os.path.join(PROJECT_ROOT, "weights_rgbd", "best_pose_model.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded {model_type} model from {weights_path}")
    return model


def load_ground_truth(obj_folder):
    """Load ground truth poses and camera info."""
    gt_path = os.path.join(obj_folder, 'gt.yml')
    info_path = os.path.join(obj_folder, 'info.yml')
    
    with open(gt_path, 'r') as f:
        gts = yaml.safe_load(f)
    with open(info_path, 'r') as f:
        infos = yaml.safe_load(f)
    
    return gts, infos


def load_frame(obj_folder, frame_idx, load_depth=False):
    """Load RGB image and optionally depth for a specific frame."""
    rgb_path = os.path.join(obj_folder, 'rgb', f'{frame_idx:04d}.png')
    rgb = cv2.imread(rgb_path)
    
    if rgb is None:
        return None, None
    
    # Convert BGR to RGB (same as dataset)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth = None
    if load_depth:
        depth_path = os.path.join(obj_folder, 'depth', f'{frame_idx:04d}.png')
        if os.path.exists(depth_path):
            # Load as uint16, filter, keep as uint16 (same as dataset and inference script)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                depth_float = depth.astype(np.float32)
                depth_float = cv2.bilateralFilter(depth_float, 5, 75, 75)
                depth = depth_float.astype(np.uint16)
    
    return rgb, depth


def get_bbox_from_xywh(x, y, w, h):
    """Convert bbox from [x, y, w, h] format to [cx, cy, x_min, y_min, x_max, y_max]."""
    cx = x + w / 2
    cy = y + h / 2
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [cx, cy, x_min, y_min, x_max, y_max]


def compute_add(model_points, pred_rot, pred_trans, gt_rot, gt_trans):
    """Compute ADD metric."""
    # Convert quaternions to rotation matrices
    pred_R = R.from_quat(pred_rot).as_matrix()
    gt_R = R.from_quat(gt_rot).as_matrix()
    
    # Transform points with predicted and GT poses
    pred_pts = (pred_R @ model_points.T).T + pred_trans
    gt_pts = (gt_R @ model_points.T).T + gt_trans
    
    # Compute mean distance
    distances = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return np.mean(distances)


def run_inference(model, model_type, rgb, depth, bbox, camera_matrix, obj_id, device):
    """Run inference on a single frame using same logic as inference_rgbd.py."""
    # Prepare input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_size = 224
    
    # Get bbox dimensions
    cx, cy = bbox[0], bbox[1]  # Center from bbox
    x_min, y_min, x_max, y_max = bbox[2], bbox[3], bbox[4], bbox[5]
    w = x_max - x_min
    h = y_max - y_min
    
    # Square crop with 1.2x padding (same as inference script)
    size = max(w, h) * 1.2
    crop_x1 = int(cx - size/2)
    crop_y1 = int(cy - size/2)
    crop_size = int(size)
    
    # Padding if crop goes outside image
    h_img, w_img = rgb.shape[:2]
    pad_l = max(0, -crop_x1)
    pad_t = max(0, -crop_y1)
    pad_r = max(0, (crop_x1 + crop_size) - w_img)
    pad_b = max(0, (crop_y1 + crop_size) - h_img)
    
    padded_rgb = cv2.copyMakeBorder(rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    if depth is not None:
        padded_depth = cv2.copyMakeBorder(depth, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    else:
        padded_depth = None
    
    # Adjust crop coordinates for padding
    adj_x1 = crop_x1 + pad_l
    adj_y1 = crop_y1 + pad_t
    
    crop_rgb = padded_rgb[adj_y1:adj_y1+crop_size, adj_x1:adj_x1+crop_size]
    
    if crop_rgb.size == 0 or crop_rgb.shape[0] != crop_size or crop_rgb.shape[1] != crop_size:
        return None, None
    
    # Resize crop
    crop_rgb_resized = cv2.resize(crop_rgb, (img_size, img_size))
    rgb_tensor = transform(crop_rgb_resized).unsqueeze(0).to(device)
    
    # Prepare inputs based on model type
    with torch.no_grad():
        if model_type == "rgb":
            # RGB model needs: rgbm (4ch), bbox_center, camera_matrix
            bbox_center_orig = np.array([cx, cy], dtype=np.float32)
            bbox_center = torch.from_numpy(bbox_center_orig).unsqueeze(0).to(device)
            cam_matrix_flat = torch.tensor([[camera_matrix[0], camera_matrix[1], 
                                             camera_matrix[2], camera_matrix[3]]], 
                                           dtype=torch.float32).to(device)
            pred_rot, pred_trans = model(rgb_tensor, bbox_center, cam_matrix_flat)
        elif model_type == "rgbd":
            # RGBD model needs: rgbdm (5ch), z_sensor, bbox_center, camera_matrix
            if depth is None or padded_depth is None:
                return None, None
            
            # Crop depth (uint16)
            crop_depth = padded_depth[adj_y1:adj_y1+crop_size, adj_x1:adj_x1+crop_size]
            
            # Resize depth (as float32, same as inference script)
            crop_depth_resized = cv2.resize(crop_depth.astype(np.float32), (img_size, img_size))
            
            # Use original bbox center (same as inference script)
            bbox_center_orig = np.array([cx, cy], dtype=np.float32)
            
            # Normalize depth for CNN input (same as training and inference)
            depth_meters = crop_depth_resized / 1000.0
            depth_min, depth_max = 0.1, 2.0
            depth_normalized = (depth_meters - depth_min) / (depth_max - depth_min)
            depth_normalized = np.clip(depth_normalized, 0, 1)
            depth_normalized[depth_meters < 0.01] = 0
            
            # Prepare tensors
            depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
            bbox_center = torch.from_numpy(bbox_center_orig).unsqueeze(0).to(device)
            cam_matrix_flat = torch.tensor([[camera_matrix[0], camera_matrix[1], 
                                             camera_matrix[2], camera_matrix[3]]], 
                                           dtype=torch.float32).to(device)
            
            # Extract z_sensor (measured depth at bbox center)
            z_sensor_val = crop_depth_resized[img_size//2, img_size//2] / 1000.0
            z_sensor = torch.tensor([[z_sensor_val]], dtype=torch.float32).to(device)
            
            # TODO: Add mask support once YOLO-seg integration is complete
            # For now, create dummy mask (all ones)
            mask_tensor = torch.ones(1, 1, img_size, img_size).to(device)
            
            # Concatenate RGB + Depth + Mask (5 channels)
            rgbdm = torch.cat([rgb_tensor, depth_tensor, mask_tensor], dim=1)
            
            # RGBD model forward pass
            pred_rot, pred_trans = model(rgbdm, z_sensor, bbox_center, cam_matrix_flat)
    
    pred_rot = pred_rot[0].cpu().numpy()
    pred_trans = pred_trans[0].cpu().numpy()
    
    return pred_rot, pred_trans


def draw_frame(rgb, corners_3d, gt_rot, gt_trans, pred_rot, pred_trans, camera_K, add_error, frame_idx):
    """Draw GT and predicted bboxes on frame with ADD error."""
    vis_img = rgb.copy()
    
    # Draw ground truth (thin green lines)
    gt_pts_2d = project_points(corners_3d, gt_rot, gt_trans, camera_K)
    draw_3d_box(vis_img, gt_pts_2d, color=(0, 255, 0), thickness=1)
    
    # Draw prediction (thicker blue lines)
    pred_pts_2d = project_points(corners_3d, pred_rot, pred_trans, camera_K)
    draw_3d_box(vis_img, pred_pts_2d, color=(255, 0, 0), thickness=3)
    
    # Add text overlay with ADD error and frame number
    cv2.rectangle(vis_img, (5, 5), (300, 80), (0, 0, 0), -1)
    cv2.putText(vis_img, f"Frame: {frame_idx:04d}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_img, f"ADD Error: {add_error*1000:.2f}mm", (15, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add legend
    cv2.rectangle(vis_img, (vis_img.shape[1] - 150, 5), (vis_img.shape[1] - 5, 80), (0, 0, 0), -1)
    cv2.line(vis_img, (vis_img.shape[1] - 140, 30), (vis_img.shape[1] - 100, 30), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis_img, "GT", (vis_img.shape[1] - 90, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(vis_img, (vis_img.shape[1] - 140, 60), (vis_img.shape[1] - 100, 60), (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis_img, "Pred", (vis_img.shape[1] - 90, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return vis_img


def generate_video(model_type, object_id, fps=30, output_dir=None, frame_duplication=1):
    """Generate sequential video for all frames of an object.
    
    Args:
        model_type: Type of model to use
        object_id: Object ID string
        fps: Output video FPS (playback speed)
        output_dir: Output directory
        frame_duplication: How many times to duplicate each frame for smoother playback
                          (e.g., 6x duplication at 30fps = effective 5fps speed)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(model_type, device)
    
    # Setup paths
    obj_folder = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data", object_id)
    mesh_dir = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
    
    if not os.path.exists(obj_folder):
        raise FileNotFoundError(f"Object folder not found: {obj_folder}")
    
    # Load mesh corners for ADD computation and visualization
    corners_3d = load_mesh_corners(mesh_dir, object_id)
    if corners_3d is None:
        raise FileNotFoundError(f"Could not load mesh for object {object_id}")
    
    # Load model points for ADD metric
    ply_path = os.path.join(mesh_dir, f"obj_{object_id}.ply")
    model_points = []
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
                model_points.append([float(vals[0]), float(vals[1]), float(vals[2])])
    model_points = np.array(model_points) / 1000.0
    distances = np.linalg.norm(model_points, axis=1)
    model_points = model_points[distances < 0.3]
    
    # Load ground truth and camera info
    gts, infos = load_ground_truth(obj_folder)
    
    # Find all available frames
    rgb_folder = os.path.join(obj_folder, 'rgb')
    all_frames = sorted([int(f.split('.')[0]) for f in os.listdir(rgb_folder) if f.endswith('.png')])
    
    if len(all_frames) == 0:
        raise ValueError(f"No frames found in {rgb_folder}")
    
    print(f"Found {len(all_frames)} frames for object {object_id}")
    print(f"Frame range: {all_frames[0]:04d} to {all_frames[-1]:04d}")
    
    # Setup output
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "demo_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"sequential_{model_type}_obj{object_id}_fps{fps}.mp4")
    
    # Get frame size from first frame
    first_rgb, _ = load_frame(obj_folder, all_frames[0], load_depth=(model_type == "rgbd"))
    h, w = first_rgb.shape[:2]
    
    # Setup video writer - try different codecs for best compatibility
    # Try mp4v (widely compatible) or XVID as fallback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Most compatible
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not writer.isOpened():
        print("Warning: mp4v codec failed, trying XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Process each frame
    add_errors = []
    skipped_frames = []
    
    print(f"\nProcessing {len(all_frames)} frames...")
    for frame_idx in tqdm(all_frames):
        # Load frame
        rgb, depth = load_frame(obj_folder, frame_idx, load_depth=(model_type == "rgbd"))
        if rgb is None:
            skipped_frames.append(frame_idx)
            continue
        
        # Load ground truth pose and bbox
        if frame_idx not in gts:
            skipped_frames.append(frame_idx)
            continue
        
        gt_rot_mat = np.array(gts[frame_idx][0]['cam_R_m2c']).reshape(3, 3)
        gt_trans = np.array(gts[frame_idx][0]['cam_t_m2c']).reshape(3) / 1000.0
        gt_rot = R.from_matrix(gt_rot_mat).as_quat()  # [x, y, z, w]
        
        # Get bbox from ground truth (same as used during training)
        gt_bbox_xywh = gts[frame_idx][0]['obj_bb']  # [x, y, w, h]
        bbox = get_bbox_from_xywh(*gt_bbox_xywh)
        
        # Get camera intrinsics
        if frame_idx not in infos:
            skipped_frames.append(frame_idx)
            continue
        
        camera_K = np.array(infos[frame_idx]['cam_K']).reshape(3, 3)
        camera_matrix = [camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]]
        
        # Run inference
        obj_id_int = int(object_id) - 1  # Model expects 0-indexed
        pred_rot, pred_trans = run_inference(model, model_type, rgb, depth, bbox, camera_matrix, obj_id_int, device)
        
        if pred_rot is None:
            skipped_frames.append(frame_idx)
            continue
        
        # Compute ADD error
        add_error = compute_add(model_points, pred_rot, pred_trans, gt_rot, gt_trans)
        add_errors.append(add_error)
        
        # Draw visualization
        vis_img = draw_frame(rgb, corners_3d, gt_rot, gt_trans, pred_rot, pred_trans, camera_K, add_error, frame_idx)
        
        # Convert RGB back to BGR for video writer
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # Write frame multiple times for smoother playback at same effective speed
        for _ in range(frame_duplication):
            writer.write(vis_img_bgr)
    
    writer.release()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Video saved to: {output_path}")
    print(f"Total frames processed: {len(all_frames) - len(skipped_frames)}/{len(all_frames)}")
    if len(skipped_frames) > 0:
        print(f"Skipped frames: {len(skipped_frames)}")
        if len(skipped_frames) <= 10:
            print(f"  Frames: {skipped_frames}")
    
    if len(add_errors) > 0:
        add_errors = np.array(add_errors)
        print(f"\nADD Statistics:")
        print(f"  Mean:   {np.mean(add_errors)*1000:.2f} mm")
        print(f"  Median: {np.median(add_errors)*1000:.2f} mm")
        print(f"  Std:    {np.std(add_errors)*1000:.2f} mm")
        print(f"  Min:    {np.min(add_errors)*1000:.2f} mm")
        print(f"  Max:    {np.max(add_errors)*1000:.2f} mm")
        
        # Compute ADD metric (percentage under 20mm threshold)
        add_20mm = np.mean(add_errors < 0.02) * 100
        print(f"  ADD@20mm: {add_20mm:.1f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate sequential video with GT vs predicted poses')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['rgb', 'rgbd'],
                        help='Model type to use')
    parser.add_argument('--object', type=str, required=True,
                        help='Object ID (e.g., "01" for ape)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS (default: 30)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: demo_videos/)')
    parser.add_argument('--frame-dup', type=int, default=1,
                        help='Frame duplication factor for smoother playback (default: 1, use 6 for 5fps-equivalent at 30fps)')
    
    args = parser.parse_args()
    
    generate_video(args.model, args.object, args.fps, args.output, args.frame_dup)


if __name__ == "__main__":
    main()

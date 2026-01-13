"""Utility functions for inference scripts."""

import os
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def load_ground_truth(data_root, obj_id_str, frame_id):
    """Load ground truth pose from LineMOD dataset.
    
    Args:
        data_root: Path to Linemod_preprocessed/data
        obj_id_str: Object ID string (e.g., "01", "02")
        frame_id: Frame number
    
    Returns:
        tuple: (rotation_matrix, translation) or (None, None)
    """
    gt_path = os.path.join(data_root, obj_id_str, "gt.yml")
    if not os.path.exists(gt_path):
        return None, None
    
    with open(gt_path, 'r') as f:
        gts = yaml.safe_load(f)
    
    if frame_id not in gts:
        return None, None
    
    for anno in gts[frame_id]:
        if str(int(anno['obj_id'])).zfill(2) == obj_id_str:
            gt_rot = np.array(anno['cam_R_m2c']).reshape(3, 3)
            gt_trans = np.array(anno['cam_t_m2c']) / 1000.0  # mm to meters
            return gt_rot, gt_trans
    
    return None, None


def load_model_points(mesh_dir, obj_id_str, num_points=500):
    """Load 3D model points for ADD computation.
    
    Args:
        mesh_dir: Path to directory containing .ply mesh files
        obj_id_str: Object ID string (e.g., "01", "02")
        num_points: Maximum number of points to sample
    
    Returns:
        numpy array of shape (N, 3) or None if mesh not found
    """
    ply_path = os.path.join(mesh_dir, f"obj_{obj_id_str}.ply")
    if not os.path.exists(ply_path):
        return None
    
    verts = []
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
                verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    
    pts = np.array(verts) / 1000.0  # mm to meters
    
    # Filter outliers and downsample
    distances = np.linalg.norm(pts, axis=1)
    pts = pts[distances < 0.5]
    
    if len(pts) > num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    
    return pts


def compute_add(pred_quat, pred_trans, gt_rot, gt_trans, model_points):
    """Compute ADD (Average Distance of Model Points) and error breakdown.
    
    Args:
        pred_quat: Predicted quaternion [x, y, z, w] (4,)
        pred_trans: Predicted translation [x, y, z] (3,)
        gt_rot: Ground truth rotation matrix (3, 3)
        gt_trans: Ground truth translation [x, y, z] (3,)
        model_points: 3D model points (N, 3)
    
    Returns:
        dict with keys: 'add_mm', 'trans_error_mm', 'rot_error_deg', 
                       'trans_xyz_mm', 'pred_trans', 'gt_trans'
    """
    # Convert quaternion to rotation matrix
    pred_R = R.from_quat([pred_quat[0], pred_quat[1], pred_quat[2], pred_quat[3]]).as_matrix()
    
    # Translation error breakdown (X, Y, Z in mm)
    trans_diff = (pred_trans - gt_trans) * 1000  # to mm
    trans_error = np.linalg.norm(trans_diff)
    
    # Rotation error (geodesic angle in degrees)
    R_diff = pred_R @ gt_rot.T
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_error_rad = np.arccos(cos_angle)
    rot_error_deg = np.degrees(rot_error_rad)
    
    # Transform model points
    gt_points = model_points @ gt_rot.T + gt_trans
    pred_points = model_points @ pred_R.T + pred_trans
    
    # Compute ADD
    add_dist = np.linalg.norm(pred_points - gt_points, axis=1).mean() * 1000
    
    return {
        'add_mm': add_dist,
        'trans_error_mm': trans_error,
        'trans_xyz_mm': trans_diff,  # [X, Y, Z] errors
        'rot_error_deg': rot_error_deg,
        'pred_trans': pred_trans,
        'gt_trans': gt_trans
    }


def parse_image_filename(img_path):
    """Parse object ID and frame ID from image path.
    
    Handles both formats:
        - LineMOD path: 'data/01/rgb/0219.png' -> ('01', 219)
        - Legacy format: '01_0219.png' -> ('01', 219)
    
    Returns:
        tuple: (obj_id_str, frame_id) or (None, None)
    """
    # Try LineMOD directory format first: data/{obj_id}/rgb/{frame}.png
    parts = img_path.replace('\\', '/').split('/')
    if 'rgb' in parts or 'depth' in parts:
        try:
            # Find obj_id (folder before 'rgb' or 'depth')
            for i, p in enumerate(parts):
                if p in ['rgb', 'depth'] and i > 0:
                    obj_id_str = parts[i - 1]
                    filename = os.path.basename(img_path)
                    frame_id = int(filename.replace('.png', '').replace('.jpg', ''))
                    return obj_id_str, frame_id
        except:
            pass
    
    # Fallback: legacy format '01_0219.png'
    filename = os.path.basename(img_path)
    parts = filename.replace('.png', '').replace('.jpg', '').split('_')
    if len(parts) >= 2:
        obj_id_str = parts[0]
        frame_id = int(parts[1])
        return obj_id_str, frame_id
    
    return None, None

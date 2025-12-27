"""Utility functions for loading camera and ground truth data."""

import os
import numpy as np
import yaml

# Default LineMOD camera intrinsics
DEFAULT_K = np.array([
    [572.4114, 0.0, 325.2611],
    [0.0, 573.57043, 242.04899],
    [0.0, 0.0, 1.0]
])


def get_gt_and_K(data_dir, obj_id_str, frame_id):
    """
    Load ground truth pose and camera intrinsics for a frame.
    
    Args:
        data_dir: Path to Linemod_preprocessed/data
        obj_id_str: Object ID string (e.g., "01", "02")
        frame_id: Frame number
    
    Returns:
        tuple: (rotation_matrix, translation, camera_matrix)
               All may be None if not found
    """
    gt_path = os.path.join(data_dir, obj_id_str, "gt.yml")
    info_path = os.path.join(data_dir, obj_id_str, "info.yml")
    
    r_mat, t, K = None, None, None
    
    # Load camera intrinsics
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            infos = yaml.safe_load(f)
        if frame_id in infos:
            K = np.array(infos[frame_id]['cam_K']).reshape(3, 3)
        elif infos:
            K = np.array(list(infos.values())[0]['cam_K']).reshape(3, 3)
    
    if K is None:
        K = DEFAULT_K.copy()

    # Load ground truth pose
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

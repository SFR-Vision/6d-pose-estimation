"""Utility functions for mesh loading and 3D bounding box operations."""

import os
import numpy as np


def load_mesh_corners(mesh_dir, obj_id_str):
    """
    Load 3D bounding box corners from mesh file.
    
    Args:
        mesh_dir: Path to directory containing .ply mesh files
        obj_id_str: Object ID string (e.g., "01", "02")
    
    Returns:
        numpy array of shape (8, 3) with box corners, or None if mesh not found
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
    
    verts = np.array(verts) / 1000.0  # mm to meters
    
    # Filter outliers
    distances = np.linalg.norm(verts, axis=1)
    verts = verts[distances < 0.3]
    if len(verts) == 0:
        return None
    
    # Use percentiles for robust bounding box
    min_pt = np.percentile(verts, 1, axis=0)
    max_pt = np.percentile(verts, 99, axis=0)
    
    return np.array([
        [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]]
    ])

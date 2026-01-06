"""Utility functions for mesh loading and 3D bounding box operations."""

import os
import numpy as np
import yaml


def load_mesh_corners_from_info(models_dir, obj_id_str):
    """
    Load 3D bounding box corners from models_info.yml (faster, official dimensions).
    
    Args:
        models_dir: Path to directory containing models_info.yml
        obj_id_str: Object ID string (e.g., "01", "02")
    
    Returns:
        numpy array of shape (8, 3) with box corners in meters, or None if not found
    """
    models_info_path = os.path.join(models_dir, "models_info.yml")
    if not os.path.exists(models_info_path):
        return None
    
    try:
        with open(models_info_path, 'r') as f:
            info = yaml.safe_load(f)
        
        # Convert obj_id_str to integer key (e.g., "02" -> 2)
        obj_key = int(obj_id_str)
        if obj_key not in info:
            return None
        
        obj_info = info[obj_key]
        
        # Get bounding box dimensions (convert mm to meters)
        min_x = obj_info['min_x'] / 1000.0
        min_y = obj_info['min_y'] / 1000.0
        min_z = obj_info['min_z'] / 1000.0
        
        max_x = (obj_info['min_x'] + obj_info['size_x']) / 1000.0
        max_y = (obj_info['min_y'] + obj_info['size_y']) / 1000.0
        max_z = (obj_info['min_z'] + obj_info['size_z']) / 1000.0
        
        # Create 8 corners (same order as mesh-based version)
        return np.array([
            [min_x, min_y, min_z], [max_x, min_y, min_z],
            [max_x, max_y, min_z], [min_x, max_y, min_z],
            [min_x, min_y, max_z], [max_x, min_y, max_z],
            [max_x, max_y, max_z], [min_x, max_y, max_z]
        ])
    except Exception as e:
        print(f"Error loading corners from models_info.yml: {e}")
        return None


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

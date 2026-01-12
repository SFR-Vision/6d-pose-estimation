"""Convert LineMOD GT masks to YOLO segmentation format."""

import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Paths
LINEMOD_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "datasets", "yolo_seg_ready")

# LineMOD objects (no 03, 07)
OBJECTS = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']


def extract_contour_polygon(mask, min_points=3, simplify_epsilon=2.0):
    """
    Extract polygon from binary mask.
    
    Args:
        mask: Binary mask (H, W) with 255=object, 0=background
        min_points: Minimum number of points for valid polygon
        simplify_epsilon: Contour approximation epsilon (lower=more points)
    
    Returns:
        Normalized polygon points as flat list [x1, y1, x2, y2, ...] or None if invalid
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Check if contour is large enough
    if cv2.contourArea(contour) < 100:  # Minimum 100 pixels
        return None
    
    # Simplify contour (reduce number of points)
    epsilon = simplify_epsilon
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Check if we have enough points
    if len(approx) < min_points:
        return None
    
    # Convert to normalized coordinates
    h, w = mask.shape
    points = []
    for point in approx:
        x, y = point[0]
        # Normalize to [0, 1]
        x_norm = x / w
        y_norm = y / h
        # Clamp to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        points.extend([x_norm, y_norm])
    
    return points


def process_linemod_dataset():
    """Convert all LineMOD masks to YOLO segmentation format."""
    
    print("=" * 60)
    print("Converting LineMOD GT Masks to YOLO Segmentation Format")
    print("=" * 60)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', split), exist_ok=True)
    
    stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
    
    for obj_id in OBJECTS:
        obj_dir = os.path.join(LINEMOD_ROOT, obj_id)
        
        if not os.path.exists(obj_dir):
            print(f"⚠ Skipping object {obj_id} (directory not found)")
            continue
        
        # Get list of all images
        rgb_dir = os.path.join(obj_dir, 'rgb')
        mask_dir = os.path.join(obj_dir, 'mask')
        
        if not os.path.exists(mask_dir):
            print(f"⚠ Skipping object {obj_id} (no mask directory)")
            continue
        
        img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        
        print(f"\nProcessing Object {obj_id}: {len(img_files)} images")
        
        for i, img_file in enumerate(tqdm(img_files, desc=f"Object {obj_id}")):
            img_id = img_file.split('.')[0]
            
            # Determine split (same as pose training)
            cycle = i % 10
            if cycle == 8:
                split = 'val'
            elif cycle == 9:
                split = 'test'
            else:
                split = 'train'
            
            # Paths
            rgb_path = os.path.join(rgb_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            
            if not os.path.exists(mask_path):
                stats['skipped'] += 1
                continue
            
            # Read mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                stats['skipped'] += 1
                continue
            
            # Threshold to binary (white=255 is object)
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Extract polygon
            polygon = extract_contour_polygon(mask_binary)
            
            if polygon is None:
                stats['skipped'] += 1
                continue
            
            # Copy image
            out_img_name = f"{obj_id}_{img_id}.jpg"
            out_img_path = os.path.join(OUTPUT_ROOT, 'images', split, out_img_name)
            shutil.copy2(rgb_path, out_img_path)
            
            # Save YOLO segmentation label
            out_label_name = f"{obj_id}_{img_id}.txt"
            out_label_path = os.path.join(OUTPUT_ROOT, 'labels', split, out_label_name)
            
            # Class ID (0-indexed)
            class_id = OBJECTS.index(obj_id)
            
            # Write: <class_id> <x1> <y1> <x2> <y2> ...
            with open(out_label_path, 'w') as f:
                f.write(f"{class_id}")
                for coord in polygon:
                    f.write(f" {coord:.6f}")
                f.write("\n")
            
            stats[split] += 1
    
    # Create dataset.yaml
    yaml_content = {
        'path': OUTPUT_ROOT,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(OBJECTS),
        'names': OBJECTS
    }
    
    yaml_path = os.path.join(OUTPUT_ROOT, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    # Print summary
    print("Conversion Complete!")
    print("-" * 40)
    print(f"Train:   {stats['train']} samples")
    print(f"Val:     {stats['val']} samples")
    print(f"Test:    {stats['test']} samples")
    print(f"Skipped: {stats['skipped']} samples (no mask or invalid contour)")

if __name__ == "__main__":
    process_linemod_dataset()

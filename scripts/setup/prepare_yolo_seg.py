"""Convert LineMOD masks to YOLO segmentation format."""

import os
import sys
import cv2
import numpy as np
import yaml
import shutil
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

LINEMOD_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "datasets", "yolo_seg_ready")
OBJECTS = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']


def extract_polygon(mask, min_points=3, epsilon=2.0):
    """Extract normalized polygon from binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None
    
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < min_points:
        return None
    
    h, w = mask.shape
    points = []
    for pt in approx:
        x, y = pt[0]
        points.extend([
            max(0.0, min(1.0, x / w)),
            max(0.0, min(1.0, y / h))
        ])
    return points


def process_dataset():
    """Convert all LineMOD masks to YOLO format."""
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', split), exist_ok=True)
    
    stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}
    
    for obj_id in OBJECTS:
        obj_dir = os.path.join(LINEMOD_ROOT, obj_id)
        rgb_dir = os.path.join(obj_dir, 'rgb')
        mask_dir = os.path.join(obj_dir, 'mask')
        
        if not os.path.exists(mask_dir):
            print(f"Skipping object {obj_id}")
            continue
        
        img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        print(f"Object {obj_id}: {len(img_files)} images")
        
        for i, img_file in enumerate(tqdm(img_files, desc=f"Object {obj_id}")):
            img_id = img_file.split('.')[0]
            
            # Split: 80% train, 10% val, 10% test
            cycle = i % 10
            split = 'val' if cycle == 8 else ('test' if cycle == 9 else 'train')
            
            mask_path = os.path.join(mask_dir, img_file)
            if not os.path.exists(mask_path):
                stats['skipped'] += 1
                continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                stats['skipped'] += 1
                continue
            
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            polygon = extract_polygon(mask_binary)
            
            if polygon is None:
                stats['skipped'] += 1
                continue
            
            # Copy image
            out_name = f"{obj_id}_{img_id}"
            shutil.copy2(
                os.path.join(rgb_dir, img_file),
                os.path.join(OUTPUT_ROOT, 'images', split, f"{out_name}.jpg")
            )
            
            # Write label
            class_id = OBJECTS.index(obj_id)
            with open(os.path.join(OUTPUT_ROOT, 'labels', split, f"{out_name}.txt"), 'w') as f:
                f.write(f"{class_id} " + " ".join(f"{c:.6f}" for c in polygon) + "\n")
            
            stats[split] += 1
    
    # Create dataset.yaml
    with open(os.path.join(OUTPUT_ROOT, 'dataset.yaml'), 'w') as f:
        yaml.dump({
            'path': OUTPUT_ROOT,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(OBJECTS),
            'names': OBJECTS
        }, f)
    
    print(f"Done: train={stats['train']}, val={stats['val']}, test={stats['test']}, skipped={stats['skipped']}")


if __name__ == "__main__":
    process_dataset()

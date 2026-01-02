"""Quick test to verify depth reading is correct in RGBD-Geometric model."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torchvision import transforms
from data.dataset_rgbd import LineMODDatasetRGBD

DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")

def test_depth_vs_gt():
    """Compare depth sensor reading to GT translation Z."""
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LineMODDatasetRGBD(DATA_ROOT, mode='val', transform=transform)
    
    print(f"Testing {len(dataset)} samples...\n")
    
    z_errors = []
    
    for i in range(min(100, len(dataset))):  # Test first 100 samples
        rgb, depth_norm, depth_raw, quat, gt_trans, obj_id, bbox_center, cam_K = dataset[i]
        
        # GT Z value (already in meters from dataset)
        z_gt = gt_trans[2].item()
        
        # What the model does: sample center of depth crop
        center = 112
        region = 5
        depth_region = depth_raw[center-region:center+region+1, center-region:center+region+1]
        valid_mask = depth_region > 0.01
        
        if valid_mask.sum() > 0:
            z_sensor = depth_region[valid_mask].median().item()
        else:
            z_sensor = 0.5
        
        z_error = abs(z_sensor - z_gt) * 1000  # Convert to mm
        z_errors.append(z_error)
        
        if i < 10:  # Print first 10
            print(f"Sample {i}: Z_sensor={z_sensor*1000:.1f}mm, Z_gt={z_gt*1000:.1f}mm, Error={z_error:.1f}mm")
    
    print(f"\n--- Statistics over {len(z_errors)} samples ---")
    print(f"Mean Z error: {np.mean(z_errors):.1f}mm")
    print(f"Median Z error: {np.median(z_errors):.1f}mm")
    print(f"Max Z error: {np.max(z_errors):.1f}mm")
    print(f"Min Z error: {np.min(z_errors):.1f}mm")
    
    # Check if errors are reasonable
    if np.mean(z_errors) < 20:
        print("\n✅ Depth reading is GOOD! Mean error < 20mm")
    else:
        print(f"\n❌ Depth reading has issues! Mean error = {np.mean(z_errors):.1f}mm")


if __name__ == "__main__":
    test_depth_vs_gt()

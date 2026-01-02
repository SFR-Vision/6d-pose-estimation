"""Per-object ADD metrics comparison for all 3 pose estimation models."""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from data.dataset_rgb import LineMODDatasetRGB
from data.dataset_rgbd import LineMODDatasetRGBD
from models.add_loss import ADDLoss

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Object names mapping (obj_id = folder_number - 1)
OBJ_NAMES = {
    0: "01-ape",
    1: "02-benchvise",
    3: "04-camera",
    4: "05-can",
    5: "06-driller",
    7: "08-duck",
    8: "09-eggbox",
    9: "10-glue",
    10: "11-holepuncher",
    11: "12-iron",
    12: "13-lamp",
    13: "14-phone",
    14: "15-cat"
}

# Symmetric objects
SYMMETRIC_OBJECTS = {8, 9}  # 09-eggbox, 10-glue

WEIGHTS = {
    'RGB': os.path.join(PROJECT_ROOT, "weights_rgb", "best_pose_model.pth"),
    'RGB-Geo': os.path.join(PROJECT_ROOT, "weights_rgb_geometric", "best_pose_model.pth"),
    'RGBD-Geo': os.path.join(PROJECT_ROOT, "weights_rgbd_geometric", "best_pose_model.pth"),
}


def load_pose_model(model_name, weights_path):
    """Load a pose estimation model."""
    if not os.path.exists(weights_path):
        return None
    
    try:
        if model_name == 'RGB':
            from models.pose_net_rgb import PoseNetRGB
            model = PoseNetRGB(pretrained=False)
        elif model_name == 'RGB-Geo':
            from models.pose_net_rgb_geometric import PoseNetRGBGeometric
            model = PoseNetRGBGeometric(pretrained=False)
        elif model_name == 'RGBD-Geo':
            from models.pose_net_rgbd_geometric import PoseNetRGBDGeometric
            model = PoseNetRGBDGeometric(pretrained=False)
        else:
            return None
        
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE).eval()
        return model
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
        return None


def evaluate_per_object(model, model_name, dataset, add_loss, is_rgbd=False, needs_geometry=False):
    """Evaluate model and return per-object ADD metrics using ADDLoss class."""
    if model is None:
        return None
    
    # Per-object metrics storage
    obj_add_sums = {}
    obj_counts = {}
    obj_acc_2cm = {}
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}", leave=False):
            sample = dataset[i]
            
            if is_rgbd:
                rgb, depth, depth_raw, gt_rot, gt_trans, obj_id, bbox_center, cam_matrix = sample
                rgb = rgb.unsqueeze(0).to(DEVICE)
                depth = depth.unsqueeze(0).to(DEVICE)
                depth_raw = depth_raw.unsqueeze(0).to(DEVICE)
                bbox_center = bbox_center.unsqueeze(0).to(DEVICE)
                cam_matrix = cam_matrix.unsqueeze(0).to(DEVICE)
                
                if needs_geometry:
                    pred_rot, pred_trans = model(rgb, depth, depth_raw, bbox_center, cam_matrix)
            else:
                rgb, gt_rot, gt_trans, obj_id, bbox_center, cam_matrix = sample
                rgb = rgb.unsqueeze(0).to(DEVICE)
                bbox_center = bbox_center.unsqueeze(0).to(DEVICE)
                cam_matrix = cam_matrix.unsqueeze(0).to(DEVICE)
                
                if needs_geometry:
                    pred_rot, pred_trans = model(rgb, bbox_center, cam_matrix)
                else:
                    pred_rot, pred_trans = model(rgb)
            
            obj_id_val = int(obj_id.item()) if hasattr(obj_id, 'item') else int(obj_id)
            gt_rot = gt_rot.unsqueeze(0).to(DEVICE)
            gt_trans = gt_trans.unsqueeze(0).to(DEVICE)
            obj_ids_tensor = torch.tensor([obj_id_val], device=DEVICE)
            
            # Use ADDLoss class for evaluation
            metrics = add_loss.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids_tensor)
            add_dist = metrics['add_mean']  # Already in mm
            acc_2cm = metrics['add_2cm_acc']  # Accuracy %
            
            if obj_id_val not in obj_add_sums:
                obj_add_sums[obj_id_val] = 0.0
                obj_counts[obj_id_val] = 0
                obj_acc_2cm[obj_id_val] = 0.0
            
            obj_add_sums[obj_id_val] += add_dist
            obj_counts[obj_id_val] += 1
            obj_acc_2cm[obj_id_val] += acc_2cm
    
    # Compute averages
    results = {}
    for obj_id in obj_add_sums:
        results[obj_id] = {
            'add_mm': obj_add_sums[obj_id] / obj_counts[obj_id],
            'acc_2cm': obj_acc_2cm[obj_id] / obj_counts[obj_id],
            'count': obj_counts[obj_id]
        }
    
    return results


def main():
    print(f"\nPer-Object ADD Metrics Comparison on {DEVICE}\n")
    
    # Create ADDLoss evaluator
    print("Loading 3D models...")
    add_loss = ADDLoss(MODEL_DIR, DEVICE)
    print(f"  Loaded models\n")
    
    # Transforms
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("Loading datasets...")
    rgb_dataset = LineMODDatasetRGB(DATA_ROOT, mode='test', transform=val_transform)
    
    try:
        rgbd_dataset = LineMODDatasetRGBD(DATA_ROOT, mode='test', transform=val_transform)
    except:
        rgbd_dataset = None
        print("  RGBD dataset not available")
    
    print(f"  {len(rgb_dataset)} test samples\n")
    
    # Load models
    print("Loading models...")
    models = {}
    for name, path in WEIGHTS.items():
        models[name] = load_pose_model(name, path)
        status = "✓" if models[name] is not None else "✗"
        print(f"  {name}: {status}")
    print()
    
    # Evaluate each model
    all_results = {}
    
    if models['RGB'] is not None:
        all_results['RGB'] = evaluate_per_object(
            models['RGB'], 'RGB', rgb_dataset, add_loss,
            is_rgbd=False, needs_geometry=False
        )
    
    if models['RGB-Geo'] is not None:
        all_results['RGB-Geo'] = evaluate_per_object(
            models['RGB-Geo'], 'RGB-Geo', rgb_dataset, add_loss,
            is_rgbd=False, needs_geometry=True
        )
    
    if rgbd_dataset is not None:
        if models['RGBD-Geo'] is not None:
            all_results['RGBD-Geo'] = evaluate_per_object(
                models['RGBD-Geo'], 'RGBD-Geo', rgbd_dataset, add_loss,
                is_rgbd=True, needs_geometry=True
            )
    
    # Print results table - ADD (mm)
    print("\n" + "=" * 80)
    print("Per-Object ADD Metrics (mm) - Lower is better")
    print("=" * 80)
    
    model_names = [name for name in ['RGB', 'RGB-Geo', 'RGBD-Geo'] if name in all_results]
    
    header = f"{'Object':<18}"
    for name in model_names:
        header += f"{name:<12}"
    print(header)
    print("-" * 80)
    
    # Get all object IDs
    all_obj_ids = set()
    for results in all_results.values():
        if results:
            all_obj_ids.update(results.keys())
    
    model_avgs = {name: [] for name in model_names}
    
    for obj_id in sorted(all_obj_ids):
        obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
        sym_marker = "*" if obj_id in SYMMETRIC_OBJECTS else ""
        row = f"{obj_name}{sym_marker:<18}"
        
        for model_name in model_names:
            if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                add_val = all_results[model_name][obj_id]['add_mm']
                row += f"{add_val:<12.1f}"
                model_avgs[model_name].append(add_val)
            else:
                row += f"{'N/A':<12}"
        
        print(row)
    
    print("-" * 80)
    
    # Averages
    avg_row = f"{'AVERAGE':<18}"
    for model_name in model_names:
        if model_avgs[model_name]:
            avg_row += f"{np.mean(model_avgs[model_name]):<12.1f}"
        else:
            avg_row += f"{'N/A':<12}"
    print(avg_row)
    
    # Print ADD-2cm accuracy table
    print("\n" + "=" * 80)
    print("Per-Object ADD-2cm Accuracy (%) - Higher is better")
    print("=" * 80)
    print(header)
    print("-" * 80)
    
    model_acc_avgs = {name: [] for name in model_names}
    
    for obj_id in sorted(all_obj_ids):
        obj_name = OBJ_NAMES.get(obj_id, f"obj_{obj_id}")
        sym_marker = "*" if obj_id in SYMMETRIC_OBJECTS else ""
        row = f"{obj_name}{sym_marker:<18}"
        
        for model_name in model_names:
            if model_name in all_results and all_results[model_name] and obj_id in all_results[model_name]:
                acc_val = all_results[model_name][obj_id]['acc_2cm']
                row += f"{acc_val:<12.1f}"
                model_acc_avgs[model_name].append(acc_val)
            else:
                row += f"{'N/A':<12}"
        
        print(row)
    
    print("-" * 80)
    
    # Averages
    avg_row = f"{'AVERAGE':<18}"
    for model_name in model_names:
        if model_acc_avgs[model_name]:
            avg_row += f"{np.mean(model_acc_avgs[model_name]):<12.1f}"
        else:
            avg_row += f"{'N/A':<12}"
    print(avg_row)
    
    print("\n* = Symmetric object (uses ADD-S metric)")
    print("=" * 80)


if __name__ == '__main__':
    main()

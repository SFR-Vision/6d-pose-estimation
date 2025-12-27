"""Compare all 4 pose estimation models on the test set."""

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

WEIGHTS = {
    'RGB': os.path.join(PROJECT_ROOT, "weights_rgb", "best_pose_model.pth"),
    'RGB-Geometric': os.path.join(PROJECT_ROOT, "weights_rgb_geometric", "best_pose_model.pth"),
    'RGBD': os.path.join(PROJECT_ROOT, "weights_rgbd", "best_pose_model.pth"),
    'RGBD-Geometric': os.path.join(PROJECT_ROOT, "weights_rgbd_geometric", "best_pose_model.pth"),
}


def load_model(model_name, weights_path):
    """Load a model if weights exist."""
    if not os.path.exists(weights_path):
        print(f"  {model_name}: Weights not found")
        return None
    
    try:
        if model_name == 'RGB':
            from models.pose_net_rgb import PoseNetRGB
            model = PoseNetRGB(pretrained=False)
        elif model_name == 'RGB-Geometric':
            from models.pose_net_rgb_geometric import PoseNetRGBGeometric
            model = PoseNetRGBGeometric(pretrained=False)
        elif model_name == 'RGBD':
            from models.pose_net_rgbd import PoseNetRGBD
            model = PoseNetRGBD(pretrained=False)
        elif model_name == 'RGBD-Geometric':
            from models.pose_net_rgbd_geometric import PoseNetRGBDGeometric
            model = PoseNetRGBDGeometric(pretrained=False)
        else:
            return None
        
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE).eval()
        print(f"  {model_name}: Loaded")
        return model
    except Exception as e:
        print(f"  {model_name}: Error - {e}")
        return None


def evaluate_model(model, model_name, loader, criterion, is_rgbd=False, needs_geometry=False):
    """Evaluate a model on the test set."""
    if model is None:
        return None
    
    model.eval()
    all_metrics = {'add': [], 'add_s': [], 'add_01d': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}", leave=False):
            if is_rgbd:
                rgb, depth, depth_raw, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix = batch
                rgb, depth, depth_raw = rgb.to(DEVICE), depth.to(DEVICE), depth_raw.to(DEVICE)
                gt_rot, gt_trans, obj_ids = gt_rot.to(DEVICE), gt_trans.to(DEVICE), obj_ids.to(DEVICE)
                bbox_center, cam_matrix = bbox_center.to(DEVICE), cam_matrix.to(DEVICE)
                
                if 'Geometric' in model_name:
                    pred_rot, pred_trans = model(rgb, depth, depth_raw, bbox_center, cam_matrix)
                else:
                    pred_rot, pred_trans = model(rgb, depth)
            else:
                rgb, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix = batch
                rgb, gt_rot, gt_trans, obj_ids = rgb.to(DEVICE), gt_rot.to(DEVICE), gt_trans.to(DEVICE), obj_ids.to(DEVICE)
                bbox_center, cam_matrix = bbox_center.to(DEVICE), cam_matrix.to(DEVICE)
                
                if needs_geometry:
                    pred_rot, pred_trans = model(rgb, bbox_center, cam_matrix)
                else:
                    pred_rot, pred_trans = model(rgb)
            
            metrics = criterion.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
            all_metrics['add'].append(metrics['add_mean'])
            all_metrics['add_s'].append(metrics['add_s_mean'])
            all_metrics['add_01d'].append(metrics['add_01d_acc'])
    
    return {
        'ADD (mm)': np.mean(all_metrics['add']),
        'ADD-S (mm)': np.mean(all_metrics['add_s']),
        'ADD-0.1d (%)': np.mean(all_metrics['add_01d']),
    }


def main():
    print(f"\nModel Comparison on {DEVICE}\n")
    
    criterion = ADDLoss(MODEL_DIR, DEVICE)
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("Loading datasets...")
    rgb_dataset = LineMODDatasetRGB(DATA_ROOT, mode='val', transform=val_transform, augment_bbox=False)
    rgb_loader = torch.utils.data.DataLoader(rgb_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    try:
        rgbd_dataset = LineMODDatasetRGBD(DATA_ROOT, mode='val', transform=val_transform, augment_bbox=False)
        rgbd_loader = torch.utils.data.DataLoader(rgbd_dataset, batch_size=16, shuffle=False, num_workers=4)
    except:
        rgbd_loader = None
        print("  RGBD dataset not available")
    
    print(f"  {len(rgb_dataset)} test samples\n")
    
    # Load models
    print("Loading models...")
    models = {}
    for name, path in WEIGHTS.items():
        models[name] = load_model(name, path)
    print()
    
    # Evaluate
    print("Evaluating models...")
    results = {}
    
    if models['RGB'] is not None:
        results['RGB'] = evaluate_model(models['RGB'], 'RGB', rgb_loader, criterion, 
                                         is_rgbd=False, needs_geometry=False)
    
    if models['RGB-Geometric'] is not None:
        results['RGB-Geometric'] = evaluate_model(models['RGB-Geometric'], 'RGB-Geometric', rgb_loader, criterion,
                                                   is_rgbd=False, needs_geometry=True)
    
    if rgbd_loader is not None:
        if models['RGBD'] is not None:
            results['RGBD'] = evaluate_model(models['RGBD'], 'RGBD', rgbd_loader, criterion,
                                              is_rgbd=True, needs_geometry=False)
        
        if models['RGBD-Geometric'] is not None:
            results['RGBD-Geometric'] = evaluate_model(models['RGBD-Geometric'], 'RGBD-Geometric', rgbd_loader, criterion,
                                                        is_rgbd=True, needs_geometry=True)
    
    # Print results
    print("\nResults:")
    print("-" * 60)
    print(f"{'Model':<20} {'ADD (mm)':<12} {'ADD-S (mm)':<12} {'ADD-0.1d (%)':<12}")
    print("-" * 60)
    
    for name, metrics in results.items():
        if metrics is not None:
            print(f"{name:<20} {metrics['ADD (mm)']:<12.2f} {metrics['ADD-S (mm)']:<12.2f} {metrics['ADD-0.1d (%)']:<12.1f}")
        else:
            print(f"{name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("-" * 60)
    
    # Find best
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_model = min(valid_results, key=lambda x: valid_results[x]['ADD (mm)'])
        print(f"\nBest by ADD: {best_model} ({valid_results[best_model]['ADD (mm)']:.2f}mm)")
        
        best_by_acc = max(valid_results, key=lambda x: valid_results[x]['ADD-0.1d (%)'])
        print(f"Best by ADD-0.1d: {best_by_acc} ({valid_results[best_by_acc]['ADD-0.1d (%)']:.1f}%)")


if __name__ == '__main__':
    main()

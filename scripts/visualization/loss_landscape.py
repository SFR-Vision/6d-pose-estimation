"""
Loss Landscape Visualization for 6D Pose Estimation Models.

Based on: Li et al. "Visualizing the Loss Landscape of Neural Nets" (NIPS 2018)
GitHub: https://github.com/tomgoldstein/loss-landscape

Key features from the paper:
1. Filter-wise normalization - normalize each filter to have same norm as weights
2. Ignore bias and batch normalization layers (set direction to zero)
3. Save directions to file for reproducibility

Usage:
    python scripts/visualization/loss_landscape.py --model rgb --grid 25
    python scripts/visualization/loss_landscape.py --model rgbd --grid 50 --sample 0.2

WARNING: This script is computationally expensive!
    - 25x25 grid = ~625 forward passes (~30-60 min)
    - 50x50 grid = ~2500 forward passes (~2-4 hours)
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
import h5py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset_rgb import LineMODDatasetRGB
from data.dataset_rgbd import LineMODDatasetRGBD
from models.pose_net_rgb import PoseNetRGB
from models.pose_net_rgbd import PoseNetRGBD
from models.pose_loss import PoseLoss


# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model_and_loader(model_type):
    """Get model, weights path, and data loader based on model type."""
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if model_type == 'rgb':
        model = PoseNetRGB(pretrained=False)
        weights_dir = os.path.join(PROJECT_ROOT, "weights_rgb")
        dataset = LineMODDatasetRGB(DATA_ROOT, mode='train', transform=transform)
        is_rgbd = False
    elif model_type == 'rgbd':
        model = PoseNetRGBD(pretrained=False)
        weights_dir = os.path.join(PROJECT_ROOT, "weights_rgbd")
        dataset = LineMODDatasetRGBD(DATA_ROOT, mode='train', transform=transform)
        is_rgbd = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    weights_path = os.path.join(weights_dir, "best_pose_model.pth")
    history_path = os.path.join(weights_dir, "training_history.json")
    
    return model, weights_path, weights_dir, history_path, dataset, is_rgbd


def is_bias_or_bn(name, param):
    """Check if parameter is bias or batch normalization layer."""
    # Bias parameters
    if 'bias' in name.lower():
        return True
    # BatchNorm parameters (weight, bias, running_mean, running_var)
    if 'bn' in name.lower() or 'batch' in name.lower() or 'norm' in name.lower():
        return True
    # 1D parameters are usually bias
    if len(param.shape) == 1:
        return True
    return False


def get_random_direction_filter_normalized(model, ignore_biasbn=True):
    """
    Generate a random direction with filter-wise normalization.
    
    From the paper:
    - Each filter in the direction is normalized to have the same norm as 
      the corresponding filter in the pre-trained weights.
    - Bias and batch normalization layers are optionally ignored (set to zero).
    
    This ensures that the direction has similar magnitude to the original weights,
    making the visualization scale-independent.
    """
    direction = []
    
    for name, param in model.named_parameters():
        if ignore_biasbn and is_bias_or_bn(name, param):
            # Set direction to zero for bias and BN layers
            d = torch.zeros_like(param)
        else:
            # Random direction with same shape as parameter
            d = torch.randn_like(param)
            
            # Filter-wise normalization
            # For conv layers: normalize each filter (dim 0)
            # For FC layers: normalize each neuron's weights (dim 0)
            if len(param.shape) >= 2:
                # Get number of filters/neurons
                num_filters = param.shape[0]
                for f in range(num_filters):
                    # Normalize this filter to have same norm as original
                    filter_norm = param[f].norm() + 1e-10
                    d_norm = d[f].norm() + 1e-10
                    d[f] = d[f] * (filter_norm / d_norm)
            else:
                # For 1D parameters (shouldn't happen if biasbn is ignored)
                norm = param.norm() + 1e-10
                d_norm = d.norm() + 1e-10
                d = d * (norm / d_norm)
        
        direction.append(d)
    
    return direction


def get_weights_as_list(model):
    """Get model weights as a list of tensors."""
    return [param.clone() for param in model.parameters()]


def set_weights_from_direction(model, base_weights, direction1, direction2, alpha, beta):
    """
    Set model weights to: base + alpha*dir1 + beta*dir2
    
    This moves the model weights in parameter space along two random directions.
    """
    with torch.no_grad():
        for param, base, d1, d2 in zip(model.parameters(), base_weights, direction1, direction2):
            param.copy_(base + alpha * d1 + beta * d2)


def compute_loss(model, loader, criterion, is_rgbd, device, max_batches=None):
    """Compute average loss over the dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            if is_rgbd:
                # RGBD: (rgbdm_5ch, z_sensor, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix)
                rgbdm, z_sensor, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix = batch
                rgbdm = rgbdm.to(device)
                z_sensor = z_sensor.to(device)
                gt_rot = gt_rot.to(device)
                gt_trans = gt_trans.to(device)
                bbox_center = bbox_center.to(device)
                cam_matrix = cam_matrix.to(device)
                pred_rot, pred_trans = model(rgbdm, z_sensor, bbox_center, cam_matrix)
            else:
                # RGB: (rgbm_4ch, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix)
                rgbm, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix = batch
                rgbm = rgbm.to(device)
                gt_rot = gt_rot.to(device)
                gt_trans = gt_trans.to(device)
                bbox_center = bbox_center.to(device)
                cam_matrix = cam_matrix.to(device)
                pred_rot, pred_trans = model(rgbm, bbox_center, cam_matrix)
            
            loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def save_surface_to_h5(filename, xcoords, ycoords, loss_grid, direction1, direction2):
    """Save loss surface and directions to HDF5 file (following Goldstein's format)."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('xcoordinates', data=xcoords)
        f.create_dataset('ycoordinates', data=ycoords)
        f.create_dataset('train_loss', data=loss_grid)
        
        # Save directions for reproducibility
        dir1_group = f.create_group('direction1')
        for i, d in enumerate(direction1):
            dir1_group.create_dataset(f'param_{i}', data=d.cpu().numpy())
        
        dir2_group = f.create_group('direction2')
        for i, d in enumerate(direction2):
            dir2_group.create_dataset(f'param_{i}', data=d.cpu().numpy())
    
    print(f"Saved surface to: {filename}")


def create_loss_landscape(model_type, grid_size=25, range_val=1.0, sample_ratio=0.1, 
                         max_batches=50, ignore_biasbn=True):
    """
    Create a 3D loss landscape visualization using the methodology from:
    Li et al. "Visualizing the Loss Landscape of Neural Nets" (NIPS 2018)
    
    Args:
        model_type: 'rgb' or 'rgbd'
        grid_size: Number of points in each direction (grid_size x grid_size total)
        range_val: Range of alpha/beta values [-range_val, +range_val]
        sample_ratio: Fraction of training data to use (for speed)
        max_batches: Maximum batches per loss computation
        ignore_biasbn: If True, set direction to zero for bias and BN layers
    """
    print(f"\n{'='*60}")
    print(f"LOSS LANDSCAPE VISUALIZATION")
    print(f"Based on: Li et al. 'Visualizing the Loss Landscape of Neural Nets'")
    print(f"{'='*60}")
    print(f"Model: {model_type.upper()}")
    print(f"Grid: {grid_size}x{grid_size} = {grid_size**2} points")
    print(f"Range: [{-range_val}, {range_val}]")
    print(f"Ignore bias/BN: {ignore_biasbn}")
    print(f"Device: {DEVICE}")
    
    # Load model and data
    model, weights_path, weights_dir, history_path, dataset, is_rgbd = get_model_and_loader(model_type)
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
    
    # Load trained weights
    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    print(f"Loaded weights from: {weights_path}")
    
    # Sample subset of data for speed
    num_samples = int(len(dataset) * sample_ratio)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Using {num_samples} samples ({sample_ratio*100:.0f}% of training data)")
    
    # Save base weights
    base_weights = get_weights_as_list(model)
    
    # Generate filter-normalized random directions
    print("Generating filter-normalized random directions...")
    direction1 = get_random_direction_filter_normalized(model, ignore_biasbn=ignore_biasbn)
    direction2 = get_random_direction_filter_normalized(model, ignore_biasbn=ignore_biasbn)
    
    # Move directions to device
    direction1 = [d.to(DEVICE) for d in direction1]
    direction2 = [d.to(DEVICE) for d in direction2]
    
    # Loss criterion (same as training)
    criterion = PoseLoss(rot_weight=2.0, trans_weight=5.0)
    
    # Create grid
    alphas = np.linspace(-range_val, range_val, grid_size)
    betas = np.linspace(-range_val, range_val, grid_size)
    loss_grid = np.zeros((grid_size, grid_size))
    
    # Compute loss at trained weights (center point)
    center_loss = compute_loss(model, loader, criterion, is_rgbd, DEVICE, max_batches)
    print(f"Center loss (trained model): {center_loss:.4f}")
    
    # Compute loss at each grid point
    print(f"\nComputing loss at {grid_size**2} grid points...")
    total_points = grid_size * grid_size
    
    with tqdm(total=total_points, desc="Loss landscape") as pbar:
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                set_weights_from_direction(model, base_weights, direction1, direction2, alpha, beta)
                loss = compute_loss(model, loader, criterion, is_rgbd, DEVICE, max_batches)
                loss_grid[j, i] = loss  # Note: j,i for correct orientation
                pbar.update(1)
    
    # Restore original weights
    with torch.no_grad():
        for param, base in zip(model.parameters(), base_weights):
            param.copy_(base)
    
    # Save results
    output_prefix = os.path.join(weights_dir, "loss_landscape")
    
    # Save to H5 (Goldstein format)
    h5_path = f"{output_prefix}.h5"
    save_surface_to_h5(h5_path, alphas, betas, loss_grid, direction1, direction2)
    
    # Save raw data to NPY
    np.save(f"{output_prefix}_data.npy", {
        'alphas': alphas, 
        'betas': betas, 
        'loss_grid': loss_grid,
        'center_loss': center_loss
    })
    
    # Create visualization
    create_visualization(alphas, betas, loss_grid, model_type, weights_dir)
    
    print(f"\nDone! Files saved to: {weights_dir}")


def create_visualization(alphas, betas, loss_grid, model_type, output_dir):
    """Create 3D surface and 2D contour plots."""
    
    A, B = np.meshgrid(alphas, betas)
    
    # Clip extreme values for better visualization
    loss_clipped = np.clip(loss_grid, 0, np.percentile(loss_grid, 99))
    
    fig = plt.figure(figsize=(16, 6))
    
    # 3D Surface Plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(A, B, loss_clipped, cmap=cm.coolwarm, 
                            linewidth=0, antialiased=True, alpha=0.9)
    
    # Mark trained model position
    center_idx = len(alphas) // 2
    ax1.scatter([0], [0], [loss_grid[center_idx, center_idx]], 
                color='black', s=100, marker='*', label='Trained model', zorder=10)
    
    ax1.set_xlabel('Direction 1 (α)')
    ax1.set_ylabel('Direction 2 (β)')
    ax1.set_zlabel('Loss')
    ax1.set_title(f'{model_type.upper()} Loss Landscape (3D)\n(Filter-normalized directions)')
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Loss')
    
    # 2D Contour Plot
    ax2 = fig.add_subplot(122)
    
    # Log scale for better visualization
    log_loss = np.log10(loss_clipped + 1e-6)
    
    contour = ax2.contourf(A, B, log_loss, levels=50, cmap=cm.coolwarm)
    ax2.contour(A, B, log_loss, levels=20, colors='white', linewidths=0.5, alpha=0.3)
    
    # Mark trained model
    ax2.scatter([0], [0], color='black', s=150, marker='*', 
                label='Trained model', zorder=10, edgecolors='white', linewidths=2)
    
    ax2.set_xlabel('Direction 1 (α)')
    ax2.set_ylabel('Direction 2 (β)')
    ax2.set_title(f'{model_type.upper()} Loss Landscape (Contour)\n(log₁₀ scale)')
    ax2.legend()
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2, label='log₁₀(Loss)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "loss_landscape.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize loss landscape using filter-normalized random directions',
        epilog='Based on: Li et al. "Visualizing the Loss Landscape of Neural Nets" (NIPS 2018)'
    )
    parser.add_argument('--model', type=str, default='rgb',
                        choices=['rgb', 'rgbd'],
                        help='Model type to visualize')
    parser.add_argument('--grid', type=int, default=25,
                        help='Grid size (default: 25, meaning 25x25=625 points)')
    parser.add_argument('--range', type=float, default=1.0,
                        help='Range for alpha/beta values (default: 1.0)')
    parser.add_argument('--sample', type=float, default=0.1,
                        help='Fraction of training data to use (default: 0.1)')
    parser.add_argument('--batches', type=int, default=50,
                        help='Max batches per loss computation (default: 50)')
    parser.add_argument('--no-ignore-biasbn', action='store_true',
                        help='Include bias and BN layers in directions (not recommended)')
    
    args = parser.parse_args()
    
    create_loss_landscape(
        model_type=args.model,
        grid_size=args.grid,
        range_val=args.range,
        sample_ratio=args.sample,
        max_batches=args.batches,
        ignore_biasbn=not args.no_ignore_biasbn
    )


if __name__ == "__main__":
    main()

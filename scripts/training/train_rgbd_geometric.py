"""Training script for RGBD Geometric pose estimation model."""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from data.dataset_rgbd import LineMODDatasetRGBD
from models.pose_net_rgbd_geometric import PoseNetRGBDGeometric
from models.pose_loss import PoseLoss
from models.add_loss import ADDLoss

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
SAVE_DIR = os.path.join(PROJECT_ROOT, "weights_rgbd_geometric")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 48
EPOCHS = 75
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_LAST = os.path.join(SAVE_DIR, "last_pose_model.pth")
CKPT_BEST = os.path.join(SAVE_DIR, "best_pose_model.pth")


def train():
    print(f"Training RGBD Geometric model on {DEVICE}")

    # Data transforms (Normalize RGB channels for pretrained backbone)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_set = LineMODDatasetRGBD(DATA_ROOT, mode='train', transform=train_transform)
    val_set = LineMODDatasetRGBD(DATA_ROOT, mode='val', transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Train: {len(train_set)}, Val: {len(val_set)} samples")

    # Model and optimizer
    model = PoseNetRGBDGeometric(pretrained=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = PoseLoss(rot_weight=2, trans_weight=5, z_only=True)
    eval_criterion = ADDLoss(MODEL_DIR, DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    
    if os.path.exists(CKPT_LAST):
        print(f"Resuming from checkpoint: {CKPT_LAST}")
        checkpoint = torch.load(CKPT_LAST, map_location=DEVICE, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"Resumed at epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
        except:
            print("Architecture mismatch, starting fresh")
    else:
        print("Starting training from scratch")

    # Load existing history if resuming
    history_path = os.path.join(SAVE_DIR, 'training_history.json')
    if start_epoch > 0 and os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f"Loaded training history with {len(history['train_loss'])} epochs")
    else:
        history = {'train_loss': [], 'val_loss': [], 'val_add': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for rgb, depth, depth_raw, z_sensor, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in pbar:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            gt_rot = gt_rot.to(DEVICE)
            gt_trans = gt_trans.to(DEVICE)
            bbox_center = bbox_center.to(DEVICE)
            cam_matrix = cam_matrix.to(DEVICE)

            optimizer.zero_grad()
            pred_rot, pred_trans = model(rgb, depth, None, bbox_center, cam_matrix)
            loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_add_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for rgb, depth, depth_raw, z_sensor, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in val_loader:
                rgb = rgb.to(DEVICE)
                depth = depth.to(DEVICE)
                # z_sensor not needed - model predicts Z from RGB+depth features
                gt_rot = gt_rot.to(DEVICE)
                gt_trans = gt_trans.to(DEVICE)
                bbox_center = bbox_center.to(DEVICE)
                cam_matrix = cam_matrix.to(DEVICE)

                pred_rot, pred_trans = model(rgb, depth, None, bbox_center, cam_matrix)
                
                val_loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
                val_loss_sum += val_loss.item()
                
                metrics = eval_criterion.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
                val_add_sum += metrics['add_mean']
                val_acc_sum += metrics['add_2cm_acc']
                val_batches += 1
        
        avg_val_loss = val_loss_sum / val_batches
        val_add = val_add_sum / val_batches
        val_acc_2cm = val_acc_sum / val_batches
        
        scheduler.step()
        
        print(f"  Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | ADD: {val_add:.1f}mm | ACC@2cm: {val_acc_2cm:.1f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Checkpointing
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'curr_acc_2cm': val_acc_2cm
        }
        torch.save(ckpt, CKPT_LAST)
        
        if val_acc_2cm > best_acc:
            best_acc = val_acc_2cm
            ckpt['best_acc'] = best_acc
            torch.save(ckpt, CKPT_BEST)
            print(f"  New best model saved (ACC@2cm: {best_acc:.2f}%)")
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_add'].append(val_add)
        history['val_acc_2cm'].append(val_acc_2cm)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save history after each epoch
        history_path = os.path.join(SAVE_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    print(f"Training history saved to {history_path}")
    
    # Plot training curves
    from utils.training_plot import plot_training_curves
    plot_training_curves(history, SAVE_DIR, model_name="RGBD-Geometric")

    print(f"\nTraining complete. Best ACC@2cm: {best_acc:.2f}%")


if __name__ == "__main__":
    train()

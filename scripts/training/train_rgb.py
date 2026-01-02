"""Training script for RGB pose estimation model."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from data.dataset_rgb import LineMODDatasetRGB
from models.pose_net_rgb import PoseNetRGB
from models.pose_loss import PoseLoss
from models.add_loss import ADDLoss

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
SAVE_DIR = os.path.join(PROJECT_ROOT, "weights_rgb")

EPOCHS = 75
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CKPT_LAST = os.path.join(SAVE_DIR, "last_pose_model.pth")
CKPT_BEST = os.path.join(SAVE_DIR, "best_pose_model.pth")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training RGB model on {device}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_set = LineMODDatasetRGB(DATA_ROOT, mode='train', transform=train_transform)
    val_set = LineMODDatasetRGB(DATA_ROOT, mode='val', transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Train: {len(train_set)}, Val: {len(val_set)} samples")

    # Model and optimizer
    model = PoseNetRGB(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = PoseLoss(rot_weight=1.0, trans_weight=10.0, rotation_loss='geodesic')
    eval_criterion = ADDLoss(MODEL_MESH_DIR, device)

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(CKPT_LAST):
        print(f"Resuming from checkpoint: {CKPT_LAST}")
        checkpoint = torch.load(CKPT_LAST, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"Resumed at epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
        except RuntimeError:
            print("Architecture mismatch, starting fresh")
    else:
        print("Starting training from scratch")

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_add': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for rgb, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in pbar:
            rgb = rgb.to(device)
            gt_rot = gt_rot.to(device)
            gt_trans = gt_trans.to(device)

            optimizer.zero_grad()
            pred_rot, pred_trans = model(rgb)
            loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_add_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for rgb, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in val_loader:
                rgb = rgb.to(device)
                gt_rot = gt_rot.to(device)
                gt_trans = gt_trans.to(device)

                pred_rot, pred_trans = model(rgb)
                
                val_loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
                val_loss_sum += val_loss.item()
                
                metrics = eval_criterion.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
                val_add_sum += metrics['add_mean']
                val_acc_sum += metrics['add_2cm_acc']
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        val_add = val_add_sum / val_batches
        val_acc = val_acc_sum / val_batches
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | ADD: {val_add:.1f}mm | ACC: {val_acc:.1f}% | LR: {current_lr:.2e}")

        # Checkpointing
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'curr_acc': val_acc,
            'curr_add': val_add
        }
        
        torch.save(ckpt, CKPT_LAST)
        
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt['best_acc'] = best_acc
            torch.save(ckpt, CKPT_BEST)
            print(f"  New best model saved (ADD-2cm: {best_acc:.2f}%)")
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_add'].append(val_add)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

    # Save history to JSON
    history_path = os.path.join(SAVE_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Plot training curves
    from utils.training_plot import plot_training_curves
    plot_training_curves(history, SAVE_DIR, model_name="RGB")

    print(f"\nTraining complete. Best ADD-2cm: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
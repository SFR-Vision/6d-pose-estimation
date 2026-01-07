"""Training script for RGB+Mask (4-channel) Geometric pose estimation."""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json

from data.dataset_rgb import LineMODDatasetRGB
from models.pose_net_rgb import PoseNetRGB
from models.pose_loss import AutoWeightedPoseLoss
from models.add_loss import ADDLoss

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
SAVE_DIR = os.path.join(PROJECT_ROOT, "weights_rgb")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 48
EPOCHS = 75
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 5  # Learning rate warmup (Bag of Tricks)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_LAST = os.path.join(SAVE_DIR, 'last_pose_model.pth')
CKPT_BEST = os.path.join(SAVE_DIR, 'best_pose_model.pth')


def train():
    print(f"Training RGB model on {DEVICE}")

    # Data transforms with color jittering augmentation (Bag of Tricks)
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets - returns 4-channel RGBM
    train_dataset = LineMODDatasetRGB(DATA_ROOT, mode='train', transform=train_transform)
    val_dataset = LineMODDatasetRGB(DATA_ROOT, mode='val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} samples\n")

    # Model and optimizer
    model = PoseNetRGB(pretrained=True).to(DEVICE)
    criterion = AutoWeightedPoseLoss().to(DEVICE)
    
    # Optimize both model and loss parameters
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), 
                            lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate warmup + cosine annealing (Bag of Tricks)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])
    
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
        history.setdefault('val_acc_2cm', [])
        print(f"Loaded training history with {len(history['train_loss'])} epochs")
    else:
        history = {'train_loss': [], 'val_loss': [], 'val_add': [], 'val_acc': [], 'val_acc_2cm': [], 'lr': []}
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for rgbm, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in pbar:
            rgbm = rgbm.to(DEVICE)
            gt_rot = gt_rot.to(DEVICE)
            gt_trans = gt_trans.to(DEVICE)
            bbox_center = bbox_center.to(DEVICE)
            cam_matrix = cam_matrix.to(DEVICE)

            optimizer.zero_grad()
            pred_rot, pred_trans = model(rgbm, bbox_center, cam_matrix)
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
        val_acc_2cm_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for rgbm, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in val_loader:
                rgbm = rgbm.to(DEVICE)
                gt_rot = gt_rot.to(DEVICE)
                gt_trans = gt_trans.to(DEVICE)
                bbox_center = bbox_center.to(DEVICE)
                cam_matrix = cam_matrix.to(DEVICE)

                pred_rot, pred_trans = model(rgbm, bbox_center, cam_matrix)
                
                val_loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
                val_loss_sum += val_loss.item()
                
                metrics = eval_criterion.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
                val_add_sum += metrics['add_mean']
                val_acc_2cm_sum += metrics['add_2cm_acc']
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        avg_val_add = val_add_sum / val_batches
        avg_val_acc_2cm = val_acc_2cm_sum / val_batches

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_add'].append(avg_val_add)
        history['val_acc_2cm'].append(avg_val_acc_2cm)
        history['lr'].append(current_lr)
        
        # Log learned weights
        weights = criterion.get_weights()
        print(f"Epoch {epoch+1}/{EPOCHS} - Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"ADD: {avg_val_add:.1f}mm | Acc@2cm: {avg_val_acc_2cm:.1f}% | LR: {current_lr:.2e}")
        print(f"  Learned weights - Rot: {weights['rot_weight']:.3f}, Trans: {weights['trans_weight']:.3f} "
              f"(σ_rot: {weights['sigma_rot']:.3f}, σ_trans: {weights['sigma_trans']:.3f})")

        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, CKPT_LAST)

        # Save best checkpoint
        if avg_val_acc_2cm > best_acc:
            best_acc = avg_val_acc_2cm
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, CKPT_BEST)
            print(f"  → New best! ACC@2cm: {best_acc:.1f}%")

        # Save history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation ACC@2cm: {best_acc:.1f}%")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    train()

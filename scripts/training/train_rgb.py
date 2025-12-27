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
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_set = LineMODDatasetRGB(DATA_ROOT, mode='train', transform=train_transform, augment_bbox=True)
    val_set = LineMODDatasetRGB(DATA_ROOT, mode='val', transform=val_transform, augment_bbox=False)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Train: {len(train_set)}, Val: {len(val_set)} samples")

    # Model and optimizer
    model = PoseNetRGB(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7)
    
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
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, gt_rot, gt_trans, obj_ids, _, _ in pbar:
            images = images.to(device)
            gt_rot = gt_rot.to(device)
            gt_trans = gt_trans.to(device)
            
            optimizer.zero_grad()
            pred_rot, pred_trans = model(images)
            
            loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_accum += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_add_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, gt_rot, gt_trans, obj_ids, _, _ in val_loader:
                images = images.to(device)
                gt_rot = gt_rot.to(device)
                gt_trans = gt_trans.to(device)
                obj_ids = obj_ids.to(device)
                
                pred_rot, pred_trans = model(images)
                
                metrics = eval_criterion.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
                val_add_sum += metrics['add_mean']
                val_acc_sum += metrics['add_01d_acc']
                val_batches += 1

        val_add = val_add_sum / val_batches
        val_acc = val_acc_sum / val_batches
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Loss: {avg_train_loss:.4f} | ADD: {val_add:.1f}mm | ADD-0.1d: {val_acc:.1f}% | LR: {current_lr:.2e}")

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
            print(f"  New best model saved (ADD-0.1d: {best_acc:.2f}%)")

    print(f"\nTraining complete. Best ADD-0.1d: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
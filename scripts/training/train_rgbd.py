"""Training script for RGBD pose estimation model."""

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

from data.dataset_rgbd import LineMODDatasetRGBD
from models.pose_net_rgbd import PoseNetRGBD
from models.pose_loss import PoseLoss
from models.add_loss import ADDLoss

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
SAVE_DIR = os.path.join(PROJECT_ROOT, "weights_rgbd")
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 75
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_LAST = os.path.join(SAVE_DIR, 'last_pose_model.pth')
CKPT_BEST = os.path.join(SAVE_DIR, 'best_pose_model.pth')


def train():
    print(f"Training RGBD model on {DEVICE}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
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
    train_set = LineMODDatasetRGBD(DATA_ROOT, mode='train', transform=train_transform, augment_bbox=True)
    val_set = LineMODDatasetRGBD(DATA_ROOT, mode='val', transform=val_transform, augment_bbox=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Train: {len(train_set)}, Val: {len(val_set)} samples")

    # Model and optimizer
    model = PoseNetRGBD(pretrained=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    criterion = PoseLoss(rot_weight=1.0, trans_weight=10.0, rotation_loss='geodesic')
    eval_criterion = ADDLoss(MODEL_MESH_DIR, DEVICE)

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

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for rgb, depth, depth_raw, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in pbar:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            gt_rot = gt_rot.to(DEVICE)
            gt_trans = gt_trans.to(DEVICE)

            optimizer.zero_grad()
            pred_rot, pred_trans = model(rgb, depth)

            loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_add_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for rgb, depth, depth_raw, gt_rot, gt_trans, obj_ids, bbox_center, cam_matrix in val_loader:
                rgb = rgb.to(DEVICE)
                depth = depth.to(DEVICE)
                gt_rot = gt_rot.to(DEVICE)
                gt_trans = gt_trans.to(DEVICE)
                obj_ids = obj_ids.to(DEVICE)

                pred_rot, pred_trans = model(rgb, depth)

                metrics = eval_criterion.eval_metrics(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
                val_add_sum += metrics['add_mean']
                val_acc_sum += metrics['add_01d_acc']
                val_batches += 1

        val_add = val_add_sum / val_batches
        val_acc = val_acc_sum / val_batches

        scheduler.step(val_acc)

        print(f"  Loss: {avg_train_loss:.4f} | ADD: {val_add:.1f}mm | ADD-0.1d: {val_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Checkpointing
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'curr_acc': val_acc
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

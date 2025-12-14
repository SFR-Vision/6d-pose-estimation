import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time

# Custom Modules
from data.dataset import LineMODDataset
from models.pose_net import PoseNet
from models.loss import ADDLoss

# ================= CONFIGURATION =================
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_MESH_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
SAVE_DIR = os.path.join(PROJECT_ROOT, "weights")

EPOCHS = 10            # Increased for better convergence

BATCH_SIZE = 16
LEARNING_RATE = 1e-4    # Higher initial LR (scheduler will reduce it)
WEIGHT_DECAY = 1e-4     # Regularization

# Checkpoint Paths
CKPT_LAST = os.path.join(SAVE_DIR, "last_pose_model.pth")
CKPT_BEST = os.path.join(SAVE_DIR, "best_pose_model.pth")
# =================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 1. Data with Augmentation
    print("📂 Loading Datasets...")
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Occlusion Simulation (Applied AFTER normalization)
        # Reduced probability (30%) to avoid being too aggressive during training
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("   Loading training set...")
    train_set = LineMODDataset(DATA_ROOT, mode='train', transform=train_transform, augment_pose=True)
    print("   Loading validation set...")
    val_set   = LineMODDataset(DATA_ROOT, mode='val', transform=val_transform, augment_pose=False)
    
    print("   Creating data loaders...")
    # Use 4 workers for faster data loading (safe number for Windows)
    # Set to 0 if you experience hanging/freezing
    num_workers = 4
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    print(f"   ✅ Ready: {len(train_set)} train, {len(val_set)} val samples")

    # 2. Model & Optimizer
    model = PoseNet(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning Rate Scheduler (Reduce on Plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Use PURE ADD Loss (set rotation/translation weights to 0)
    criterion = ADDLoss(MODEL_MESH_DIR, device, rot_weight=0.0, trans_weight=0.0)

    # 3. AUTO-RESUME LOGIC (with Architecture Compatibility Check)
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(CKPT_LAST):
        print(f"🔄 Found checkpoint: {CKPT_LAST}")
        checkpoint = torch.load(CKPT_LAST)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"   ✅ Resuming from Epoch {start_epoch+1} (Best Loss so far: {best_val_loss:.4f})")
        except RuntimeError as e:
            print(f"   ⚠️ Checkpoint architecture mismatch!")
            print(f"   This happens when model architecture changed.")
            print(f"   Starting fresh training with NEW architecture...")
            print(f"   (Old checkpoint backed up, you can restore it if needed)")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        print("✨ No checkpoint found. Starting fresh.")

    # 4. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for images, gt_rot, gt_trans, obj_ids in pbar:
            images = images.to(device)
            gt_rot = gt_rot.to(device)
            gt_trans = gt_trans.to(device)
            obj_ids = obj_ids.to(device)
            
            optimizer.zero_grad()
            pred_rot, pred_trans = model(images)
            
            # ADD Loss (Meters)
            loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
            
            loss.backward()
            
            # Gradient Clipping (Prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'ADD Loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, gt_rot, gt_trans, obj_ids in val_loader:
                images = images.to(device)
                gt_rot = gt_rot.to(device)
                gt_trans = gt_trans.to(device)
                obj_ids = obj_ids.to(device)
                
                pred_rot, pred_trans = model(images)
                loss = criterion(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   Train ADD: {epoch_loss:.4f} m | Val ADD: {avg_val_loss:.4f} m | LR: {current_lr:.2e}")
        
        # 5. Saving Logic (Save Dicts, not just weights)
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'curr_val_loss': avg_val_loss
        }

        # Save 'Last' (Always)
        torch.save(checkpoint_data, CKPT_LAST)

        # Save 'Best' (Only if improved)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Update the best loss in the dictionary before saving
            checkpoint_data['best_val_loss'] = best_val_loss
            torch.save(checkpoint_data, CKPT_BEST)
            print(f"   ⭐ New Best Model Saved! ({avg_val_loss:.4f} m)")

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    train()
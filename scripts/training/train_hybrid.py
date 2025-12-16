import os
import sys

# Fix OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm

# Custom Modules
from data.dataset_hybrid import LineMODDatasetHybrid
from models.pose_net_hybrid import PoseNetHybrid
from models.loss import ADDLoss

# ================= CONFIGURATION =================
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "datasets", "Linemod_preprocessed", "models")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights_hybrid")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss weights (pure ADD - no separate rotation/translation weights)
ROT_WEIGHT = 0.0
TRANS_WEIGHT = 0.0

# ================= MAIN EXECUTION =================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("HYBRID POSE ESTIMATION TRAINING (ResNet50 Rotation + Geometric Translation)")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Loss Weights - Rotation: {ROT_WEIGHT} | Translation: {TRANS_WEIGHT}")
    print("="*70 + "\n")

    # ================= DATA LOADING =================
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("📦 Loading Datasets...")
    try:
        train_set = LineMODDatasetHybrid(DATA_ROOT, mode='train', transform=train_transform, augment_bbox=True)
        val_set = LineMODDatasetHybrid(DATA_ROOT, mode='val', transform=val_transform, augment_bbox=False)
        
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"✅ Train: {len(train_set)} samples | Val: {len(val_set)} samples\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # ================= MODEL INITIALIZATION =================
    model = PoseNetHybrid(pretrained=True).to(DEVICE)
    criterion = ADDLoss(MODEL_DIR, DEVICE, rot_weight=ROT_WEIGHT, trans_weight=TRANS_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    print(f"\n🚀 Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # ================= RESUME FROM CHECKPOINT =================
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    checkpoint_path = os.path.join(WEIGHTS_DIR, 'last_pose_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"📂 Found checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])
        print(f"✅ Resumed from epoch {start_epoch} (Best Val Loss: {best_val_loss:.6f})\n")
    else:
        print("🆕 Starting training from scratch\n")

    # ================= TRAINING LOOP =================
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        for rgb_batch, quat_batch, trans_batch, obj_ids, bbox_centers, cam_matrices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            rgb_batch = rgb_batch.to(DEVICE)
            quat_batch = quat_batch.to(DEVICE)
            trans_batch = trans_batch.to(DEVICE)
            obj_ids = obj_ids.to(DEVICE)
            bbox_centers = bbox_centers.to(DEVICE)
            cam_matrices = cam_matrices.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass - hybrid model with pinhole computation
            # Predicts rotation + Z from RGB only, computes X,Y geometrically
            pred_rot, pred_trans = model(rgb_batch, bbox_centers, cam_matrices)
            
            # Compute loss
            loss = criterion(pred_rot, pred_trans, quat_batch, trans_batch, obj_ids)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        z_errors = []
        
        with torch.no_grad():
            for rgb_batch, quat_batch, trans_batch, obj_ids, bbox_centers, cam_matrices in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
                rgb_batch = rgb_batch.to(DEVICE)
                quat_batch = quat_batch.to(DEVICE)
                trans_batch = trans_batch.to(DEVICE)
                obj_ids = obj_ids.to(DEVICE)
                bbox_centers = bbox_centers.to(DEVICE)
                cam_matrices = cam_matrices.to(DEVICE)
                
                pred_rot, pred_trans = model(rgb_batch, bbox_centers, cam_matrices)
                loss = criterion(pred_rot, pred_trans, quat_batch, trans_batch, obj_ids)
                val_loss += loss.item()
                
                # Track Z error separately (meters)
                z_error = torch.abs(pred_trans[:, 2] - trans_batch[:, 2]).mean().item()
                z_errors.append(z_error)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_z_error = sum(z_errors) / len(z_errors) if z_errors else 0.0
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # --- LOGGING ---
        epoch_time = time.time() - start_time
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"   Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"   Z Error: {avg_z_error*1000:.1f}mm (only learned component)")
        print(f"   Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # --- CHECKPOINTING ---
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, os.path.join(WEIGHTS_DIR, 'best_pose_model.pth'))
            print(f"   ✅ Best model saved (Val Loss: {avg_val_loss:.6f})")
        
        # Save last model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
        }, os.path.join(WEIGHTS_DIR, 'last_pose_model.pth'))
        
        print()

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Weights saved to: {WEIGHTS_DIR}")
    print("="*70)

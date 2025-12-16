import torch
from torch.utils.data import Dataset
import yaml
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class LineMODDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, img_size=224, augment_bbox=True):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.augment_bbox = augment_bbox and (mode == 'train')  # Only augment during training
        self.all_data = []
        
        # Check if path exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root dir not found: {root_dir}")

        obj_folders = [f for f in sorted(os.listdir(root_dir)) if f.isdigit()]
        
        print(f"[{mode.upper()}] Processing {len(obj_folders)} objects...")
        for obj_folder in obj_folders:
            base_path = os.path.join(root_dir, obj_folder)
            gt_path = os.path.join(base_path, 'gt.yml')
            rgb_path = os.path.join(base_path, 'rgb')
            
            if not os.path.exists(gt_path): continue
                
            with open(gt_path, 'r') as f:
                gts = yaml.safe_load(f)
            
            # Sort images to ensure interleaved split works correctly
            images = sorted([img for img in os.listdir(rgb_path) if img.endswith(".png")])
            
            for i, img_name in enumerate(images):
                frame_id = int(img_name.split('.')[0])
                
                # --- INTERLEAVED SPLIT (Same as YOLO) ---
                cycle = i % 10
                if cycle == 8:   split_name = 'val'
                elif cycle == 9: split_name = 'test'
                else:            split_name = 'train'
                
                if split_name != mode:
                    continue
                # ----------------------------------------

                if frame_id in gts:
                    for anno in gts[frame_id]:
                        if str(int(anno['obj_id'])).zfill(2) == obj_folder:
                            self.all_data.append({
                                'img_path': os.path.join(rgb_path, img_name),
                                'obj_id': int(obj_folder) - 1, # 0-indexed class ID
                                'bbox': anno['obj_bb'],
                                'cam_R_m2c': anno['cam_R_m2c'],
                                'cam_t_m2c': anno['cam_t_m2c']
                            })

        print(f"[{mode.upper()}] Loaded {len(self.all_data)} samples from {root_dir}")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        item = self.all_data[idx]
        
        # 1. Load Image
        image = cv2.imread(item['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Get Ground Truth Data
        gt_rot_mat = np.array(item['cam_R_m2c']).reshape(3, 3)
        gt_trans = np.array(item['cam_t_m2c']) # mm
        
        x, y, w, h = item['bbox']

        # --- AUGMENTATION START ---
        if self.augment_bbox:
            # Box Jitter (Simulate YOLO imperfections)
            jitter_x = int(np.random.uniform(-0.15, 0.15) * w)
            jitter_y = int(np.random.uniform(-0.15, 0.15) * h)
            scale_w = int(np.random.uniform(-0.2, 0.2) * w)
            scale_h = int(np.random.uniform(-0.2, 0.2) * h)
            
            x += jitter_x
            y += jitter_y
            w += scale_w
            h += scale_h
        # --- AUGMENTATION END ---
        
        # 3. Smart Crop (Square)
        c_x, c_y = x + w/2, y + h/2
        size = max(w, h) * 1.2
        
        x1 = int(c_x - size/2)
        y1 = int(c_y - size/2)
        
        # Pad handles out-of-bounds
        h_img, w_img, _ = image.shape
        pad_w = max(0, x1 + int(size) - w_img) + max(0, -x1)
        pad_h = max(0, y1 + int(size) - h_img) + max(0, -y1)
        
        if pad_w > 0 or pad_h > 0:
            image = cv2.copyMakeBorder(image, max(0, -y1), max(0, y1 + int(size) - h_img), 
                                       max(0, -x1), max(0, x1 + int(size) - w_img), 
                                       cv2.BORDER_CONSTANT, value=0)
            x1 += max(0, -x1)
            y1 += max(0, -y1)
            
        crop = image[y1:y1+int(size), x1:x1+int(size)]
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        
        # 4. Prepare Labels
        # Translation: mm -> meters
        translation = torch.tensor(gt_trans / 1000.0, dtype=torch.float32)
        
        # Rotation: Use scipy default order [x, y, z, w]
        r = R.from_matrix(gt_rot_mat)
        quaternion = torch.tensor(r.as_quat(), dtype=torch.float32)  # [x, y, z, w]

        # Class ID
        obj_id = torch.tensor(item['obj_id'], dtype=torch.long)
        
        if self.transform:
            crop = self.transform(crop)
            
        return crop, quaternion, translation, obj_id
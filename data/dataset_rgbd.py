import torch
from torch.utils.data import Dataset
import yaml
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class LineMODDatasetRGBD(Dataset):
    """LineMOD Dataset with RGB-D support"""
    def __init__(self, root_dir, mode='train', transform=None, img_size=224, augment_pose=True):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.augment_pose = augment_pose and (mode == 'train')
        self.all_data = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root dir not found: {root_dir}")

        obj_folders = [f for f in sorted(os.listdir(root_dir)) if f.isdigit()]
        
        print(f"[{mode.upper()}] Processing {len(obj_folders)} objects with DEPTH...")
        for obj_folder in obj_folders:
            base_path = os.path.join(root_dir, obj_folder)
            gt_path = os.path.join(base_path, 'gt.yml')
            rgb_path = os.path.join(base_path, 'rgb')
            depth_path = os.path.join(base_path, 'depth')
            
            if not os.path.exists(gt_path) or not os.path.exists(depth_path):
                continue
                
            with open(gt_path, 'r') as f:
                gts = yaml.safe_load(f)
            
            images = sorted([img for img in os.listdir(rgb_path) if img.endswith(".png")])
            
            for i, img_name in enumerate(images):
                frame_id = int(img_name.split('.')[0])
                
                # Interleaved split
                cycle = i % 10
                if cycle == 8:   split_name = 'val'
                elif cycle == 9: split_name = 'test'
                else:            split_name = 'train'
                
                if split_name != mode:
                    continue

                if frame_id in gts:
                    for anno in gts[frame_id]:
                        if str(int(anno['obj_id'])).zfill(2) == obj_folder:
                            self.all_data.append({
                                'img_path': os.path.join(rgb_path, img_name),
                                'depth_path': os.path.join(depth_path, img_name),
                                'obj_id': int(obj_folder) - 1,
                                'bbox': anno['obj_bb'],
                                'cam_R_m2c': anno['cam_R_m2c'],
                                'cam_t_m2c': anno['cam_t_m2c']
                            })

        print(f"[{mode.upper()}] Loaded {len(self.all_data)} RGB-D samples")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        item = self.all_data[idx]
        
        # 1. Load RGB Image
        rgb_image = cv2.imread(item['img_path'])
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Depth Image
        depth_image = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            depth_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint16)
        
        # 3. Ground Truth
        gt_rot_mat = np.array(item['cam_R_m2c']).reshape(3, 3)
        gt_trans = np.array(item['cam_t_m2c'])
        
        x, y, w, h = item['bbox']

        # 4. Augmentation
        if self.augment_pose:
            # Rotation noise
            noise_angles = np.random.uniform(-5, 5, 3) * np.pi / 180.0
            noise_rot = R.from_euler('xyz', noise_angles).as_matrix()
            gt_rot_mat = noise_rot @ gt_rot_mat
            
            # Translation noise
            t_noise = np.random.uniform(-20, 20, 3)
            gt_trans = gt_trans + t_noise
            
            # Box jitter
            jitter_x = int(np.random.uniform(-0.15, 0.15) * w)
            jitter_y = int(np.random.uniform(-0.15, 0.15) * h)
            scale_w = int(np.random.uniform(-0.2, 0.2) * w)
            scale_h = int(np.random.uniform(-0.2, 0.2) * h)
            
            x += jitter_x
            y += jitter_y
            w += scale_w
            h += scale_h
        
        # 5. Smart Crop (Square)
        c_x, c_y = x + w/2, y + h/2
        size = max(w, h) * 1.2
        x1 = int(c_x - size/2)
        y1 = int(c_y - size/2)
        
        # Padding
        h_img, w_img = rgb_image.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, (x1 + int(size)) - w_img)
        pad_b = max(0, (y1 + int(size)) - h_img)
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            rgb_image = cv2.copyMakeBorder(rgb_image, pad_t, pad_b, pad_l, pad_r, 
                                           cv2.BORDER_CONSTANT, value=0)
            depth_image = cv2.copyMakeBorder(depth_image, pad_t, pad_b, pad_l, pad_r,
                                            cv2.BORDER_CONSTANT, value=0)
            x1 += pad_l
            y1 += pad_t
            
        # Crop
        rgb_crop = rgb_image[y1:y1+int(size), x1:x1+int(size)]
        depth_crop = depth_image[y1:y1+int(size), x1:x1+int(size)]
        
        # Resize
        rgb_crop = cv2.resize(rgb_crop, (self.img_size, self.img_size))
        depth_crop = cv2.resize(depth_crop, (self.img_size, self.img_size))
        
        # 6. Normalize depth first (mm -> meters)
        depth_crop = depth_crop.astype(np.float32) / 1000.0  # mm to meters
        
        # 7. Apply bilateral filter to depth for noise reduction (only during training)
        if self.augment_pose and depth_crop.max() > 0:
            # bilateralFilter requires float32, which we now have
            depth_crop = cv2.bilateralFilter(depth_crop, 5, 0.1, 0.1)
        
        # 8. Normalize to [0, 1] range
        depth_crop = np.clip(depth_crop / 2.0, 0, 1)  # Assume max 2m depth
        depth_crop = depth_crop[..., np.newaxis]  # Add channel dimension
        
        # 8. Prepare labels
        translation = torch.tensor(gt_trans, dtype=torch.float32) / 1000.0
        r = R.from_matrix(gt_rot_mat)
        quaternion = torch.tensor(r.as_quat(), dtype=torch.float32)
        obj_id = torch.tensor(item['obj_id'], dtype=torch.long)
        
        # 9. Apply transforms
        if self.transform:
            rgb_crop = self.transform(rgb_crop)
        
        # Depth transform (to tensor)
        depth_crop = torch.from_numpy(depth_crop).permute(2, 0, 1).float()
        
        return rgb_crop, depth_crop, quaternion, translation, obj_id

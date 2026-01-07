"""LineMOD dataset for 4-channel (RGB+Mask) pose estimation."""

import os
import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

# Suppress OpenCV warnings
cv2.setLogLevel(3)


class LineMODDatasetRGB(Dataset):
    """
    LineMOD dataset that returns RGB+Mask as 4-channel input.
    Mask is provided as 4th channel rather than applied to RGB.
    """
    
    def __init__(self, root_dir, mode='train', transform=None, img_size=224):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.all_data = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root dir not found: {root_dir}")

        self._load_data()

    def _load_data(self):
        """Load all samples from the dataset."""
        obj_folders = [f for f in sorted(os.listdir(self.root_dir)) if f.isdigit()]
        
        for obj_folder in obj_folders:
            base_path = os.path.join(self.root_dir, obj_folder)
            gt_path = os.path.join(base_path, 'gt.yml')
            info_path = os.path.join(base_path, 'info.yml')
            rgb_path = os.path.join(base_path, 'rgb')
            
            if not os.path.exists(gt_path) or not os.path.exists(info_path):
                continue
                
            try:
                with open(gt_path, 'r') as f:
                    gts = yaml.safe_load(f)
                with open(info_path, 'r') as f:
                    infos = yaml.safe_load(f)
            except:
                continue
            
            if not gts or not infos:
                continue
            
            rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
            
            for i, img_name in enumerate(rgb_files):
                cycle = i % 10
                if (self.mode == 'val' and cycle != 8) or \
                   (self.mode == 'test' and cycle != 9) or \
                   (self.mode == 'train' and cycle >= 8):
                    continue
                
                frame_id = int(img_name.split('.')[0])
                
                if frame_id not in gts or frame_id not in infos:
                    continue
                
                if isinstance(gts[frame_id], list):
                    for anno in gts[frame_id]:
                        if str(int(anno['obj_id'])).zfill(2) == obj_folder:
                            mask_path = os.path.join(base_path, 'mask', img_name)
                            # Skip samples without masks (e.g., object 15)
                            if not os.path.exists(mask_path):
                                continue
                            self.all_data.append({
                                'img_path': os.path.join(rgb_path, img_name),
                                'mask_path': mask_path,
                                'obj_id': int(obj_folder) - 1,
                                'bbox': anno['obj_bb'],
                                'cam_R_m2c': anno['cam_R_m2c'],
                                'cam_t_m2c': anno['cam_t_m2c'],
                                'cam_K': infos[frame_id]['cam_K']
                            })

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        """
        Returns:
            rgbm_tensor: (4, H, W) - RGB + Mask as 4th channel
            quaternion: (4,) rotation as [x,y,z,w]
            translation: (3,) in meters
            obj_id: object ID
            bbox_center: (2,) center in pixels
            camera_matrix: (4,) [fx, fy, cx, cy]
        """
        item = self.all_data[idx]
        
        # Load RGB image
        rgb_image = cv2.imread(item['img_path'])
        if rgb_image is None:
            raise FileNotFoundError(f"Failed to load image: {item['img_path']}")
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Load mask (REQUIRED for this dataset - no fallback)
        if not os.path.exists(item['mask_path']):
            raise FileNotFoundError(f"Mask file not found: {item['mask_path']}")
        
        mask = cv2.imread(item['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {item['mask_path']}")
        
        # Normalize to [0, 1]
        mask = (mask > 127).astype(np.float32)
        
        # Ground truth
        gt_rot_mat = np.array(item['cam_R_m2c']).reshape(3, 3)
        gt_trans = np.array(item['cam_t_m2c'])
        
        # Validate rotation matrix
        det_R = np.linalg.det(gt_rot_mat)
        if not (0.99 < det_R < 1.01):
            raise ValueError(f"Invalid rotation matrix determinant: {det_R}")
        
        # Validate translation
        if not np.all(np.isfinite(gt_trans)):
            raise ValueError(f"Invalid translation values: {gt_trans}")
        
        # Bbox center in original image coordinates
        x, y, w, h = item['bbox']
        bbox_center_gt = torch.tensor([x + w/2, y + h/2], dtype=torch.float32)
        
        # Square crop with padding
        c_x, c_y = x + w/2, y + h/2
        size = max(w, h) * 1.2
        size = int(round(size))
        x1 = int(round(c_x - size/2))
        y1 = int(round(c_y - size/2))
        
        h_img, w_img = rgb_image.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, (x1 + size) - w_img)
        pad_b = max(0, (y1 + size) - h_img)
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            rgb_image = cv2.copyMakeBorder(rgb_image, pad_t, pad_b, pad_l, pad_r, 
                                           cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_CONSTANT, value=0)
            x1 += pad_l
            y1 += pad_t
            
        rgb_crop = rgb_image[y1:y1+size, x1:x1+size]
        mask_crop = mask[y1:y1+size, x1:x1+size]
        
        if rgb_crop.shape[0] != size or rgb_crop.shape[1] != size:
            raise RuntimeError(f"Crop size mismatch: got {rgb_crop.shape}, expected ({size}, {size})")
        
        # Resize both RGB and mask
        rgb_crop = cv2.resize(rgb_crop, (self.img_size, self.img_size))
        mask_crop = cv2.resize(mask_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Labels
        translation = torch.tensor(gt_trans, dtype=torch.float32) / 1000.0
        r = R.from_matrix(gt_rot_mat)
        quaternion = torch.tensor(r.as_quat(), dtype=torch.float32)
        obj_id = torch.tensor(item['obj_id'], dtype=torch.long)
        
        # Camera intrinsics
        cam_K = np.array(item['cam_K']).reshape(3, 3)
        camera_matrix = torch.tensor([cam_K[0,0], cam_K[1,1], cam_K[0,2], cam_K[1,2]], 
                                     dtype=torch.float32)
        
        # Convert to tensor (HWC -> CHW)
        rgb_tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_crop).unsqueeze(0).float()  # (1, H, W)
        
        # Apply transform to RGB channels only
        if self.transform is not None:
            # Transform expects 3-channel input
            rgb_tensor = self.transform(rgb_tensor)
        
        # Concatenate RGB + Mask -> 4 channels
        rgbm_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)  # (4, H, W)
        
        return rgbm_tensor, quaternion, translation, obj_id, bbox_center_gt, camera_matrix

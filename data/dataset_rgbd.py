"""LineMOD dataset for RGB-D pose estimation models."""

import os

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


class LineMODDatasetRGBD(Dataset):
    """
    LineMOD dataset with RGB, depth, and mask support (5-channel input).
    Returns: rgbdm (5ch), z_sensor, quaternion, translation, obj_id, bbox_center, camera_matrix
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
            depth_path = os.path.join(base_path, 'depth')
            
            if not os.path.exists(gt_path) or not os.path.exists(depth_path) or not os.path.exists(info_path):
                continue
                
            with open(gt_path, 'r') as f:
                gts = yaml.safe_load(f)
            
            with open(info_path, 'r') as f:
                infos = yaml.safe_load(f)
            
            images = sorted([img for img in os.listdir(rgb_path) if img.endswith(".png")])
            
            for i, img_name in enumerate(images):
                frame_id = int(img_name.split('.')[0])
                
                # Interleaved split: 80% train, 10% val, 10% test
                cycle = i % 10
                if cycle == 8:
                    split_name = 'val'
                elif cycle == 9:
                    split_name = 'test'
                else:
                    split_name = 'train'
                
                if split_name != self.mode:
                    continue

                if frame_id in gts and frame_id in infos:
                    for anno in gts[frame_id]:
                        if str(int(anno['obj_id'])).zfill(2) == obj_folder:
                            mask_path = os.path.join(base_path, 'mask', img_name)
                            # Skip samples without masks
                            if not os.path.exists(mask_path):
                                continue
                            self.all_data.append({
                                'img_path': os.path.join(rgb_path, img_name),
                                'depth_path': os.path.join(depth_path, img_name),
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
        item = self.all_data[idx]
        
        # Load RGB image
        rgb_image = cv2.imread(item['img_path'])
        if rgb_image is None:
            raise FileNotFoundError(f"Failed to load image: {item['img_path']}")
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        depth_image = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            # Fallback to zeros to keep sample usable
            depth_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint16)
        # Light denoise while preserving edges (convert to float32 for bilateral filter)
        depth_float = depth_image.astype(np.float32)
        depth_float = cv2.bilateralFilter(depth_float, 5, 75, 75)
        depth_image = depth_float.astype(np.uint16)
        
        # Load mask
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
        
        # Validate rotation matrix and translation
        det_R = np.linalg.det(gt_rot_mat)
        if not (0.99 < det_R < 1.01):
            raise ValueError(f"Invalid rotation matrix determinant: {det_R}")
        if not np.all(np.isfinite(gt_trans)):
            raise ValueError(f"Invalid translation values: {gt_trans}")
        
        # Camera intrinsics
        cam_K = np.array(item['cam_K']).reshape(3, 3).astype(np.float32)
        if cam_K.shape != (3, 3):
            raise ValueError(f"Invalid camera matrix shape: {cam_K.shape}")
        
        x, y, w, h = item['bbox']
        bbox_center_gt = np.array([x + w/2, y + h/2], dtype=np.float32)

        # Square crop with rounding
        c_x, c_y = x + w/2, y + h/2
        size = int(round(max(w, h) * 1.2))
        x1 = int(round(c_x - size/2))
        y1 = int(round(c_y - size/2))

        # Padding
        h_img, w_img = rgb_image.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, (x1 + size) - w_img)
        pad_b = max(0, (y1 + size) - h_img)

        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            rgb_image = cv2.copyMakeBorder(rgb_image, pad_t, pad_b, pad_l, pad_r, 
                                           cv2.BORDER_CONSTANT, value=0)
            depth_image = cv2.copyMakeBorder(depth_image, pad_t, pad_b, pad_l, pad_r,
                                            cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_CONSTANT, value=0)
            x1 += pad_l
            y1 += pad_t

        # Crop
        rgb_crop = rgb_image[y1:y1+size, x1:x1+size]
        depth_crop = depth_image[y1:y1+size, x1:x1+size]
        mask_crop = mask[y1:y1+size, x1:x1+size]

        if rgb_crop.shape[:2] != (size, size) or depth_crop.shape[:2] != (size, size):
            raise RuntimeError(f"Crop size mismatch: rgb {rgb_crop.shape}, depth {depth_crop.shape}, expected ({size}, {size})")

        # Use original bbox center (same as dataset_rgb.py for consistency)
        bbox_center = bbox_center_gt.copy()

        # Sample sensor Z from native-resolution crop center before any resize
        # Use mask to only sample from object pixels (not background)
        depth_native_meters = depth_crop.astype(np.float32) / 1000.0
        center_y, center_x = size // 2, size // 2
        region = 5
        y1_region = max(0, center_y - region)
        y2_region = min(size, center_y + region + 1)
        x1_region = max(0, center_x - region)
        x2_region = min(size, center_x + region + 1)
        depth_center_region = depth_native_meters[y1_region:y2_region, x1_region:x2_region]
        mask_center_region = mask_crop[y1_region:y2_region, x1_region:x2_region]
        
        # Only use depth from object pixels (mask > 0.5) that have valid depth readings
        valid_mask = (depth_center_region > 0.01) & (mask_center_region > 0.5)
        if valid_mask.sum() > 0:
            z_sensor = np.median(depth_center_region[valid_mask])
        else:
            # Fallback to any valid depth if no masked pixels
            valid_depth = depth_center_region > 0.01
            if valid_depth.sum() > 0:
                z_sensor = np.median(depth_center_region[valid_depth])
            else:
                z_sensor = 0.5  # final fallback
        z_sensor = np.clip(z_sensor, 0.3, 1.7)  # Match LineMOD depth range [0.51m-1.45m] with margin

        # Resize (use nearest for depth and mask to avoid blending metric values)
        rgb_crop = cv2.resize(rgb_crop, (self.img_size, self.img_size))
        depth_crop_resized = cv2.resize(depth_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask_crop_resized = cv2.resize(mask_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Process resized depth for CNN
        depth_crop_resized = depth_crop_resized.astype(np.float32)
        depth_raw_meters = depth_crop_resized / 1000.0

        # Normalize depth for CNN input (based on actual LineMOD depth range)
        depth_min = 0.3
        depth_max = 1.7
        depth_crop_normalized = (depth_raw_meters - depth_min) / (depth_max - depth_min)
        depth_crop_normalized = np.clip(depth_crop_normalized, 0, 1)
        depth_crop_normalized[depth_raw_meters < 0.01] = 0
        
        depth_crop_normalized = depth_crop_normalized[..., np.newaxis]

        # Labels
        translation = torch.tensor(gt_trans, dtype=torch.float32) / 1000.0
        r = R.from_matrix(gt_rot_mat)
        quaternion = torch.tensor(r.as_quat(), dtype=torch.float32)
        obj_id = torch.tensor(item['obj_id'], dtype=torch.long)

        # Convert RGB to tensor (HWC -> CHW) before applying transform
        rgb_tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms to RGB tensor only (expects Normalize, not ToTensor)
        if self.transform:
            rgb_tensor = self.transform(rgb_tensor)

        # Convert other channels to tensors
        depth_crop_tensor = torch.from_numpy(depth_crop_normalized).permute(2, 0, 1).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask_crop_resized).unsqueeze(0).float()  # (1, H, W)
        z_sensor_tensor = torch.tensor(z_sensor, dtype=torch.float32)
        bbox_center_tensor = torch.from_numpy(bbox_center).float()
        
        # Convert camera matrix to [fx, fy, cx, cy] format
        camera_matrix = torch.tensor([cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]], dtype=torch.float32)
        
        # Concatenate RGB (3) + Depth (1) + Mask (1) → 5 channels
        rgbdm_tensor = torch.cat([rgb_tensor, depth_crop_tensor, mask_tensor], dim=0)  # (5, H, W)

        return rgbdm_tensor, z_sensor_tensor, quaternion, translation, obj_id, bbox_center_tensor, camera_matrix

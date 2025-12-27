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
    LineMOD dataset with RGB and depth support.
    Returns: rgb, depth, depth_raw, quaternion, translation, obj_id, bbox_center, camera_matrix
    """
    
    def __init__(self, root_dir, mode='train', transform=None, img_size=224, augment_bbox=True):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        self.augment_bbox = augment_bbox and (mode == 'train')
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
                            self.all_data.append({
                                'img_path': os.path.join(rgb_path, img_name),
                                'depth_path': os.path.join(depth_path, img_name),
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
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        depth_image = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            depth_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint16)
        
        # Ground truth
        gt_rot_mat = np.array(item['cam_R_m2c']).reshape(3, 3)
        gt_trans = np.array(item['cam_t_m2c'])
        
        # Camera intrinsics
        cam_K = np.array(item['cam_K']).reshape(3, 3).astype(np.float32)
        
        x_orig, y_orig, w_orig, h_orig = item['bbox']
        bbox_center_gt = np.array([x_orig + w_orig/2, y_orig + h_orig/2], dtype=np.float32)
        
        x, y, w, h = x_orig, y_orig, w_orig, h_orig

        # Bbox augmentation
        if self.augment_bbox:
            jitter_x = int(np.random.uniform(-0.05, 0.05) * w)
            jitter_y = int(np.random.uniform(-0.05, 0.05) * h)
            scale_w = int(np.random.uniform(-0.1, 0.1) * w)
            scale_h = int(np.random.uniform(-0.1, 0.1) * h)
            x += jitter_x
            y += jitter_y
            w += scale_w
            h += scale_h

        # Square crop
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

        crop_size = int(size)

        # Map bbox center to crop coordinates
        center_padded_x = bbox_center_gt[0] + pad_l
        center_padded_y = bbox_center_gt[1] + pad_t
        center_crop_x = center_padded_x - x1
        center_crop_y = center_padded_y - y1
        
        center_in_crop = np.array([center_crop_x, center_crop_y], dtype=np.float32)
        scale = self.img_size / crop_size
        center_resized = center_in_crop * scale
        center_resized = np.clip(center_resized, 0, self.img_size - 1)

        # Adjust camera intrinsics for crop and resize
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        cx_crop = (cx + pad_l - x1) * scale
        cy_crop = (cy + pad_t - y1) * scale
        fx_crop = fx * scale
        fy_crop = fy * scale
        cam_K_crop = np.array([
            [fx_crop, 0, cx_crop],
            [0, fy_crop, cy_crop],
            [0, 0, 1]
        ], dtype=np.float32)

        # Resize
        rgb_crop = cv2.resize(rgb_crop, (self.img_size, self.img_size))
        depth_crop = cv2.resize(depth_crop, (self.img_size, self.img_size))

        # Process depth
        depth_crop = depth_crop.astype(np.float32)

        # Raw depth in meters (for pinhole Z computation)
        depth_raw_meters = depth_crop / 1000.0

        # Normalize depth for CNN input (global linear normalization)
        depth_min = 0.1
        depth_max = 1.6
        depth_crop_normalized = (depth_raw_meters - depth_min) / (depth_max - depth_min)
        depth_crop_normalized = np.clip(depth_crop_normalized, 0, 1)
        depth_crop_normalized[depth_raw_meters < 0.01] = 0
        
        depth_crop_normalized = depth_crop_normalized[..., np.newaxis]

        # Labels
        translation = torch.tensor(gt_trans, dtype=torch.float32) / 1000.0
        r = R.from_matrix(gt_rot_mat)
        quaternion = torch.tensor(r.as_quat(), dtype=torch.float32)
        obj_id = torch.tensor(item['obj_id'], dtype=torch.long)

        # Apply transforms
        if self.transform:
            rgb_crop = self.transform(rgb_crop)

        # Convert to tensors
        depth_crop_tensor = torch.from_numpy(depth_crop_normalized).permute(2, 0, 1).float()
        depth_raw_tensor = torch.from_numpy(depth_raw_meters).float()
        bbox_center_tensor = torch.from_numpy(center_resized).float()
        cam_K_tensor = torch.from_numpy(cam_K_crop).float()

        return rgb_crop, depth_crop_tensor, depth_raw_tensor, quaternion, translation, obj_id, bbox_center_tensor, cam_K_tensor

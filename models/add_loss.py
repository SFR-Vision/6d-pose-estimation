"""ADD (Average Distance of Model Points) loss for pose estimation evaluation."""

import os

import numpy as np
import torch
import torch.nn as nn

# LineMOD symmetric objects (eggbox and glue)
SYMMETRIC_OBJECT_IDS = {9, 10}


class ADDLoss(nn.Module):
    """
    ADD loss and evaluation metrics for 6D pose estimation.
    Supports ADD, ADD-S (for symmetric objects), and ADD-0.1d accuracy.
    """
    
    def __init__(self, model_dir, device, rot_weight=0.0, trans_weight=0.0):
        super(ADDLoss, self).__init__()
        self.points = {}
        self.diameters = {}
        self.device = device
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        
        self._load_models(model_dir)

    def _load_models(self, model_dir):
        """Load 3D object models and compute diameters."""
        # Load official diameters if available
        models_info_path = os.path.join(model_dir, "models_info.yml")
        official_diameters = {}
        
        if os.path.exists(models_info_path):
            import yaml
            with open(models_info_path, 'r') as f:
                models_info = yaml.safe_load(f)
            for obj_key, obj_data in models_info.items():
                try:
                    obj_id = int(obj_key) - 1
                    if 'diameter' in obj_data:
                        official_diameters[obj_id] = obj_data['diameter'] / 1000.0
                except:
                    pass
        
        # Load PLY files
        num_points = 500
        obj_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".ply")])
        
        for ply_file in obj_files:
            try:
                obj_id = int(ply_file.split('_')[1].split('.')[0]) - 1
            except:
                continue
                
            path = os.path.join(model_dir, ply_file)
            pts = self._load_ply(path) / 1000.0
            
            # Filter outliers
            distances = np.linalg.norm(pts, axis=1)
            pts = pts[distances < 0.5]
            
            # Get diameter
            if obj_id in official_diameters:
                diameter = official_diameters[obj_id]
            elif pts.shape[0] > 10:
                sample_pts = pts[np.random.choice(pts.shape[0], min(100, pts.shape[0]), replace=False)]
                pairwise_dist = np.linalg.norm(sample_pts[:, None] - sample_pts[None, :], axis=2)
                diameter = np.max(pairwise_dist)
            else:
                diameter = 0.1
            
            self.diameters[obj_id] = diameter
            
            # Downsample
            if pts.shape[0] > num_points:
                idx = np.random.choice(pts.shape[0], num_points, replace=False)
                pts = pts[idx]
            
            self.points[obj_id] = torch.from_numpy(pts.astype(np.float32)).to(self.device)

    def _load_ply(self, path):
        """Parse ASCII PLY file."""
        with open(path, 'r') as f:
            lines = f.readlines()
            
        verts = []
        header_end = False
        for line in lines:
            if "end_header" in line:
                header_end = True
                continue
            if header_end:
                vals = line.strip().split()
                if len(vals) >= 3:
                    verts.append([float(vals[0]), float(vals[1]), float(vals[2])])
                    
        return np.array(verts)

    def forward(self, pred_r, pred_t, gt_r, gt_t, obj_ids):
        """Compute ADD/ADD-S loss."""
        batch_size = pred_r.shape[0]
        device = pred_r.device
        
        pred_R_mat = self._quat_to_mat(pred_r)
        gt_R_mat = self._quat_to_mat(gt_r)
        
        # Group by object ID
        obj_groups = {}
        for i in range(batch_size):
            oid = int(obj_ids[i].item())
            if oid in self.points:
                if oid not in obj_groups:
                    obj_groups[oid] = []
                obj_groups[oid].append(i)
        
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for oid, indices in obj_groups.items():
            indices = torch.tensor(indices, device=device, dtype=torch.long)
            n_samples = len(indices)
            model_points = self.points[oid]
            is_symmetric = oid in SYMMETRIC_OBJECT_IDS
            
            pred_R = pred_R_mat[indices]
            gt_R = gt_R_mat[indices]
            pred_t_batch = pred_t[indices]
            gt_t_batch = gt_t[indices]
            
            gt_points = torch.matmul(model_points.unsqueeze(0), gt_R.transpose(-1, -2)) + gt_t_batch.unsqueeze(1)
            pred_points = torch.matmul(model_points.unsqueeze(0), pred_R.transpose(-1, -2)) + pred_t_batch.unsqueeze(1)
            
            if is_symmetric:
                diff = pred_points.unsqueeze(2) - gt_points.unsqueeze(1)
                pairwise_dist = torch.norm(diff, dim=3)
                min_dist = pairwise_dist.min(dim=2)[0]
                sample_losses = min_dist.mean(dim=1)
            else:
                dist = torch.norm(pred_points - gt_points, dim=2)
                sample_losses = dist.mean(dim=1)
            
            total_loss = total_loss + sample_losses.sum()
            count += n_samples
        
        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / count
    
    def train_loss(self, pred_r, pred_t, gt_r, gt_t, obj_ids):
        """Alias for forward()."""
        return self.forward(pred_r, pred_t, gt_r, gt_t, obj_ids)

    @torch.no_grad()
    def eval_metrics(self, pred_r, pred_t, gt_r, gt_t, obj_ids):
        """Compute evaluation metrics: ADD, ADD-S, ADD-0.1d."""
        batch_size = pred_r.shape[0]
        
        pred_R_mat = self._quat_to_mat(pred_r)
        gt_R_mat = self._quat_to_mat(gt_r)
        
        add_distances = []
        add_s_distances = []
        add_01d_correct = []
        
        for i in range(batch_size):
            oid = int(obj_ids[i].item())
            
            if oid not in self.points:
                continue
                
            model_points = self.points[oid]
            diameter = self.diameters.get(oid, 0.1)
            threshold = 0.1 * diameter
            
            gt_points = torch.mm(model_points, gt_R_mat[i].T) + gt_t[i]
            pred_points = torch.mm(model_points, pred_R_mat[i].T) + pred_t[i]
            
            # ADD
            add_dist = torch.norm(pred_points - gt_points, dim=1, p=2).mean()
            add_distances.append(add_dist.item())
            
            # ADD-S
            diff = pred_points.unsqueeze(1) - gt_points.unsqueeze(0)
            pairwise_dist = torch.norm(diff, dim=2)
            min_dist = pairwise_dist.min(dim=1)[0]
            add_s_dist = min_dist.mean()
            add_s_distances.append(add_s_dist.item())
            
            # ADD-0.1d
            is_symmetric = oid in SYMMETRIC_OBJECT_IDS
            effective_dist = add_s_dist if is_symmetric else add_dist
            add_01d_correct.append(1.0 if effective_dist.item() < threshold else 0.0)
        
        return {
            'add_mean': np.mean(add_distances) * 1000 if add_distances else 0,
            'add_s_mean': np.mean(add_s_distances) * 1000 if add_s_distances else 0,
            'add_01d_acc': np.mean(add_01d_correct) * 100 if add_01d_correct else 0,
        }

    def _quat_to_mat(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        x2, y2, z2 = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        row0 = torch.stack([1 - 2*y2 - 2*z2, 2*xy - 2*wz, 2*xz + 2*wy], dim=1)
        row1 = torch.stack([2*xy + 2*wz, 1 - 2*x2 - 2*z2, 2*yz - 2*wx], dim=1)
        row2 = torch.stack([2*xz - 2*wy, 2*yz + 2*wx, 1 - 2*x2 - 2*y2], dim=1)
        
        return torch.stack([row0, row1, row2], dim=1)

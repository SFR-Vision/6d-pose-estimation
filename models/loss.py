import torch
import torch.nn as nn
import os
import numpy as np

class ADDLoss(nn.Module):
    def __init__(self, model_dir, device, rot_weight=1.0, trans_weight=1.0):
        super(ADDLoss, self).__init__()
        self.points = {}
        self.device = device
        self.l1 = nn.L1Loss()
        self.rot_weight = rot_weight      # Weight for rotation component
        self.trans_weight = trans_weight  # Weight for translation component
        
        print(f"⚖️ Initializing Enhanced ADD Loss (Geometry-Aware)...")
        print(f"   Rotation Weight: {rot_weight} | Translation Weight: {trans_weight}")
        print(f"   Loading 3D Models from: {model_dir}")

        # Load .ply files for objects 1 to 15
        # We downsample points to 500 to keep training fast
        num_points = 500
        
        obj_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".ply")])
        
        for ply_file in obj_files:
            # Filename is usually 'obj_01.ply' -> ID = 0 (for class 0)
            # Adjust parsing based on your exact filename
            try:
                obj_id = int(ply_file.split('_')[1].split('.')[0]) - 1
            except:
                continue
                
            path = os.path.join(model_dir, ply_file)
            pts = self.load_ply(path)

            # 1. Convert mm to Meters
            pts = pts / 1000.0
            
            # 2. FILTER OUTLIERS (The Critical Fix)
            # Some LineMOD meshes have noise. We keep points within 0.5m radius.
            # Calculate distance from center (0,0,0)
            distances = np.linalg.norm(pts, axis=1)
            valid_mask = distances < 0.5  # Filter points further than 50cm
            pts = pts[valid_mask]
            
            # Randomly select 500 points
            if pts.shape[0] > num_points:
                idx = np.random.choice(pts.shape[0], num_points, replace=False)
                pts = pts[idx]
            
            self.points[obj_id] = torch.from_numpy(pts.astype(np.float32)).to(device)

        print(f"   Loaded {len(self.points)} object meshes.")

    def load_ply(self, path):
        # Simple PLY parser for ASCII files (LineMOD standard)
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
        """
        Enhanced loss with separate rotation and translation components
        pred_r: [B, 4] Quaternions
        pred_t: [B, 3] Translation
        obj_ids: [B] Class IDs to select the correct 3D points
        """
        total_add_loss = 0.0
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        batch_size = pred_r.shape[0]

        # Convert Quaternions to Rotation Matrices
        pred_R_mat = self.quat_to_mat(pred_r) # [B, 3, 3]
        gt_R_mat = self.quat_to_mat(gt_r)     # [B, 3, 3]

        for i in range(batch_size):
            oid = int(obj_ids[i].item())
            
            if oid not in self.points:
                continue
                
            model_points = self.points[oid] # [N, 3]
            
            # 1. Transform points with Ground Truth
            # P_gt = (R_gt * points) + t_gt
            gt_points_trans = torch.mm(model_points, gt_R_mat[i].T) + gt_t[i]
            
            # 2. Transform points with Prediction
            # P_pred = (R_pred * points) + t_pred
            pred_points_trans = torch.mm(model_points, pred_R_mat[i].T) + pred_t[i]
            
            # 3. ADD Loss: Mean distance between corresponding points
            dist = torch.norm(pred_points_trans - gt_points_trans, dim=1, p=2)
            total_add_loss += torch.mean(dist)
            
            # 4. Rotation Loss: Compare rotated points (translation removed)
            gt_points_rot = torch.mm(model_points, gt_R_mat[i].T)
            pred_points_rot = torch.mm(model_points, pred_R_mat[i].T)
            rot_dist = torch.norm(pred_points_rot - gt_points_rot, dim=1, p=2)
            total_rot_loss += torch.mean(rot_dist)
            
            # 5. Translation Loss: Direct L1 distance
            trans_dist = torch.norm(pred_t[i] - gt_t[i], p=2)
            total_trans_loss += trans_dist

        avg_add = total_add_loss / batch_size
        avg_rot = total_rot_loss / batch_size
        avg_trans = total_trans_loss / batch_size
        
        # Combined weighted loss
        combined_loss = avg_add + (self.rot_weight * avg_rot) + (self.trans_weight * avg_trans)
        
        return combined_loss

    def quat_to_mat(self, q):
        # Convert quaternion [w, x, y, z] to 3x3 rotation matrix
        # Based on standard formula
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        x2, y2, z2 = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        row0 = torch.stack([1 - 2*y2 - 2*z2, 2*xy - 2*wz, 2*xz + 2*wy], dim=1)
        row1 = torch.stack([2*xy + 2*wz, 1 - 2*x2 - 2*z2, 2*yz - 2*wx], dim=1)
        row2 = torch.stack([2*xz - 2*wy, 2*yz + 2*wx, 1 - 2*x2 - 2*y2], dim=1)
        
        return torch.stack([row0, row1, row2], dim=1) # [B, 3, 3]
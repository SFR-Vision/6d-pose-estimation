"""Fast pose loss for training using direct parameter comparison."""

import torch
import torch.nn as nn


class PoseLoss(nn.Module):
    """
    Training loss for pose estimation.
    Computes rotation loss (geodesic or L1) and translation loss (L1).
    """
    
    def __init__(self, rot_weight=1.0, trans_weight=1.0, rotation_loss='geodesic'):
        super(PoseLoss, self).__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.rotation_loss_type = rotation_loss

    def forward(self, pred_rot, pred_trans, gt_rot, gt_trans, obj_ids=None):
        """Compute combined rotation and translation loss."""
        if self.rotation_loss_type == 'geodesic':
            rot_loss = self._geodesic_distance(pred_rot, gt_rot)
        else:
            rot_loss = self._quaternion_l1(pred_rot, gt_rot)
        
        trans_loss = torch.nn.functional.l1_loss(pred_trans, gt_trans)
        
        return (self.rot_weight * rot_loss) + (self.trans_weight * trans_loss)
    
    def _geodesic_distance(self, q1, q2):
        """
        Geodesic distance between quaternions using atan2 for numerical stability.
        Avoids gradient singularity when quaternions are nearly identical.
        """
        q1 = torch.nn.functional.normalize(q1, p=2, dim=1)
        q2 = torch.nn.functional.normalize(q2, p=2, dim=1)
        
        # Handle quaternion double-cover
        dot = torch.sum(q1 * q2, dim=1, keepdim=True)
        q2 = torch.where(dot < 0, -q2, q2)
        
        # Stable geodesic using atan2
        q_diff = q1 - q2
        q_sum = q1 + q2
        
        diff_norm = torch.norm(q_diff, p=2, dim=1)
        sum_norm = torch.norm(q_sum, p=2, dim=1)
        
        angle = 2 * torch.atan2(diff_norm, sum_norm)
        return angle.mean()
    
    def _quaternion_l1(self, q1, q2):
        """L1 distance between quaternions, accounting for double-cover."""
        q1 = torch.nn.functional.normalize(q1, p=2, dim=1)
        q2 = torch.nn.functional.normalize(q2, p=2, dim=1)
        
        dist_pos = torch.sum(torch.abs(q1 - q2), dim=1)
        dist_neg = torch.sum(torch.abs(q1 + q2), dim=1)
        dist = torch.min(dist_pos, dist_neg)
        
        return dist.mean()
    
    def train_loss(self, pred_rot, pred_trans, gt_rot, gt_trans, obj_ids=None):
        """Alias for forward()."""
        return self.forward(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)

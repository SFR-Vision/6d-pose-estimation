"""Fast pose loss for training using direct parameter comparison."""

import torch
import torch.nn as nn


class PoseLoss(nn.Module):
    """
    Training loss for pose estimation.
    Rotation: geodesic quaternion distance (always).
    Translation: L1; optionally supervise only Z for geometric models.
    """
    
    def __init__(self, rot_weight=1.0, trans_weight=1.0, z_only=False):
        super(PoseLoss, self).__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.z_only = z_only

    def forward(self, pred_rot, pred_trans, gt_rot, gt_trans, obj_ids=None):
        """Compute combined rotation and translation loss."""
        rot_loss = self._geodesic_distance(pred_rot, gt_rot)
        
        if self.z_only:
            # For geometric models: X,Y derived from Z via pinhole, only supervise Z
            z_pred = pred_trans[:, 2]
            z_gt = gt_trans[:, 2]
            trans_loss = torch.nn.functional.l1_loss(z_pred, z_gt)
        else:
            # For non-geometric models: supervise full XYZ translation
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
    
    def train_loss(self, pred_rot, pred_trans, gt_rot, gt_trans, obj_ids=None):
        """Alias for forward()."""
        return self.forward(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)

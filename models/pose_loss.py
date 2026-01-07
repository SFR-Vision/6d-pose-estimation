"""Fast pose loss for training using direct parameter comparison."""

import torch
import torch.nn as nn


class PoseLoss(nn.Module):
    """
    Training loss for pose estimation with geometric X,Y.
    Rotation: geodesic quaternion distance.
    Translation: L1 loss on Z-depth only (X,Y computed geometrically, not supervised).
    """
    
    def __init__(self, rot_weight=1.0, trans_weight=1.0):
        super(PoseLoss, self).__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def forward(self, pred_rot, pred_trans, gt_rot, gt_trans, obj_ids=None):
        """Compute combined rotation and translation loss."""
        rot_loss = self._geodesic_distance(pred_rot, gt_rot)
        
        # Only supervise Z-depth (X,Y are computed geometrically from Z)
        z_pred = pred_trans[:, 2]
        z_gt = gt_trans[:, 2]
        trans_loss = torch.nn.functional.l1_loss(z_pred, z_gt)
        
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


class AutoWeightedPoseLoss(nn.Module):
    """
    Automatic weighted pose loss using homoscedastic uncertainty.
    Based on: Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses 
    for Scene Geometry and Semantics" (CVPR 2018)
    
    Learns optimal weights for rotation and translation losses by modeling task-dependent 
    uncertainty. The network learns log(σ²) parameters that automatically balance the 
    two loss components during training.
    
    Formula: L = L_rot/(2σ_rot²) + L_trans/(2σ_trans²) + log(σ_rot) + log(σ_trans)
    
    Benefits:
    - No manual hyperparameter tuning of rot_weight/trans_weight
    - Automatically adapts to different loss scales (radians vs meters)
    - Better gradient balance → improved convergence → higher accuracy
    """
    
    def __init__(self):
        super(AutoWeightedPoseLoss, self).__init__()
        # Learnable log-variance parameters (initialized to log(1) = 0)
        # log(σ²) formulation ensures σ² stays positive
        self.log_var_rot = nn.Parameter(torch.zeros(1))
        self.log_var_trans = nn.Parameter(torch.zeros(1))
    
    def forward(self, pred_rot, pred_trans, gt_rot, gt_trans, obj_ids=None):
        """
        Compute automatically weighted loss.
        
        Returns:
            loss: Combined weighted loss
        """
        # Compute individual losses
        rot_loss = self._geodesic_distance(pred_rot, gt_rot)
        
        # Only supervise Z-depth (X,Y are computed geometrically from Z)
        z_pred = pred_trans[:, 2]
        z_gt = gt_trans[:, 2]
        trans_loss = torch.nn.functional.l1_loss(z_pred, z_gt)
        
        # Compute precisions (inverse variance)
        precision_rot = torch.exp(-self.log_var_rot)
        precision_trans = torch.exp(-self.log_var_trans)
        
        # Weighted loss with regularization (canonical formula: 0.5 factor on all terms)
        loss = (
            0.5 * precision_rot * rot_loss +
            0.5 * precision_trans * trans_loss +
            0.5 * (self.log_var_rot + self.log_var_trans)
        )
        return loss
    
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
    
    def get_weights(self):
        """
        Get current learned weights for logging/monitoring.
        
        Returns:
            dict: {'rot_weight': float, 'trans_weight': float, 'sigma_rot': float, 'sigma_trans': float}
        """
        with torch.no_grad():
            sigma_rot = torch.exp(0.5 * self.log_var_rot).item()
            sigma_trans = torch.exp(0.5 * self.log_var_trans).item()
            # Effective weights (higher when σ is smaller)
            weight_rot = torch.exp(-self.log_var_rot).item()
            weight_trans = torch.exp(-self.log_var_trans).item()
        
        return {
            'rot_weight': weight_rot,
            'trans_weight': weight_trans,
            'sigma_rot': sigma_rot,
            'sigma_trans': sigma_trans
        }
    
    def train_loss(self, pred_rot, pred_trans, gt_rot, gt_trans, obj_ids=None):
        """Alias for forward()."""
        return self.forward(pred_rot, pred_trans, gt_rot, gt_trans, obj_ids)

"""RGB-D pose estimation with geometric translation from depth sensor.

Simplified architecture:
- Single RGB backbone for rotation prediction
- Direct depth sensor reading for geometric translation (pinhole model)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNetRGBDGeometric(nn.Module):
    """
    RGB-D pose estimation with geometric translation.
    Uses RGB backbone for rotation; computes translation directly from depth sensor.
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGBDGeometric, self).__init__()
        
        # Single RGB backbone for rotation
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Rotation head
        self.rot_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)
        )

    def forward(self, rgb, depth=None, depth_raw=None, bbox_center=None, camera_matrix=None):
        """Forward pass: RGB -> rotation, depth sensor -> translation."""
        # Rotation from RGB backbone
        features = self.backbone(rgb).view(rgb.size(0), -1)
        rotation = self.rot_head(features)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        # Translation from depth sensor (geometric)
        if depth_raw is not None and bbox_center is not None and camera_matrix is not None:
            translation = self._compute_pinhole_translation(depth_raw, bbox_center, camera_matrix)
        else:
            translation = torch.zeros(rgb.size(0), 3, device=rgb.device)
            translation[:, 2] = 0.5
        
        return rotation, translation
    
    def _compute_pinhole_translation(self, depth_raw, bbox_center, camera_matrix):
        """Compute X, Y, Z using pinhole camera model and depth sensor readings."""
        batch_size = depth_raw.size(0)
        device = depth_raw.device
        
        if camera_matrix.dim() == 2:
            camera_matrix = camera_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        fx = camera_matrix[:, 0, 0]
        fy = camera_matrix[:, 1, 1]
        cx = camera_matrix[:, 0, 2]
        cy = camera_matrix[:, 1, 2]
        
        u_crop = bbox_center[:, 0].clamp(0, 223)
        v_crop = bbox_center[:, 1].clamp(0, 223)
        
        # Sample Z at bbox center
        u_idx = u_crop.long().clamp(0, 223)
        v_idx = v_crop.long().clamp(0, 223)
        z = depth_raw[torch.arange(batch_size, device=device), v_idx, u_idx]
        
        # Handle zero/invalid depths
        z = torch.where(z > 0.01, z, torch.tensor(0.5, device=device))
        z = torch.clamp(z, min=0.1, max=2.0)
        
        # Apply pinhole equations
        x = (u_crop - cx) * z / fx
        y = (v_crop - cy) * z / fy
        
        return torch.stack([x, y, z], dim=1)


if __name__ == "__main__":
    model = PoseNetRGBDGeometric()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    rgb = torch.randn(2, 3, 224, 224)
    depth_raw = torch.rand(2, 224, 224) * 1.5
    bbox_center = torch.tensor([[112, 112], [100, 120]], dtype=torch.float32)
    cam_K = torch.tensor([[[500, 0, 112], [0, 500, 112], [0, 0, 1]]], dtype=torch.float32).expand(2, -1, -1)
    
    rot, trans = model(rgb, None, depth_raw, bbox_center, cam_K)
    print(f"Rotation: {rot.shape}, Translation: {trans.shape}")

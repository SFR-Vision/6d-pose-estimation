"""RGB-D pose estimation with learned Z offset from depth.

Improved architecture:
- RGB backbone for rotation prediction
- Small CNN processes depth image to learn surface-to-centroid Z offset
- Combines sensor depth with learned offset for accurate Z
- Uses pinhole model for X, Y translation
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNetRGBDGeometric(nn.Module):
    """
    RGB-D pose estimation with learned Z offset.
    Uses RGB backbone for rotation; learns Z offset from depth to correct
    sensor depth (surface) to object centroid depth.
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGBDGeometric, self).__init__()
        
        # RGB backbone for rotation
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
        
        # Stronger CNN for Z offset prediction from depth crop
        # Learns: Z_centroid = Z_sensor + offset
        self.z_backbone = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Z offset predictor - predicts the correction from surface to centroid
        self.z_predictor = nn.Sequential(
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, rgb, depth=None, z_sensor=None, bbox_center=None, camera_matrix=None):
        """Forward pass: RGB -> rotation, depth + CNN -> corrected translation."""
        # Rotation from RGB backbone
        features = self.backbone(rgb).view(rgb.size(0), -1)
        rotation = self.rot_head(features)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        # Translation with learned Z offset
        if depth is not None and z_sensor is not None and bbox_center is not None and camera_matrix is not None:
            translation = self._compute_corrected_translation(depth, z_sensor, bbox_center, camera_matrix)
        else:
            translation = torch.zeros(rgb.size(0), 3, device=rgb.device)
            translation[:, 2] = 0.5
        
        return rotation, translation
    
    def _compute_corrected_translation(self, depth, z_sensor, bbox_center, camera_matrix):
        """Compute translation with learned Z offset from depth CNN.
        
        Args:
            depth: Normalized depth crop [B, 1, 224, 224] for CNN input
            z_sensor: Pre-computed sensor Z from native crop [B] in meters
            bbox_center: Original image bbox center [B, 2]
            camera_matrix: Camera intrinsics [B, 3, 3]
        """
        batch_size = z_sensor.size(0)
        device = z_sensor.device
        
        if camera_matrix.dim() == 2:
            camera_matrix = camera_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        fx = camera_matrix[:, 0, 0]
        fy = camera_matrix[:, 1, 1]
        cx = camera_matrix[:, 0, 2]
        cy = camera_matrix[:, 1, 2]
        
        # Learn Z offset from depth image (surface -> centroid correction)
        z_features = self.z_backbone(depth)
        z_offset = self.z_predictor(z_features).squeeze(-1)  # [B]
        
        # Corrected Z = sensor Z + learned offset
        # Offset can be positive (centroid behind surface) or negative
        z_corrected = z_sensor + z_offset
        z_corrected = torch.clamp(z_corrected, min=0.1, max=2.0)
        
        # Apply pinhole equations for X, Y
        u_orig = bbox_center[:, 0]
        v_orig = bbox_center[:, 1]
        x = (u_orig - cx) * z_corrected / fx
        y = (v_orig - cy) * z_corrected / fy
        
        return torch.stack([x, y, z_corrected], dim=1)


if __name__ == "__main__":
    model = PoseNetRGBDGeometric()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.rand(2, 1, 224, 224)  # Normalized depth for CNN
    depth_raw = torch.rand(2, 224, 224) * 1.5  # Raw depth in meters
    bbox_center = torch.tensor([[300, 250], [280, 230]], dtype=torch.float32)
    cam_K = torch.tensor([[[572.4, 0, 325.2], [0, 573.5, 242.0], [0, 0, 1]]], dtype=torch.float32).expand(2, -1, -1)
    
    rot, trans = model(rgb, depth, depth_raw, bbox_center, cam_K)
    print(f"Rotation: {rot.shape}, Translation: {trans.shape}")
    print(f"Trans values: {trans}")

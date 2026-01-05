"""RGB-D pose estimation with dual-stream fusion.

Architecture:
- RGB backbone extracts appearance features
- Depth CNN extracts geometric features
- Fused features predict rotation (from both RGB + depth)
- Fused features predict Z (from both RGB + depth)
- X, Y computed geometrically using pinhole model
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNetRGBDGeometric(nn.Module):
    """
    RGB-D dual-stream pose estimation.
    Fuses RGB and depth features to predict rotation and Z translation.
    Computes X, Y geometrically from predicted Z.
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGBDGeometric, self).__init__()
        
        # RGB stream - ResNet50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Depth stream - CNN for geometric features
        self.depth_backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Fusion and rotation head - takes concatenated RGB+depth features
        self.rot_head = nn.Sequential(
            nn.Linear(2048 + 512, 1024),  # RGB (2048) + Depth (512)
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)  # Quaternion
        )
        
        # Z prediction head - also uses fused features
        self.z_predictor = nn.Sequential(
            nn.Linear(2048 + 512, 512),  # RGB (2048) + Depth (512)
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Z depth
        )

    def forward(self, rgb, depth=None, z_sensor=None, bbox_center=None, camera_matrix=None):
        """Forward pass: fuse RGB+depth for rotation and Z, compute X,Y geometrically."""
        # Extract features from both streams
        rgb_features = self.rgb_backbone(rgb).view(rgb.size(0), -1)  # [B, 2048]
        depth_features = self.depth_backbone(depth).view(depth.size(0), -1)  # [B, 512]
        
        # Fuse features
        fused_features = torch.cat([rgb_features, depth_features], dim=1)  # [B, 2560]
        
        # Predict rotation from fused features
        rotation = self.rot_head(fused_features)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        # Predict translation
        if bbox_center is not None and camera_matrix is not None:
            translation = self._compute_translation(fused_features, bbox_center, camera_matrix)
        else:
            translation = torch.zeros(rgb.size(0), 3, device=rgb.device)
            translation[:, 2] = 0.5
        
        return rotation, translation
    
    def _compute_translation(self, fused_features, bbox_center, camera_matrix):
        """Compute translation: predict Z from fused features, compute X,Y geometrically.
        
        Args:
            fused_features: Concatenated RGB+depth features [B, 2560]
            bbox_center: Original image bbox center [B, 2]
            camera_matrix: Camera intrinsics [B, 3, 3]
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        
        if camera_matrix.dim() == 2:
            camera_matrix = camera_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        fx = camera_matrix[:, 0, 0]
        fy = camera_matrix[:, 1, 1]
        cx = camera_matrix[:, 0, 2]
        cy = camera_matrix[:, 1, 2]
        
        # Predict Z from fused RGB+depth features
        z_pred = self.z_predictor(fused_features).squeeze(-1)  # [B]
        z_pred = torch.clamp(z_pred, min=0.1, max=2.0)
        
        # Apply pinhole equations for X, Y using predicted Z
        u_orig = bbox_center[:, 0]
        v_orig = bbox_center[:, 1]
        x = (u_orig - cx) * z_pred / fx
        y = (v_orig - cy) * z_pred / fy
        
        return torch.stack([x, y, z_pred], dim=1)


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

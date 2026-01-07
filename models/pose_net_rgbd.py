"""RGB-D-Mask pose estimation with 5-channel input (RGB + Depth + Mask).

Architecture:
- 5-channel input: RGB (3) + Depth (1) + Mask (1)
- Dual-stream fusion: RGB+Mask backbone + Depth+Mask backbone
- Fused features predict rotation and Z-offset from sensor depth
- Final Z = z_sensor + z_offset (leverages depth measurement as prior)
- X, Y computed geometrically from predicted Z using pinhole model
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNetRGBD(nn.Module):
    """
    RGB-D-Mask dual-stream pose estimation with geometric translation.
    Takes 5 channels: RGB (3) + Depth (1) + Mask (1).
    Learns rotation and Z-depth from fused RGB+depth features with mask guidance.
    Computes X,Y translation geometrically from predicted Z using pinhole projection.
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGBD, self).__init__()
        
        # RGB+Mask stream (4 channels) - modified ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Modify first conv to accept 4 channels (RGB + Mask)
        original_conv1 = resnet.conv1
        self.rgb_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                # Copy RGB weights
                self.rgb_conv1.weight[:, :3, :, :] = original_conv1.weight
                # Initialize mask channel
                self.rgb_conv1.weight[:, 3:4, :, :] = torch.randn_like(self.rgb_conv1.weight[:, 3:4, :, :]) * 0.01
        
        # Build RGB+Mask backbone
        self.rgb_backbone = nn.Sequential(
            self.rgb_conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten()
        )
        
        # Depth+Mask stream (2 channels) - CNN for geometric features
        self.depth_backbone = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),  # Depth + Mask
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Rotation head - fuses RGB+Mask (2048) + Depth+Mask (512) features
        self.rot_head = nn.Sequential(
            nn.Linear(2048 + 512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)  # Quaternion [x, y, z, w]
        )
        
        # Z-offset prediction head - predicts offset from sensor depth to object center
        # This makes learning easier: model predicts small correction, not absolute depth
        self.z_head = nn.Sequential(
            nn.Linear(2048 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Z-offset in meters (can be positive or negative)
        )

    def forward(self, rgbdm, z_sensor, bbox_center, camera_matrix):
        """
        Forward pass: predict rotation and Z-depth, compute X,Y geometrically.
        
        Args:
            rgbdm: (B, 5, H, W) - RGB (3ch) + Depth (1ch) + Mask (1ch)
            z_sensor: (B,) - Ground truth depth at bbox center (for supervision only)
            bbox_center: (B, 2) - Object center in pixels [u, v]
            camera_matrix: (B, 4) - [fx, fy, cx, cy]
        
        Returns:
            rotation: (B, 4) - Quaternion [x, y, z, w]
            translation: (B, 3) - [X, Y, Z] in meters (Z predicted, X,Y geometric)
        """
        batch_size = rgbdm.size(0)
        
        # Split channels
        rgb = rgbdm[:, :3, :, :]      # RGB (3 channels)
        depth = rgbdm[:, 3:4, :, :]   # Depth (1 channel)
        mask = rgbdm[:, 4:5, :, :]    # Mask (1 channel)
        
        # RGB+Mask stream
        rgbm = torch.cat([rgb, mask], dim=1)  # (B, 4, H, W)
        rgb_features = self.rgb_backbone(rgbm)  # (B, 2048)
        
        # Depth+Mask stream
        depthm = torch.cat([depth, mask], dim=1)  # (B, 2, H, W)
        depth_features = self.depth_backbone(depthm)  # (B, 512)
        
        # Fuse features
        fused_features = torch.cat([rgb_features, depth_features], dim=1)  # (B, 2560)
        
        # Predict rotation from fused features
        rotation = self.rot_head(fused_features)  # (B, 4)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        # Predict Z-offset from fused features (offset from sensor surface depth to object center)
        z_offset = self.z_head(fused_features).squeeze(1)  # (B,) - can be positive or negative
        
        # Final depth = sensor reading + learned offset
        # z_sensor is surface depth, offset adjusts to object center
        z_pred = z_sensor + z_offset
        z_pred = torch.clamp(z_pred, min=0.1, max=3.0)  # Ensure valid depth range
        
        # Geometric translation from predicted Z
        translation = self._compute_translation_geometric(z_pred, bbox_center, camera_matrix)
        
        return rotation, translation
    
    def _compute_translation_geometric(self, z_pred, bbox_center, camera_matrix):
        """
        Compute translation using predicted depth (fully geometric, no additional learning).
        
        Args:
            z_pred: (B,) - Predicted depth at object center (meters)
            bbox_center: (B, 2) - Object center [u, v] in pixels
            camera_matrix: (B, 4) - [fx, fy, cx, cy]
        
        Returns:
            translation: (B, 3) - [X, Y, Z] in meters
        """
        fx = camera_matrix[:, 0]
        fy = camera_matrix[:, 1]
        cx = camera_matrix[:, 2]
        cy = camera_matrix[:, 3]
        
        u = bbox_center[:, 0]
        v = bbox_center[:, 1]
        
        # Pinhole projection with predicted depth
        X = (u - cx) * z_pred / fx
        Y = (v - cy) * z_pred / fy
        Z = z_pred
        
        return torch.stack([X, Y, Z], dim=1)  # (B, 3)


"""RGB+Mask pose estimation with geometric translation using pinhole camera model."""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNetRGB(nn.Module):
    """
    Pose estimation model using RGBM (4-channel) input with geometric translation.
    Accepts RGB + Mask as 4th channel. Learns rotation and Z-depth; computes X, Y using pinhole camera model.
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGB, self).__init__()
        
        # Load pretrained ResNet50 for RGB channels
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Modify first conv layer to accept 4 channels (RGB + Mask)
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            # Copy pretrained weights for RGB channels
            with torch.no_grad():
                self.conv1.weight[:, :3, :, :] = original_conv1.weight
                # Initialize mask channel weights (4th channel) with small random values
                self.conv1.weight[:, 3:4, :, :] = torch.randn_like(self.conv1.weight[:, 3:4, :, :]) * 0.01
        
        # Build RGB backbone with modified conv1
        self.rgb_backbone = nn.Sequential(
            self.conv1,
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
        
        # Lightweight CNN for Z-depth prediction (also takes 4 channels)
        self.z_backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Z-depth predictor
        self.z_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, rgbm, bbox_center, camera_matrix):
        """
        Forward pass with geometric translation computation.
        
        Args:
            rgbm: (B, 4, H, W) - RGB + Mask as 4th channel
            bbox_center: (B, 2) - object center in pixels [cx, cy]
            camera_matrix: (B, 4) - [fx, fy, cx, cy]
        
        Returns:
            pred_rot: (B, 4) - quaternion [x, y, z, w]
            pred_trans: (B, 3) - translation [X, Y, Z] in meters
        """
        B = rgbm.size(0)
        
        # Predict rotation from RGB+Mask features
        rgb_features = self.rgb_backbone(rgbm)  # (B, 2048)
        pred_rot = self.rot_head(rgb_features)  # (B, 4)
        
        # Normalize quaternion
        pred_rot = pred_rot / (torch.norm(pred_rot, dim=1, keepdim=True) + 1e-8)
        
        # Predict Z-depth from RGB+Mask
        z_features = self.z_backbone(rgbm)  # (B, 256)
        z_norm = self.z_predictor(z_features).squeeze(-1)  # (B,)
        pred_Z = z_norm * 1.5  # Scale to [0, 1.5] meters (LineMOD range)
        
        # Geometric translation using pinhole camera model
        fx = camera_matrix[:, 0]
        fy = camera_matrix[:, 1]
        cx = camera_matrix[:, 2]
        cy = camera_matrix[:, 3]
        
        u = bbox_center[:, 0]
        v = bbox_center[:, 1]
        
        pred_X = (u - cx) * pred_Z / fx
        pred_Y = (v - cy) * pred_Z / fy
        
        pred_trans = torch.stack([pred_X, pred_Y, pred_Z], dim=1)  # (B, 3)
        
        return pred_rot, pred_trans

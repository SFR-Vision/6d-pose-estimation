import torch
import torch.nn as nn
import torchvision.models as models

class PoseNetHybrid(nn.Module):
    """
    Hybrid Pose Estimation Model (RGB-only input):
    - ResNet50 for rotation prediction (learned from RGB)
    - Z-depth predictor (learned from RGB features)
    - X,Y translation computed geometrically using pinhole camera model
    """
    def __init__(self, pretrained=True):
        super(PoseNetHybrid, self).__init__()
        
        print("🔧 Initializing Hybrid PoseNet (RGB-only: Learned Z + Geometric X,Y)...")
        
        # 1. Shared RGB Backbone: ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Rotation Head (from RGB features)
        self.rot_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)  # Quaternion (x, y, z, w) - scipy order
        )
        
        # 3. Custom Lightweight CNN for Z-depth prediction (from RGB)
        # Built from scratch to predict distance from camera
        self.z_backbone = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56x32
            
            # Block 2: 56x56x32 -> 28x28x64
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28x64
            
            # Block 3: 28x28x64 -> 14x14x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x128
            
            # Block 4: 14x14x128 -> 7x7x256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x256
            
            # Global Average Pooling: 7x7x256 -> 256
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 4. Z-Depth Predictor (from custom CNN features)
        self.z_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: Z distance only (in meters)
        )
        
        # Initialization - start around 0.5m depth (typical LineMOD distance)
        self.z_predictor[-1].bias.data.fill_(0.5)

    def forward(self, rgb, bbox_center=None, camera_matrix=None):
        """
        Args:
            rgb: [B, 3, 224, 224] RGB image (ONLY INPUT)
            bbox_center: [B, 2] Bounding box center (u, v) in pixels (required for geometric X,Y)
            camera_matrix: [B, 3, 3] Camera intrinsics (required for geometric X,Y)
        """
        # 1. Rotation: Extract features from ResNet50
        rgb_features = self.rgb_backbone(rgb).view(rgb.size(0), -1)  # [B, 2048]
        rotation = self.rot_head(rgb_features)  # [B, 4]
        
        # 2. Z-depth: Extract features from custom CNN
        z_features = self.z_backbone(rgb).view(rgb.size(0), -1)  # [B, 256]
        z_pred = self.z_predictor(z_features)  # [B, 1]
        
        # 4. Compute X, Y using Pinhole Camera Model (Geometric - NOT learned)
        if bbox_center is not None and camera_matrix is not None:
            translation = self.compute_pinhole_translation(z_pred, bbox_center, camera_matrix)
        else:
            # Fallback: if no camera info, just use [0, 0, Z]
            translation = torch.cat([
                torch.zeros_like(z_pred),  # X - placeholder
                torch.zeros_like(z_pred),  # Y - placeholder
                z_pred  # Z - predicted
            ], dim=1)
        
        # Normalize quaternion
        rotation = rotation / (torch.norm(rotation, dim=1, keepdim=True) + 1e-8)
        
        return rotation, translation

    def compute_pinhole_translation(self, z_pred, bbox_center, camera_matrix):
        """
        Compute X, Y translation using pinhole camera model given Z.
        
        Pinhole equations (inverse projection):
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        Args:
            z_pred: [B, 1] Predicted Z distance in meters
            bbox_center: [B, 2] Bounding box center (u, v) in pixels
            camera_matrix: [B, 3, 3] or [3, 3] Camera intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        
        Returns:
            translation: [B, 3] Full translation (X, Y, Z) in meters
        """
        # Handle case where camera_matrix might not have batch dimension
        if camera_matrix.dim() == 2:
            camera_matrix = camera_matrix.unsqueeze(0).expand(z_pred.size(0), -1, -1)
        
        # Extract camera parameters
        fx = camera_matrix[:, 0, 0].unsqueeze(1)  # [B, 1]
        fy = camera_matrix[:, 1, 1].unsqueeze(1)  # [B, 1]
        cx = camera_matrix[:, 0, 2].unsqueeze(1)  # [B, 1]
        cy = camera_matrix[:, 1, 2].unsqueeze(1)  # [B, 1]
        
        # Bbox center in pixels
        u = bbox_center[:, 0].unsqueeze(1)  # [B, 1]
        v = bbox_center[:, 1].unsqueeze(1)  # [B, 1]
        
        # Apply pinhole equations
        x_pred = (u - cx) * z_pred / fx
        y_pred = (v - cy) * z_pred / fy
        
        translation = torch.cat([x_pred, y_pred, z_pred], dim=1)  # [B, 3]
        return translation

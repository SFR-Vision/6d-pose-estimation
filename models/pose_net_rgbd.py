import torch
import torch.nn as nn
import torchvision.models as models

class PoseNetRGBD(nn.Module):
    """
    Enhanced PoseNet with RGB-D fusion
    Uses both RGB and Depth for better accuracy
    """
    def __init__(self, pretrained=True):
        super(PoseNetRGBD, self).__init__()
        
        print("🔧 Initializing RGB-D PoseNet...")
        
        # 1. RGB Backbone: ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet_rgb = models.resnet50(weights=weights)
        self.rgb_backbone = nn.Sequential(*list(resnet_rgb.children())[:-1])
        
        # 2. Depth Backbone: ResNet50 (upgraded for better accuracy)
        resnet_depth = models.resnet50(weights=None)  # No pretrained weights for depth
        # Modify first conv to accept 1-channel depth input
        resnet_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_backbone = nn.Sequential(*list(resnet_depth.children())[:-1])
        
        # 3. Fusion Layer (Simple)
        rgb_dim = 2048  # ResNet50
        depth_dim = 2048  # ResNet50 (upgraded from ResNet18)
        fused_dim = 2048
        
        self.fusion = nn.Sequential(
            nn.Linear(rgb_dim + depth_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 4. Rotation Head (Moderate depth)
        self.rot_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # Quaternion
        )
        
        # 5. Translation Head (Moderate depth)
        self.trans_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # Translation
        )
        
        # Initialization
        self.trans_head[-1].bias.data.fill_(0)
        self.trans_head[-1].bias.data[2] = 0.5

    def forward(self, rgb, depth):
        """
        Args:
            rgb: [B, 3, 224, 224] RGB image
            depth: [B, 1, 224, 224] Depth image (normalized)
        """
        # Extract features
        rgb_features = self.rgb_backbone(rgb).view(rgb.size(0), -1)
        depth_features = self.depth_backbone(depth).view(depth.size(0), -1)
        
        # Fuse RGB and Depth (simple concatenation)
        fused = self.fusion(torch.cat([rgb_features, depth_features], dim=1))
        
        # Predict pose
        rotation = self.rot_head(fused)
        translation = self.trans_head(fused)
        
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        return rotation, translation

if __name__ == "__main__":
    # Test
    model = PoseNetRGBD()
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)
    rot, trans = model(rgb, depth)
    print(f"Rotation: {rot.shape}, Translation: {trans.shape}")

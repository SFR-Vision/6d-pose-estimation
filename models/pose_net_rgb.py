"""RGB-only pose estimation network using ResNet50 backbone."""

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNetRGB(nn.Module):
    """
    Pose estimation model using RGB input.
    Predicts rotation (quaternion) and translation (x, y, z).
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGB, self).__init__()
        
        # Backbone: ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Rotation head (outputs 4-dim quaternion)
        self.rot_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        
        # Translation head (outputs x, y, z in meters)
        self.trans_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
        # Initialize translation bias to typical depth
        self.trans_head[-1].bias.data.fill_(0)
        self.trans_head[-1].bias.data[2] = 0.5

    def forward(self, x):
        """Forward pass: RGB image -> (rotation, translation)."""
        features = self.backbone(x).view(x.size(0), -1)
        
        rotation = self.rot_head(features)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        translation = self.trans_head(features)
        
        return rotation, translation


if __name__ == "__main__":
    model = PoseNetRGB()
    x = torch.randn(2, 3, 224, 224)
    rot, trans = model(x)
    print(f"Rotation: {rot.shape}, Translation: {trans.shape}")
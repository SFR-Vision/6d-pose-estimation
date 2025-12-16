import torch
import torch.nn as nn
import torchvision.models as models

class PoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PoseNet, self).__init__()
        
        # 1. Backbone: ResNet50 (Per PDF Instructions)
        print("🔧 Initializing ResNet50 Backbone...")
        # Load weights
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove the last fully connected layer (fc) and pooling layer
        # We want the spatial features or the flattened vector before classification
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Heads (Deeper & Wider for Better Precision)
        # ResNet50 output is 2048 dim (ResNet34 was 512)
        input_dim = 2048 
        
        # Rotation Head (Output: 4 values for Quaternion) - EXPANDED
        self.rot_head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # Quaternion (w, x, y, z)
        )
        
        # Translation Head (Output: 3 values for x, y, z) - EXPANDED
        self.trans_head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # Translation (x, y, z)
        )

        # 3. Smart Initialization
        # Initialize translation to start around z=0.5 meters (common depth)
        # This prevents the loss from starting massive.
        self.trans_head[-1].bias.data.fill_(0)
        self.trans_head[-1].bias.data[2] = 0.5

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        
        # 1. Extract Features
        features = self.backbone(x) # Output: [Batch, 2048, 1, 1]
        features = features.view(features.size(0), -1) # Flatten -> [Batch, 2048]
        
        # 2. Predict Heads
        rotation = self.rot_head(features)
        translation = self.trans_head(features)
        
        # Normalize quaternion to valid unit vector
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        return rotation, translation

if __name__ == "__main__":
    # Test the model shape
    net = PoseNet()
    dummy_input = torch.randn(2, 3, 224, 224)
    rot, trans = net(dummy_input)
    print(f"\n✅ Output Shapes Check (ResNet50):")
    print(f"   Rotation: {rot.shape} (Should be [2, 4])")
    print(f"   Translation: {trans.shape} (Should be [2, 3])")
"""RGB-D pose estimation network with cross-modal attention fusion."""

import torch
import torch.nn as nn
import torchvision.models as models


class CrossModalAttention(nn.Module):
    """Cross-modal attention for RGB-Depth feature fusion."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, rgb_feat, depth_feat):
        B = rgb_feat.size(0)
        
        q = self.q_proj(rgb_feat).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(depth_feat).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(depth_feat).view(B, self.num_heads, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).view(B, -1)
        return self.out_proj(out)


class PoseNetRGBD(nn.Module):
    """
    RGB-D pose estimation with cross-modal attention fusion.
    Uses dual ResNet50 backbones for RGB and depth, with attention-based fusion.
    """
    
    def __init__(self, pretrained=True):
        super(PoseNetRGBD, self).__init__()
        
        # RGB backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet_rgb = models.resnet50(weights=weights)
        self.rgb_backbone = nn.Sequential(*list(resnet_rgb.children())[:-1])
        
        # Depth backbone (adapted from RGB pretrained weights)
        resnet_depth = models.resnet50(weights=weights)
        original_conv1 = resnet_depth.conv1
        resnet_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            if pretrained:
                resnet_depth.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
        
        self.depth_backbone = nn.Sequential(*list(resnet_depth.children())[:-1])
        
        # Feature dimensions
        feat_dim = 2048
        fused_dim = 1024
        
        # Cross-modal attention
        self.rgb_norm = nn.LayerNorm(feat_dim)
        self.depth_norm = nn.LayerNorm(feat_dim)
        self.cross_attention = CrossModalAttention(feat_dim, num_heads=8, dropout=0.1)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )
        
        # Rotation head
        self.rot_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 4)
        )
        
        # Translation head
        self.trans_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 3)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in [self.fusion, self.rot_head, self.trans_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        self.trans_head[-1].bias.data[2] = 0.5

    def forward(self, rgb, depth, depth_raw=None, bbox_center=None, camera_matrix=None):
        """Forward pass: RGB + Depth -> (rotation, translation)."""
        batch_size = rgb.size(0)
        
        # Extract backbone features
        rgb_feat = self.rgb_backbone(rgb).view(batch_size, -1)
        depth_feat = self.depth_backbone(depth).view(batch_size, -1)
        
        # Normalize features
        rgb_feat = self.rgb_norm(rgb_feat)
        depth_feat = self.depth_norm(depth_feat)
        
        # Cross-modal attention
        rgb_enhanced = rgb_feat + self.cross_attention(rgb_feat, depth_feat)
        
        # Fuse and predict
        combined = torch.cat([rgb_enhanced, depth_feat], dim=1)
        fused = self.fusion(combined)
        
        rotation = self.rot_head(fused)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        
        translation = self.trans_head(fused)
        
        return rotation, translation
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PoseNetRGBD()
    print(f"Total Parameters: {model.count_parameters() / 1e6:.2f}M")
    
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)
    rot, trans = model(rgb, depth)
    print(f"Rotation: {rot.shape}, Translation: {trans.shape}")

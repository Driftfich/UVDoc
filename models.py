import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


def coord_grid(b: int, h: int, w: int, device) -> torch.Tensor:
    """(B, 2, H, W) CoordConv channels (x, y) in [-1, 1]."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device), indexing='ij')
    return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure minimum reduction to avoid 0 channels
        reduced_channels = max(channels // reduction, 1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel attention part
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention part
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_in))
        
        return x * spatial_att

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Apply CBAM attention to the skip connection features
        self.attention = CBAM(skip_c)
        # Depthwise Separable Convolution
        self.conv = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_c + skip_c, in_c + skip_c, 3, padding=1, groups=in_c + skip_c, bias=False),
            nn.BatchNorm2d(in_c + skip_c),
            nn.ReLU(inplace=True),
            # Pointwise conv
            nn.Conv2d(in_c + skip_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            
            # Depthwise conv
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            # Pointwise conv
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        
        # Apply attention to skip connection
        skip = self.attention(skip)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ContextBlock(nn.Module):
    """Simplified ASPP (Atrous Spatial Pyramid Pooling) for Mobile"""
    def __init__(self, in_c, out_c):
        super().__init__()
        # 1. Global Average Pooling (The "God View")
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        # 2. 1x1 Conv (Local detail)
        self.conv1x1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        
        # 3. Dilated Conv (Wide view) - Dilation=2
        # Sees a larger area without more parameters
        self.dilated = nn.Conv2d(in_c, out_c, 3, padding=2, dilation=2, bias=False)
        
        self.final = nn.Sequential(
            nn.Conv2d(out_c * 3, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        
        # Branch 1: Global Context
        gap = self.gap(x)
        gap = F.interpolate(gap, size=size, mode='bilinear', align_corners=True)
        
        # Branch 2 & 3
        c1 = self.conv1x1(x)
        dil = self.dilated(x)
        
        # Combine
        out = torch.cat([gap, c1, dil], dim=1)
        return self.final(out)

class SobelLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Horizontal edge filter
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        
        # Vertical edge filter
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y.weight.data = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
        
        # Freeze params
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
    def forward(self, x):
        # x: (B, 3, H, W) -> Convert to grayscale for edges
        gray = torch.mean(x, dim=1, keepdim=True)
        
        gx = self.sobel_x(gray)
        gy = self.sobel_y(gray)
        
        # Magnitude
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        # Normalize roughly to 0-1
        mag = torch.tanh(mag)
        return mag

class MobileNetV3UNet(nn.Module):
    """Legacy forward-UV model (U, V, Mask logits at input resolution)."""
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.sobel = SobelLayer()
        
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # Modify first layer for 6 channels (RGB + CoordConv + Sobel)
        original_first_layer = backbone.features[0][0]
        new_first_layer = nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Initialize
        with torch.no_grad():
            new_first_layer.weight[:, :3, :, :] = original_first_layer.weight
            nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            
        backbone.features[0][0] = new_first_layer
        
        return_nodes = {
            'features.3': 'layer1',   # 1/4
            'features.6': 'layer2',   # 1/8
            'features.12': 'layer3',  # 1/16
            'features.16': 'layer4',  # 1/32
        }
        self.encoder = create_feature_extractor(backbone, return_nodes)
        
        self.context_bot = ContextBlock(960, 64)
        
        self.up1 = UpBlock(64, 112, 112)
        self.up2 = UpBlock(112, 40, 40)
        self.up3 = UpBlock(40, 24, 24)
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(24, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(16 + 2 + 1, num_classes, 1) # +2 for Coord, +1 for Sobel

    def forward(self, x):
        edges = self.sobel(x[:, :3, :, :])
        b, _, h, w = x.shape
        grid = coord_grid(b, h, w, x.device)
        x_encoder = torch.cat([x, edges], dim=1) if x.shape[1] == 5 else torch.cat([x, grid, edges], dim=1)
            
        features = self.encoder(x_encoder)
        e1 = features['layer1']
        e2 = features['layer2']
        e3 = features['layer3']
        e4 = features['layer4']
        
        center = self.context_bot(e4)
        
        d1 = self.up1(center, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3)
        d5 = self.up5(d4)
        
        d5 = torch.cat([d5, grid, edges], dim=1)
        return self.final(d5)

class FastViTUNet(nn.Module):
    """
    Input: (B, 3, H, W) RGB in [0, 1], or (B, 5, H, W) RGB + CoordConv channels.

    backward_map=True:  returns the backward map (B, 2, H/8, W/8), see backward_map.py.
                        Predicted as identity grid + learned offset, so values may leave [-1, 1]
                        when the page is cut off by the image border.
    backward_map=False: legacy forward-UV model, returns (B, 3, H, W) logits (U, V, Mask).
    """
    def __init__(self, backward_map: bool = True, pretrained: bool = True):
        super().__init__()
        self.backward_map = backward_map
        
        self.sobel = SobelLayer()
        self.encoder = timm.create_model('fastvit_t8', pretrained=pretrained, in_chans=6, features_only=True)
        self.context_bot = ContextBlock(384, 128)
        
        self.up1 = UpBlock(in_c=128, skip_c=192, out_c=256) # 8x8 -> 16x16
        self.up2 = UpBlock(in_c=256, skip_c=96, out_c=128)  # 16x16 -> 32x32
        
        # Refines the 1/8 resolution features (fine e1 features are skipped on purpose against wobbly lines)
        self.refine1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_sparse = nn.Conv2d(32, 2 if backward_map else 3, 1)
        if backward_map:
            # Start at the identity map (page fills the image)
            nn.init.zeros_(self.final_sparse.weight)
            nn.init.zeros_(self.final_sparse.bias)

    def forward(self, x):
        edges = self.sobel(x[:, :3, :, :])
        b, _, h, w = x.shape
        x_encoder = torch.cat([x, edges], dim=1) if x.shape[1] == 5 else torch.cat([x, coord_grid(b, h, w, x.device), edges], dim=1)
            
        features = self.encoder(x_encoder)
        e2, e3, e4 = features[1], features[2], features[3]
        
        center = self.context_bot(e4)
        d1 = self.up1(center, e3)
        d2 = self.up2(d1, e2)
        r3 = self.refine3(self.refine2(self.refine1(d2)))
        out_sparse = self.final_sparse(r3)

        if self.backward_map:
            gh, gw = out_sparse.shape[-2:]
            return out_sparse + coord_grid(b, gh, gw, x.device)

        return F.interpolate(out_sparse, size=(h, w), mode='bilinear', align_corners=True)


def load_fastvit_unet(path, backward_map: bool = True, device='cpu') -> FastViTUNet:
    """Loads a checkpoint for inference (no pretrained download)."""
    model = FastViTUNet(backward_map=backward_map, pretrained=False)
    state = torch.load(path, map_location=device)
    # Older checkpoints contain an unused refine4 block
    state = {k: v for k, v in state.items() if not k.startswith('refine4.')}
    model.load_state_dict(state)
    return model.to(device).eval()

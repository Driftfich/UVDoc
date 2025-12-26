import argparse
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm

# define weights
w_mask = 1.0 # Document Mask Loss
w_uv = 10.0 # UV + Canny Edge Loss => Dewarping and Edge Alignment Loss
w_grad = 4.0 # Gradient Loss => Forces similar gradient in the GT and Predicted UV Map => Smoothness Loss
w_shading = 8.0 # Shading Loss
w_structure = 8.0 # Structural Loss => Edge Reprojection Alignment
w_stiffness = 2.0 # Stiffness Loss => Paper Curvature Assumption
w_tv_shading = 0 # TV Shading Loss => Shading Smoothness Loss
w_rigidity = 2.0 # Jacobian Rigidity Loss => Local Uniformity in the Predicted UV Map => Wobbly Lines Fix

weights = {
    "Mask": w_mask,
    "UV": w_uv,
    "Grad": w_grad,
    "Shading": w_shading,
    "Structure": w_structure,
    "Stiffness": w_stiffness,
    "TV": w_tv_shading,
    "Rigid": w_rigidity
}

# --- 1. Dataset with CoordConv ---
class UVDocUNetDataset(Dataset):
    def __init__(self, root_dir, split='train', size=(256, 256), transforms=None):
        self.root = Path(root_dir) / split
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.input_images_dir = self.root / "input_images"
        self.ids = [f.stem for f in self.images_dir.glob("*.jpg")]
        self.size = size
        self.transforms = transforms
        
        # Load Pre-computed Texture Edges and Sobel Targets
        self.flat_edge_map = Path("downloads/UVDoc_final/flat_edges")
        self.warped_edge_map = Path("downloads/UVDoc_final/warped_edges")
        
        # Precompute Coordinate Grids
        h, w = size
        xx_ones = np.ones((h, w), dtype=np.int32)
        xx_range = np.arange(w, dtype=np.int32)
        xx = xx_ones * xx_range[None, :] # (H, W) values 0..W-1
        
        yy_ones = np.ones((h, w), dtype=np.int32)
        yy_range = np.arange(h, dtype=np.int32)
        yy = yy_range[:, None] * yy_ones # (H, W) values 0..H-1
        
        # Normalize to [-1, 1] for better NN learning
        xx = xx / (w - 1) * 2 - 1
        yy = yy / (h - 1) * 2 - 1
        
        self.grid = np.stack([xx, yy], axis=0).astype(np.float32) # (2, H, W)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Load Image
        img_path = self.images_dir / f"{img_id}.jpg"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load Mask (UV Map)
        mask_path = self.masks_dir / f"{img_id}.npy"
        label_data = np.load(str(mask_path)) # (H, W, 5) or (H, W, 2)
        
        if label_data.shape[-1] >= 5:
            uv_map_gt = label_data[..., :2]
            shading_rgb = label_data[..., 2:5] # BGR
            shading_rgb = shading_rgb[..., ::-1].copy() # BGR -> RGB
            # Convert to 1-channel luminance
            # Note: shading_rgb is already float32 [0,1] from the .npy files
            shading_gt = cv2.cvtColor(shading_rgb, cv2.COLOR_RGB2GRAY)[..., None]
        else:
            uv_map_gt = label_data
            shading_gt = np.zeros((*uv_map_gt.shape[:2], 1), dtype=np.float32)
        
        # Target prep
        valid_mask = (uv_map_gt[..., 0] != -1).astype(np.float32)
        
        uv_clean = uv_map_gt.copy()
        uv_clean[uv_clean == -1] = 0
        
        # Ensure shading is clean (masked)
        shading_gt = shading_gt * valid_mask[..., None]

        # Load Flat Texture Canny Map
        flat_canny_path = self.flat_edge_map / f"{img_id}.png"
        if flat_canny_path.exists():
            flat_canny = cv2.imread(str(flat_canny_path), cv2.IMREAD_GRAYSCALE)
            flat_canny = flat_canny.astype(np.float32) / 255.0
        else:
            flat_canny = np.zeros(self.size, dtype=np.float32)

        # Load Warped Texture Canny Map (Target)
        warped_canny_path = self.warped_edge_map / f"{img_id}.png"
        if warped_canny_path.exists():
            warped_canny = cv2.imread(str(warped_canny_path), cv2.IMREAD_GRAYSCALE)
            warped_canny = warped_canny.astype(np.float32) / 255.0
        else:
            warped_canny = np.zeros(self.size, dtype=np.float32)
        
        # Resize if needed (safety check)
        if flat_canny.shape != self.size:
            flat_canny = cv2.resize(flat_canny, (self.size[1], self.size[0]))
        if warped_canny.shape != self.size:
            warped_canny = cv2.resize(warped_canny, (self.size[1], self.size[0]))
        
        # Channels: 0,1=UV, 2=Mask, 3=Shading(Luma), 4=FlatCanny, 5=WarpedCanny
        target = np.concatenate([
            uv_clean, 
            valid_mask[..., None], 
            shading_gt,
            flat_canny[..., None],
            warped_canny[..., None]
        ], axis=-1)
        
        # Apply Augmentations
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=target)
            img = augmented['image']
            target = augmented['mask']
        
        # Preprocessing
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # (3, H, W)
        
        # Add CoordConv channels
        img = np.concatenate([img, self.grid], axis=0) # (5, H, W)
        
        target = np.transpose(target, (2, 0, 1))
        
        return torch.from_numpy(img), torch.from_numpy(target)

# --- 2. Model: MobileNetV3-UNet (5-Channel Input) ---
class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
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
    def __init__(self, num_classes=4): # U, V, Mask, Shading (Luma)
        super().__init__()
        
        self.sobel = SobelLayer()
        
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # Modify first layer for 6 channels (RGB + CoordConv + Sobel)
        # Original: Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_first_layer = backbone.features[0][0]
        new_first_layer = nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Initialize
        with torch.no_grad():
            new_first_layer.weight[:, :3, :, :] = original_first_layer.weight
            # Init coord + sobel channels with kaiming
            nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            
        backbone.features[0][0] = new_first_layer
        
        return_nodes = {
            'features.3': 'layer1',   # 1/4
            'features.6': 'layer2',   # 1/8
            'features.12': 'layer3',  # 1/16
            'features.16': 'layer4',  # 1/32
        }
        self.encoder = create_feature_extractor(backbone, return_nodes)
        
        # # Context Block at Bottleneck
        self.context_bot = ContextBlock(960, 64)
        
        # Up1 input reduced from 960 to 64
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 0. Precompute Edges (Feature Engineering) from RGB
        edges = self.sobel(x[:, :3, :, :]) # (B, 1, H, W)
        
        # 1. Generate coordinate grid for the input resolution
        b, _, h, w = x.shape
        xx_range = torch.linspace(-1, 1, w, device=x.device)
        yy_range = torch.linspace(-1, 1, h, device=x.device)
        yy, xx = torch.meshgrid(yy_range, xx_range, indexing='ij')
        grid = torch.stack([xx, yy], dim=0) # (2, H, W)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        
        # 2. Construct 6-channel input for Encoder: (RGB, CoordX, CoordY, Edges)
        if x.shape[1] == 5:
            # x already has RGB + CoordConv from Dataset
            x_encoder = torch.cat([x, edges], dim=1)
        else:
            # Inference case or raw RGB
            x_encoder = torch.cat([x, grid, edges], dim=1)
            
        features = self.encoder(x_encoder)
        e1 = features['layer1']
        e2 = features['layer2']
        e3 = features['layer3']
        e4 = features['layer4']
        
        # # Apply Context Block
        center = self.context_bot(e4)
        
        d1 = self.up1(center, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3)
        d5 = self.up5(d4)
        
        # Concatenate grid AND edges before final layer
        d5 = torch.cat([d5, grid, edges], dim=1)
        
        out = self.final(d5)
        return self.sigmoid(out)

# --- 3. Loss with Gradients ---
class GeometricLoss(nn.Module):
    def __init__(self, scales=None):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.bce = nn.BCELoss()
        self.sobel = SobelLayer() # Use the differentiable layer we built
        
        # Default scales to 1.0 if not provided
        self.scales = scales if scales is not None else {
            "Mask": 1.0, "UV": 1.0, "Grad": 1.0, "Shading": 1.0, 
            "Structure": 1.0, "Stiffness": 1.0, "TV": 1.0, "Rigid": 1.0
        }
        
    def gradient_loss(self, pred_uv, target_uv, mask):
        # Dy: (B, 2, H-1, W)
        dy_pred = torch.abs(pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :])
        dy_gt = torch.abs(target_uv[:, :, 1:, :] - target_uv[:, :, :-1, :])
        
        # Dx: (B, 2, H, W-1)
        dx_pred = torch.abs(pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1])
        dx_gt = torch.abs(target_uv[:, :, :, 1:] - target_uv[:, :, :, :-1])
        
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        loss_dy = torch.abs(dy_pred - dy_gt) * mask_y
        loss_dx = torch.abs(dx_pred - dx_gt) * mask_x
        
        return (loss_dy.sum() + loss_dx.sum()) / (mask_y.sum() + mask_x.sum() + 1e-6)

    def curvature_loss(self, pred_uv, mask):
        # pred_uv: (B, 2, H, W)
        
        # 2nd Derivative in X (u_xx): (u_{i+1} - 2u_i + u_{i-1})
        # This is effectively a [1, -2, 1] convolution
        # We want this to be 0 (straight line) or match the GT curvature.
        # Since paper is smooth, minimizing the absolute magnitude of the 2nd derivative 
        # acts as a strong regularizer against high-frequency wobbles.
        
        # d/dx
        dx = pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1]
        # d^2/dx^2 (Acceleration)
        dx2 = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        
        # d/dy
        dy = pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :]
        # d^2/dy^2
        dy2 = dy[:, :, 1:, :] - dy[:, :, :-1, :]
        
        # Masking: We need to shrink the mask to match the smaller output dimensions
        # dx2 is (H, W-2), dy2 is (H-2, W)
        m_x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
        m_y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
        
        # We want the curvature to be minimal (smooth paper assumption)
        loss_x = torch.mean(torch.abs(dx2) * m_x)
        loss_y = torch.mean(torch.abs(dy2) * m_y)
        
        return loss_x + loss_y

    def tv_shading_loss(self, shading, mask):
        # Penalize neighboring pixel differences in shading (Total Variation)
        # dh: (B, 1, H-1, W)
        dh = torch.abs(shading[:, :, 1:, :] - shading[:, :, :-1, :])
        # dw: (B, 1, H, W-1)
        dw = torch.abs(shading[:, :, :, 1:] - shading[:, :, :, :-1])
        
        mask_h = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_w = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        loss_h = (dh * mask_h).sum() / (mask_h.sum() + 1e-6)
        loss_w = (dw * mask_w).sum() / (mask_w.sum() + 1e-6)
        
        return loss_h + loss_w

    def jacobian_rigidity_loss(self, pred_uv, mask):
        # pred_uv: (B, 2, H, W)
        # Calculates the Jacobian matrix elements: du/dx, du/dy, dv/dx, dv/dy
        # We want the local transformation to be uniform (constant derivatives)
        
        du_dx = pred_uv[:, 0:1, :, 1:] - pred_uv[:, 0:1, :, :-1]
        du_dy = pred_uv[:, 0:1, 1:, :] - pred_uv[:, 0:1, :-1, :]
        dv_dx = pred_uv[:, 1:2, :, 1:] - pred_uv[:, 1:2, :, :-1]
        dv_dy = pred_uv[:, 1:2, 1:, :] - pred_uv[:, 1:2, :-1, :]
        
        # Penalize variance of these derivatives in local neighborhoods
        # A simple way is to take the 1st derivative of the 1st derivatives (which is 2nd derivative)
        # but focused on the orthogonality.
        # For simplicity and efficiency, we ensure du_dx == dv_dy and du_dy == -dv_dx (Cauchy-Riemann like)
        # OR we just penalize high-frequency changes in these derivatives.
        
        # We'll use the "Rigidity" formulation: (du/dx - 1)^2 + (dv/dy - 1)^2 + (du/dy)^2 + (dv/dx)^2
        # But since our UV is normalized 0-1, the "1" depends on resolution.
        # Instead, we just penalize the difference between neighboring Jacobian elements.
        
        diff_u_x = du_dx[:, :, :, 1:] - du_dx[:, :, :, :-1]
        diff_u_y = du_dy[:, :, 1:, :] - du_dy[:, :, :-1, :]
        diff_v_x = dv_dx[:, :, :, 1:] - dv_dx[:, :, :, :-1]
        diff_v_y = dv_dy[:, :, 1:, :] - dv_dy[:, :, :-1, :]
        
        m_x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
        m_y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
        
        loss = (diff_u_x.abs() * m_x).mean() + (diff_u_y.abs() * m_y).mean() + \
               (diff_v_x.abs() * m_x).mean() + (diff_v_y.abs() * m_y).mean()
        
        return loss

    def forward(self, pred, target, input_img):
        pred_uv = pred[:, :2, :, :]
        pred_mask = pred[:, 2, :, :]
        pred_shading = pred[:, 3:4, :, :]
        
        target_uv = target[:, :2, :, :]
        target_mask = target[:, 2, :, :]
        target_shading = target[:, 3:4, :, :]
        
        # These are in FLAT domain (0..1 UV space)
        # Channels: 0,1=UV, 2=Mask, 3=Shading(Luma), 4=FlatCanny, 5=WarpedCanny
        flat_canny_map = target[:, 4, :, :].unsqueeze(1) # (B, 1, H, W)
        warped_canny_map = target[:, 5, :, :].unsqueeze(1) # (B, 1, H, W)
        
        # 1. Mask Loss
        loss_mask = self.bce(pred_mask, target_mask)
        
        mask_expanded = target_mask.unsqueeze(1).repeat(1, 2, 1, 1)
        
        # --- PREPARE GRIDS FOR SAMPLING ---
        # Convert 0..1 UVs to -1..1 Grid coordinates for grid_sample
        # GT UVs (for warping weights)
        grid_gt = target_uv.permute(0, 2, 3, 1) * 2.0 - 1.0 # (B, H, W, 2)
        # Pred UVs (for structural loss)
        grid_pred = pred_uv.permute(0, 2, 3, 1) * 2.0 - 1.0 # (B, H, W, 2)
        
        
        # 2. Edge-Weighted UV L1 Loss
        # We use the precomputed synthetic warped edges as weights.
        # This aligns the weights with the actual pixels in the input image.
        
        # Weights: 1.0 (base) + 5.0 * WarpedEdges
        edge_weights = 1.0 + 5.0 * warped_canny_map
        edge_weights_uv = edge_weights.repeat(1, 2, 1, 1)
        
        loss_uv = self.l1(pred_uv, target_uv)
        loss_uv = (loss_uv * mask_expanded * edge_weights_uv).sum() / (mask_expanded.sum() + 1e-6)
        
        # 3. Multi-Scale Gradient Loss
        # Scale 1 (Original)
        loss_grad_1 = self.gradient_loss(pred_uv, target_uv, mask_expanded)
        
        # Scale 2 (Downsample 2x)
        pred_uv_2 = F.interpolate(pred_uv, scale_factor=0.5, mode='bilinear')
        target_uv_2 = F.interpolate(target_uv, scale_factor=0.5, mode='bilinear')
        mask_2 = F.interpolate(mask_expanded, scale_factor=0.5, mode='nearest')
        loss_grad_2 = self.gradient_loss(pred_uv_2, target_uv_2, mask_2)
        
        # Scale 4 (Downsample 4x)
        pred_uv_4 = F.interpolate(pred_uv, scale_factor=0.25, mode='bilinear')
        target_uv_4 = F.interpolate(target_uv, scale_factor=0.25, mode='bilinear')
        mask_4 = F.interpolate(mask_expanded, scale_factor=0.25, mode='nearest')
        loss_grad_4 = self.gradient_loss(pred_uv_4, target_uv_4, mask_4)
        
        loss_grad_total = loss_grad_1 + loss_grad_2 + loss_grad_4
        
        # 4. Shading Loss
        mask_expanded_shading = target_mask.unsqueeze(1)
        loss_shading = self.l1(pred_shading, target_shading)
        loss_shading = (loss_shading * mask_expanded_shading).sum() / (mask_expanded_shading.sum() + 1e-6)
        
        # 5. Structural Loss (Warped Canny Alignment)
        # Forces Predicted UVs to align the Flat Texture edges with the GT Warped Texture edges.
        # Since we precompute the synthetic target in generate_edges.py, 
        # this loss is now noise-free and can reach 0.
        
        pred_warped_edges = F.grid_sample(flat_canny_map, grid_pred, align_corners=True, mode='bilinear')
        
        loss_structure = self.l1(pred_warped_edges, warped_canny_map)
        loss_structure = (loss_structure * target_mask.unsqueeze(1)).sum() / (target_mask.sum() + 1e-6)
        
        # 6. Stiffness (Curvature) Loss
        loss_stiffness = self.curvature_loss(pred_uv, mask_expanded)

        # 7. TV Shading Loss
        loss_tv_shading = self.tv_shading_loss(pred_shading, target_mask.unsqueeze(1))
        
        # 8. Jacobian Rigidity Loss (Fixes wobbly lines)
        loss_rigidity = self.jacobian_rigidity_loss(pred_uv, target_mask.unsqueeze(1))

        # print loss components
        components = {
            "Mask": loss_mask.item(),
            "UV": loss_uv.item(),
            "Grad": loss_grad_total.item(),
            "Shading": loss_shading.item(),
            "Structure": loss_structure.item(),
            "Stiffness": loss_stiffness.item(),
            "TV": loss_tv_shading.item(),
            "Rigid": loss_rigidity.item()
        }

        total_weights = (w_mask + w_uv + w_grad + w_shading + 
                         w_structure + w_stiffness + w_tv_shading + w_rigidity)

        total_loss = (w_mask * self.scales["Mask"] * loss_mask + 
                w_uv * self.scales["UV"] * loss_uv + 
                w_grad * self.scales["Grad"] * loss_grad_total + 
                w_shading * self.scales["Shading"] * loss_shading + 
                w_structure * self.scales["Structure"] * loss_structure + 
                w_stiffness * self.scales["Stiffness"] * loss_stiffness + 
                w_tv_shading * self.scales["TV"] * loss_tv_shading +
                w_rigidity * self.scales["Rigid"] * loss_rigidity) / total_weights
        
        debug_info = {
            "pred_warped_edges": pred_warped_edges,
            "warped_canny_map": warped_canny_map
        }
        
        return total_loss, components, debug_info

def get_train_transforms(args):
    return A.Compose([
        A.ColorJitter(
            brightness=args.aug_color, 
            contrast=args.aug_color, 
            saturation=args.aug_color, 
            hue=args.aug_color/2, 
            p=args.aug_prob
        ),
        A.GaussNoise(std_range=(0.0, args.aug_noise_std), mean_range=(0.0, 0.0), p=args.aug_prob),
    ])

def calibrate_loss_scales(model, train_loader, device, num_steps=1000):
    """
    Runs a warmup phase to compute scaling factors for each loss component.
    """
    import copy
    print(f"Starting Calibration Phase ({num_steps} steps)...")
    
    # Use a copy of the model so calibration updates don't persist
    temp_model = copy.deepcopy(model).to(device)
    temp_model.train()
    
    optimizer = optim.AdamW(temp_model.parameters(), lr=1e-3)
    # Use unscaled criterion for calibration
    criterion = GeometricLoss(scales=None).to(device)
    
    sums = {}
    count = 0
    
    pbar = tqdm(total=num_steps, desc="Calibrating")
    
    step = 0
    while step < num_steps:
        for imgs, targets in train_loader:
            if step >= num_steps: break
            
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            preds = temp_model(imgs)
            loss, components, _ = criterion(preds, targets, imgs)
            
            loss.backward()
            optimizer.step()
            
            # Accumulate
            for k, v in components.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            
            step += 1
            pbar.update(1)
            
    pbar.close()
    
    # Compute scales: scale = 1.0 / mean_value
    # This makes every scaled component roughly magnitude 1.0
    means = {k: v / count for k, v in sums.items()}
    scales = {k: 1.0 / (v + 1e-8) for k, v in means.items()}
    
    print("Calibration complete.")
    # for k, v in means.items():
    #     print(f"  {k}: {v:.6f} -> Scale: {scales[k]:.6f}")
        
    return scales

def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_ds = UVDocUNetDataset(args.data_dir, 'train', transforms=get_train_transforms(args))
    val_ds = UVDocUNetDataset(args.data_dir, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    model = MobileNetV3UNet(num_classes=4).to(device)
    
    # --- 1. Calibration / Scaling ---
    import json
    scales_path = Path(args.scales_file)
    
    if args.calibrate or not scales_path.exists():
        scales = calibrate_loss_scales(model, train_loader, device, num_steps=1000)
        with open(scales_path, 'w') as f:
            json.dump(scales, f, indent=4)
        print(f"Saved scales to {scales_path}")
    else:
        with open(scales_path, 'r') as f:
            scales = json.load(f)
        print(f"Loaded scales from {scales_path}")

    for k, v in scales.items():
        print(f"  {k}: {v:.6f} -> Factor: {v:.6f}")
    for k, v in weights.items():
        print(f"  {k}: {v:.6f} -> Weight: {v:.6f}")
    for k, v in weights.items():
        print(f"  {k}: {v:.6f} -> Scale: {scales[k] * weights[k]:.6f}")
        

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = GeometricLoss(scales=scales).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    print("------ Dataset Info ------\n" \
        f"Train Size: {len(train_ds)}\n" \
        f"Val Size: {len(val_ds)}\n" \
        "------ Training Config ------\n" \
        f"Batch Size: {args.batch}\n" \
        f"Log Interval: {args.log_interval}\n" \
        f"Augment Prob: {args.aug_prob}\n" \
        f"Augment Color: {args.aug_color}\n" \
        f"Augment Noise Std: {args.aug_noise_std}\n" \
        )
    print(f"Training for {args.epochs} epochs with CoordConv + Gradient Loss...")
    best_val = float('inf')
    
    global_step = 0
    accumulated_components = {}

    if args.verbose:
        verbose_dir = Path(args.verbose_dir)
        verbose_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # 1. Main Forward Pass
            preds = model(imgs)
            loss, components, debug_info = criterion(preds, targets, imgs)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Accumulate components
            for k, v in components.items():
                accumulated_components[k] = accumulated_components.get(k, 0.0) + v
            
            global_step += 1
            if global_step % args.log_interval == 0:
                avg_comp = {k: v / args.log_interval for k, v in accumulated_components.items()}
                log_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_comp.items()])
                print(f"Step {global_step} | {log_str}")
                accumulated_components = {}
                
                # Verbose Visualization
                if args.verbose:
                    # Take first sample from batch
                    pred_edges = debug_info["pred_warped_edges"][0, 0].detach().cpu().numpy()
                    gt_edges = debug_info["warped_canny_map"][0, 0].detach().cpu().numpy()
                    mask = targets[0, 2].detach().cpu().numpy() # Document Mask
                    
                    # Create RGB image: Red = Prediction, Blue = Ground Truth
                    h, w = pred_edges.shape
                    vis = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Apply mask to edges so we only see the document area
                    pred_edges_masked = pred_edges * mask
                    gt_edges_masked = gt_edges * mask
                    
                    vis[..., 0] = (pred_edges_masked * 255).astype(np.uint8) # Red
                    vis[..., 2] = (gt_edges_masked * 255).astype(np.uint8)   # Blue
                    
                    # Overlap channel (Green) to show magenta where they align
                    # This makes alignment much more obvious
                    overlap = np.minimum(pred_edges_masked, gt_edges_masked)
                    vis[..., 1] = (overlap * 255).astype(np.uint8)
                    
                    save_path = verbose_dir / f"step_{global_step:06d}.png"
                    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss, _, _ = criterion(preds, targets, imgs)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "unet_best.pth")
            print("  -> Saved best model")
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_unet")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--aug_color", type=float, default=0.2, help="Strength of color jitter")
    parser.add_argument("--aug_noise_std", type=float, default=0.2, help="Max standard deviation for Gaussian noise (0-1 range)")
    parser.add_argument("--log_interval", type=int, default=1000, help="Interval for printing accumulated loss components")
    parser.add_argument("--calibrate", action='store_true', help="Force run calibration phase")
    parser.add_argument("--scales_file", default="loss_scales.json", help="Path to save/load loss scaling factors")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose visualization during training")
    parser.add_argument("--verbose_dir", default="train_vis_debug", help="Directory to save verbose visualization images")
    args = parser.parse_args()
    
    train_unet(args)

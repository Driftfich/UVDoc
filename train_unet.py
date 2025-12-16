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
        self.edge_map_dir = Path("downloads/UVDoc_final/texture_edges")
        self.sobel_target_dir = Path("downloads/UVDoc_final/texture_sobel")
        
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
        
        if label_data.shape[-1] == 5:
            uv_map_gt = label_data[..., :2]
            shading_gt = label_data[..., 2:] # BGR
            shading_gt = shading_gt[..., ::-1].copy() # BGR -> RGB
        else:
            uv_map_gt = label_data
            shading_gt = np.zeros((*uv_map_gt.shape[:2], 3), dtype=np.float32)
        
        # Target prep
        valid_mask = (uv_map_gt[..., 0] != -1).astype(np.float32)
        
        # Erode mask to remove border artifacts (e.g. green background from geometry images)
        # 5x5 kernel removes ~2 pixels of border, which clears interpolation artifacts
        # kernel = np.ones((5, 5), np.uint8)
        # valid_mask = cv2.erode(valid_mask, kernel, iterations=1)
        
        uv_clean = uv_map_gt.copy()
        uv_clean[uv_clean == -1] = 0
        
        # Ensure shading is clean (masked)
        shading_gt = shading_gt * valid_mask[..., None]

        # Load Sobel Target
        sobel_target_path = self.sobel_target_dir / f"{img_id}.png"
        if sobel_target_path.exists():
            sobel_target = cv2.imread(str(sobel_target_path), cv2.IMREAD_GRAYSCALE)
            sobel_target = sobel_target.astype(np.float32) / 255.0
        else:
            sobel_target = np.zeros(self.size, dtype=np.float32)

        # Load Texture Edge Map
        tex_edge_map_path = self.edge_map_dir / f"{img_id}.png"
        if tex_edge_map_path.exists():
            tex_edge_map = cv2.imread(str(tex_edge_map_path), cv2.IMREAD_GRAYSCALE)
            tex_edge_map = tex_edge_map.astype(np.float32) / 255.0
        else:
            tex_edge_map = np.zeros(self.size, dtype=np.float32)
        
        # Resize if needed (safety check)
        if sobel_target.shape != self.size:
            sobel_target = cv2.resize(sobel_target, (self.size[1], self.size[0]))
        if tex_edge_map.shape != self.size:
            tex_edge_map = cv2.resize(tex_edge_map, (self.size[1], self.size[0]))
        
        # Channels: 0,1=UV, 2=Mask, 3,4,5=Shading(RGB), 6=TextureEdge, 7=SobelTarget
        target = np.concatenate([
            uv_clean, 
            valid_mask[..., None], 
            shading_gt,
            tex_edge_map[..., None],
            sobel_target[..., None]
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
        if x.shape[2:] != skip.shape[2:]:
            x = transforms.functional.resize(x, size=skip.shape[2:])
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
    def __init__(self, num_classes=6): # U, V, Mask, R, G, B
        super().__init__()
        
        self.sobel = SobelLayer()
        
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        # Modify first layer for 5 channels (CoordConv)
        # Original: Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_first_layer = backbone.features[0][0]
        new_first_layer = nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Initialize
        with torch.no_grad():
            new_first_layer.weight[:, :3, :, :] = original_first_layer.weight
            # Init coord channels with small noise
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
        # 0. Precompute Edges (Feature Engineering)
        edges = self.sobel(x) # (B, 1, H, W)
        
        # Generate coordinate grid for the input resolution
        b, _, h, w = x.shape
        xx_ones = torch.ones(h, w, dtype=torch.int32, device=x.device)
        xx_range = torch.arange(w, dtype=torch.int32, device=x.device)
        xx = xx_ones * xx_range[None, :]
        yy_ones = torch.ones(h, w, dtype=torch.int32, device=x.device)
        yy_range = torch.arange(h, dtype=torch.int32, device=x.device)
        yy = yy_range[:, None] * yy_ones
        
        xx = xx / (w - 1) * 2 - 1
        yy = yy / (h - 1) * 2 - 1
        grid = torch.stack([xx, yy], dim=0).float() # (2, H, W)
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
        
        features = self.encoder(x)
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
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.bce = nn.BCELoss()
        self.sobel = SobelLayer() # Use the differentiable layer we built
        
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

    def forward(self, pred, target, input_img):
        pred_uv = pred[:, :2, :, :]
        pred_mask = pred[:, 2, :, :]
        pred_shading = pred[:, 3:, :, :]
        
        target_uv = target[:, :2, :, :]
        target_mask = target[:, 2, :, :]
        target_shading = target[:, 3:6, :, :]
        target_weights = target[:, 6, :, :].unsqueeze(1) # (B, 1, H, W)
        target_sobel = target[:, 7, :, :].unsqueeze(1)   # (B, 1, H, W)
        
        # 1. Mask Loss
        loss_mask = self.bce(pred_mask, target_mask)
        
        mask_expanded = target_mask.unsqueeze(1).repeat(1, 2, 1, 1)
        
        # 2. Edge-Weighted UV L1 Loss
        # Weights: 1.0 (base) + 5.0 * EdgeMap
        edge_weights = 1.0 + 5.0 * target_weights
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
        mask_expanded_shading = target_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        loss_shading = self.l1(pred_shading, target_shading)
        loss_shading = (loss_shading * mask_expanded_shading).sum() / (mask_expanded_shading.sum() + 1e-6)
        
        # 5. Structural Loss (Strategy B) - Updated for Albedo
        # Calculate Albedo to remove lighting/shadows before Sobel check
        
        # Clamp shading to avoid div/0
        shading_clamped = torch.clamp(pred_shading, 0.1, 1.0)
        
        # Input is normalized 0-1. Albedo = Input / Shading
        # input_img has 5 channels (RGB + XY), we need RGB
        pred_albedo = input_img[:, :3, :, :] / shading_clamped
        pred_albedo = torch.clamp(pred_albedo, 0, 1)
        
        # Prepare Grid (Sigmoid 0-1 -> Grid -1 to 1)
        grid = pred_uv.permute(0, 2, 3, 1) # (B, H, W, 2)
        grid = grid * 2.0 - 1.0
        
        # Differentiable Unwarp on ALBEDO
        unwarped_albedo = F.grid_sample(pred_albedo, grid, align_corners=True)
        
        # Compute Sobel on Unwarped Albedo
        sobel_pred = self.sobel(unwarped_albedo)
        
        # Loss: L1 distance between Predicted Albedo Structure and Target Structure
        loss_structure = self.l1(sobel_pred, target_sobel)
        # Use simple mask (1 channel) for structure loss
        loss_structure = (loss_structure * target_mask.unsqueeze(1)).sum() / (target_mask.sum() + 1e-6)
        
        # 6. Stiffness (Curvature) Loss
        loss_stiffness = self.curvature_loss(pred_uv, mask_expanded)
        
        # Weights: Mask=1.0, UV=10.0, Grad=5.0, Shading=5.0, Structure=5.0, Stiffness=10.0
        return loss_mask + 10.0 * loss_uv + 5.0 * loss_grad_total + 5.0 * loss_shading + 5.0 * loss_structure + 10.0 * loss_stiffness

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

def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_ds = UVDocUNetDataset(args.data_dir, 'train', transforms=get_train_transforms(args))
    val_ds = UVDocUNetDataset(args.data_dir, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    model = MobileNetV3UNet(num_classes=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = GeometricLoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    print(f"Training for {args.epochs} epochs with CoordConv + Gradient Loss...")
    best_val = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # 1. Main Forward Pass
            preds = model(imgs)
            loss = criterion(preds, targets, imgs)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets, imgs)
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--aug_color", type=float, default=0.2, help="Strength of color jitter")
    parser.add_argument("--aug_noise_std", type=float, default=0.2, help="Max standard deviation for Gaussian noise (0-1 range)")
    args = parser.parse_args()
    
    train_unet(args)

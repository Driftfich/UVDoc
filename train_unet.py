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

# --- 1. Dataset with CoordConv ---
class UVDocUNetDataset(Dataset):
    def __init__(self, root_dir, split='train', size=(256, 256)):
        self.root = Path(root_dir) / split
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.ids = [f.stem for f in self.images_dir.glob("*.jpg")]
        self.size = size
        
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
        uv_map_gt = np.load(str(mask_path)) # (H, W, 2)
        
        # Target prep
        valid_mask = (uv_map_gt[..., 0] != -1).astype(np.float32)
        uv_clean = uv_map_gt.copy()
        uv_clean[uv_clean == -1] = 0
        target = np.concatenate([uv_clean, valid_mask[..., None]], axis=-1)
        
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

class MobileNetV3UNet(nn.Module):
    def __init__(self, num_classes=3): # U, V, Mask
        super().__init__()
        
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
        
        self.up1 = UpBlock(960, 112, 112)
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
        
        self.final = nn.Conv2d(16, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        e1 = features['layer1']
        e2 = features['layer2']
        e3 = features['layer3']
        e4 = features['layer4']
        
        d1 = self.up1(e4, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3)
        d5 = self.up5(d4)
        
        out = self.final(d5)
        return self.sigmoid(out)

# --- 3. Loss with Gradients ---
class GeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.bce = nn.BCELoss()
        
    def gradient_loss(self, pred_uv, target_uv, mask):
        # pred_uv: (B, 2, H, W)
        
        # Dy: (B, 2, H-1, W)
        dy_pred = torch.abs(pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :])
        dy_gt = torch.abs(target_uv[:, :, 1:, :] - target_uv[:, :, :-1, :])
        
        # Dx: (B, 2, H, W-1)
        dx_pred = torch.abs(pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1])
        dx_gt = torch.abs(target_uv[:, :, :, 1:] - target_uv[:, :, :, :-1])
        
        # Mask needs slicing to match dims
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        # L1 on gradients
        loss_dy = torch.abs(dy_pred - dy_gt) * mask_y
        loss_dx = torch.abs(dx_pred - dx_gt) * mask_x
        
        return (loss_dy.sum() + loss_dx.sum()) / (mask_y.sum() + mask_x.sum() + 1e-6)

    def forward(self, pred, target):
        pred_uv = pred[:, :2, :, :]
        pred_mask = pred[:, 2, :, :]
        
        target_uv = target[:, :2, :, :]
        target_mask = target[:, 2, :, :]
        
        # 1. Mask Loss
        loss_mask = self.bce(pred_mask, target_mask)
        
        # 2. UV L1 Loss
        mask_expanded = target_mask.unsqueeze(1).repeat(1, 2, 1, 1)
        loss_uv = self.l1(pred_uv, target_uv)
        loss_uv = (loss_uv * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
        
        # 3. Gradient Loss (Stiffness)
        loss_grad = self.gradient_loss(pred_uv, target_uv, mask_expanded)
        
        # Weights: Mask=1.0, UV=10.0 (geometry is subtle), Grad=1.0
        return loss_mask + 10.0 * loss_uv + 1.0 * loss_grad

def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_ds = UVDocUNetDataset(args.data_dir, 'train')
    val_ds = UVDocUNetDataset(args.data_dir, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    model = MobileNetV3UNet(num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = GeometricLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    print(f"Training for {args.epochs} epochs with CoordConv + Gradient Loss...")
    best_val = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
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
                loss = criterion(preds, targets)
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
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_unet(args)

import argparse
import os
import time
import json
import orjson
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm

# Enable EXR support for Doc3D
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# define weights
w_mask = 1.0 # Document Mask Loss
w_uv = 10.0 # UV + Canny Edge Loss => Dewarping and Edge Alignment Loss
w_grad = 4.0 # Gradient Loss => Forces similar gradient in the GT and Predicted UV Map => Smoothness Loss
w_shading = 8.0 # Shading Loss
w_stiffness = 2.0 # Stiffness Loss => Paper Curvature Assumption
w_tv_shading = 0.1 # TV Shading Loss => Shading Smoothness Loss
w_rigidity = 2.0 # Jacobian Rigidity Loss => Local Uniformity in the Predicted UV Map => Wobbly Lines Fix
w_structure = 8.0 # Structure Loss => Edge Alignment Loss

# NEW: Content / Perceptual Losses (Replaces old 'Structure' loss)
w_reproj = 10.0    # Pixel L1 diff between Unwarped Prediction and Flat GT
w_vgg = 5.0        # VGG Perceptual loss (Feature matching)

weights = {
    "Mask": w_mask,
    "UV": w_uv,
    "Grad": w_grad,
    "Shading": w_shading,
    "Stiffness": w_stiffness,
    "TV": w_tv_shading,
    "Rigid": w_rigidity,
    "Reproj": w_reproj,
    "VGG": w_vgg,
    "Structure": w_structure
}

# --- 1. UVDoc Dataset ---
class UVDocUNetDataset(Dataset):
    def __init__(self, root_dir, split='train', size=(256, 256), transforms=None):
        self.root = Path(root_dir) / split
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.ids = [f.stem for f in self.images_dir.glob("*.jpg")]
        self.size = size
        self.transforms = transforms
        
        # We need the original textures to calculate VGG/Reprojection loss
        self.tex_dir = Path("downloads/UVDoc_final/textures")
        self.meta_dir = Path("downloads/UVDoc_final/metadata_sample")

        # Load Pre-computed Texture Edges and Sobel Targets
        self.flat_edge_map = Path("downloads/UVDoc_final/flat_edges")
        self.warped_edge_map = Path("downloads/UVDoc_final/warped_edges")

        # Load Shadow Clean Warped Image with black background
        self.shadow_clean_warped_dir = Path("downloads/UVDoc_final/warped_textures")
        
        # Precompute Coordinate Grids
        h, w = size
        xx_range = np.arange(w, dtype=np.int32)
        yy_range = np.arange(h, dtype=np.int32)
        xx, yy = np.meshgrid(xx_range, yy_range)
        
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
        
        # Resize image to match model input size if necessary
        if img.shape[:2] != self.size:
             img = cv2.resize(img, (self.size[1], self.size[0]))
        
        # Load Mask (UV Map)
        mask_path = self.masks_dir / f"{img_id}.npy"
        label_data = np.load(str(mask_path)) # (H, W, 5) or (H, W, 2)
        
        if label_data.shape[-1] >= 5:
            uv_map_gt = label_data[..., :2]
            shading_rgb = label_data[..., 2:5] # BGR
            shading_rgb = shading_rgb[..., ::-1].copy() # BGR -> RGB
            # Convert to 1-channel luminance
            shading_gt = cv2.cvtColor(shading_rgb, cv2.COLOR_RGB2GRAY)[..., None]
        else:
            # print(f"Warning: Shading not found for {img_id}")
            uv_map_gt = label_data
            shading_gt = np.zeros((*uv_map_gt.shape[:2], 1), dtype=np.float32)
        
        # Target prep
        valid_mask = (uv_map_gt[..., 0] != -1).astype(np.float32)
        
        # Resize masks if necessary (nearest neighbor for masks/UVs to preserve values)
        if uv_map_gt.shape[:2] != self.size:
             uv_map_gt = cv2.resize(uv_map_gt, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
             valid_mask = cv2.resize(valid_mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
             shading_gt = cv2.resize(shading_gt, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
             if len(shading_gt.shape) == 2: shading_gt = shading_gt[..., None]

        uv_clean = uv_map_gt.copy()
        uv_clean[uv_clean == -1] = 0
        
        # Ensure shading is clean (masked)
        shading_gt = shading_gt * valid_mask[..., None]

        # Load Flat Ground Truth Texture (For VGG Loss)
        meta_path = self.meta_dir / f"{img_id}.json"
        flat_img = np.zeros(self.size + (3,), dtype=np.uint8) # Default black
        
        if meta_path.exists():
            try:
                with open(meta_path, 'rb') as f: meta = orjson.loads(f.read())
                tex_name = Path(meta.get("texture_name", "")).name
                tex_path = self.tex_dir / tex_name
                if tex_path.exists():
                    tex = cv2.imread(str(tex_path))
                    tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
                    flat_img = cv2.resize(tex, (self.size[1], self.size[0]))
            except: pass
        flat_img = flat_img.astype(np.float32) / 255.0

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
        
        # Resize if needed
        if flat_canny.shape != self.size:
            flat_canny = cv2.resize(flat_canny, (self.size[1], self.size[0]))
        if warped_canny.shape != self.size:
            warped_canny = cv2.resize(warped_canny, (self.size[1], self.size[0]))

        # Load Shadow Clean Warped Image
        clean_warped_img_path = self.shadow_clean_warped_dir / f"{img_id}.png"
        if clean_warped_img_path.exists():
            clean_warped_img = cv2.imread(str(clean_warped_img_path))
            clean_warped_img = cv2.cvtColor(clean_warped_img, cv2.COLOR_BGR2RGB)
            clean_warped_img = cv2.resize(clean_warped_img, (self.size[1], self.size[0])).astype(np.float32) / 255.0
        else:
            clean_warped_img = np.zeros(self.size + (3,), dtype=np.float32)
        
        # Channels: 0,1=UV, 2=Mask, 3=Shading(Luma), 4,5,6=FlatRGB, 7=FlatCanny, 8=WarpedCanny, 9,10,11=CleanWarped
        target = np.concatenate([
            uv_clean, 
            valid_mask[..., None], 
            shading_gt,
            flat_img,
            flat_canny[..., None],
            warped_canny[..., None],
            clean_warped_img
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

# --- 2. Doc3D Dataset (NEW) ---
class Doc3DDataset(Dataset):
    def __init__(self, root_dir, split='train', size=(256, 256), transforms=None):
        """
        root_dir: Path to 'dataset_doc3d' created by build script.
        Expected structure:
          root_dir/
            img/ (distorted input)
            uv/ (exr)
            alb/ (clean warped)
            flat_albedo/ (reconstructed flat)
            shading/ (grayscale)
        """
        self.root = Path(root_dir)
        self.size = size
        self.transforms = transforms
        self.split = split
        
        # Load split
        split_file = self.root / "split.json"
        if split_file.exists():
            with open(split_file, "rb") as f:
                splits = orjson.loads(f.read())
                self.ids = splits.get(split, [])
        else:
            # Fallback if no split file: use all images
            print(f"Warning: Split file not found at {split_file}. Loading all images.")
            self.ids = [f.stem for f in (self.root / "img").glob("*.png")]

        # Directories
        self.img_dir = self.root / "img"
        self.uv_dir = self.root / "uv"
        self.alb_dir = self.root / "alb"
        self.flat_dir = self.root / "flat_albedo"
        self.shading_dir = self.root / "shading"
        
        # Edges (Optional, generated by generate_edges.py)
        # Assuming generate_edges.py output to dataset_doc3d/flat_edges etc.
        self.flat_edges_dir = self.root / "flat_edges"
        self.warped_edges_dir = self.root / "warped_edges"

        # Precompute CoordConv Grid
        h, w = size
        xx_range = np.arange(w, dtype=np.int32)
        yy_range = np.arange(h, dtype=np.int32)
        xx, yy = np.meshgrid(xx_range, yy_range)
        xx = xx / (w - 1) * 2 - 1
        yy = yy / (h - 1) * 2 - 1
        self.grid = np.stack([xx, yy], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # 1. Load Input Image
        img_path = self.img_dir / f"{img_id}.png"
        img = cv2.imread(str(img_path))
        if img is None:
             # Fallback: create black image
             img = np.zeros(self.size + (3,), dtype=np.uint8)
        else:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img.shape[:2] != self.size:
             img = cv2.resize(img, (self.size[1], self.size[0]))

        # 2. Load UV Map (.exr)
        # Build convention (build_uvdoc_unet.py):
        #   ch0 = U [0,1] or -1 for background
        #   ch1 = V [0,1] or -1 for background
        #   ch2 = document mask (~1 inside, 0 outside)
        uv_path = self.uv_dir / f"{img_id}.exr"
        if uv_path.exists():
            uv_raw = cv2.imread(str(uv_path), cv2.IMREAD_UNCHANGED)
            if uv_raw.ndim == 3 and uv_raw.shape[2] >= 3:
                uv_map  = uv_raw[..., :2]   # U, V
                uv_mask = uv_raw[..., 2]    # stored document mask
            elif uv_raw.ndim == 3 and uv_raw.shape[2] == 2:
                uv_map  = uv_raw
                uv_mask = (uv_map[..., 0] >= 0).astype(np.float32)  # -1 sentinel
            else:
                uv_map  = np.zeros(self.size + (2,), dtype=np.float32)
                uv_mask = np.zeros(self.size, dtype=np.float32)
        else:
            uv_map  = np.zeros(self.size + (2,), dtype=np.float32)
            uv_mask = np.zeros(self.size, dtype=np.float32)

        if uv_map.shape[:2] != self.size:
            uv_map  = cv2.resize(uv_map,  (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
            uv_mask = cv2.resize(uv_mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

        # 3. Create Mask from the stored mask channel (matches UVDoc convention)
        mask = (uv_mask > 0.5).astype(np.float32)

        # Zero out background UV values so the target is consistent with UVDocUNetDataset
        # (which does uv_clean[uv_clean == -1] = 0 for the same reason)
        uv_map = uv_map.copy()
        uv_map[mask == 0] = 0.0

        # 4. Load Shading
        shading_path = self.shading_dir / f"{img_id}.png"
        if shading_path.exists():
            shading = cv2.imread(str(shading_path), cv2.IMREAD_GRAYSCALE)
            shading = shading.astype(np.float32) / 255.0
        else:
            shading = np.zeros(self.size, dtype=np.float32)
            
        if shading.shape != self.size:
             shading = cv2.resize(shading, (self.size[1], self.size[0]))
        shading = shading[..., None] * mask[..., None] # Mask shading

        # 5. Load Flat Albedo (Flat RGB)
        flat_path = self.flat_dir / f"{img_id}.png"
        if flat_path.exists():
            flat_img = cv2.imread(str(flat_path))
            flat_img = cv2.cvtColor(flat_img, cv2.COLOR_BGR2RGB)
            flat_img = flat_img.astype(np.float32) / 255.0
        else:
            flat_img = np.zeros(self.size + (3,), dtype=np.float32)
            
        if flat_img.shape[:2] != self.size:
             flat_img = cv2.resize(flat_img, (self.size[1], self.size[0]))

        # 6. Load Clean Warped (Alb)
        alb_path = self.alb_dir / f"{img_id}.png"
        if alb_path.exists():
            clean_warped = cv2.imread(str(alb_path))
            clean_warped = cv2.cvtColor(clean_warped, cv2.COLOR_BGR2RGB)
            clean_warped = clean_warped.astype(np.float32) / 255.0
        else:
            clean_warped = np.zeros(self.size + (3,), dtype=np.float32)
            
        if clean_warped.shape[:2] != self.size:
             clean_warped = cv2.resize(clean_warped, (self.size[1], self.size[0]))

        # 7. Edges (Fallback to zeros if generate_edges wasn't run)
        flat_edge_path = self.flat_edges_dir / f"{img_id}.png"
        if flat_edge_path.exists():
            flat_canny = cv2.imread(str(flat_edge_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            flat_canny = np.zeros(self.size, dtype=np.float32)

        warped_edge_path = self.warped_edges_dir / f"{img_id}.png"
        if warped_edge_path.exists():
            warped_canny = cv2.imread(str(warped_edge_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            warped_canny = np.zeros(self.size, dtype=np.float32)

        if flat_canny.shape != self.size: flat_canny = cv2.resize(flat_canny, (self.size[1], self.size[0]))
        if warped_canny.shape != self.size: warped_canny = cv2.resize(warped_canny, (self.size[1], self.size[0]))

        # Assemble Target
        # Must match UVDocUNetDataset structure exactly
        # Channels: 0,1=UV, 2=Mask, 3=Shading, 4-6=Flat, 7=FlatCanny, 8=WarpedCanny, 9-11=CleanWarped
        target = np.concatenate([
            uv_map,                 # 0-1
            mask[..., None],        # 2
            shading,                # 3
            flat_img,               # 4-6
            flat_canny[..., None],  # 7
            warped_canny[..., None],# 8
            clean_warped            # 9-11
        ], axis=-1)

        # Transforms
        if self.transforms:
            augmented = self.transforms(image=img, mask=target)
            img = augmented['image']
            target = augmented['mask']

        # Preprocessing
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # (3, H, W)
        img = np.concatenate([img, self.grid], axis=0) # (5, H, W)
        
        target = np.transpose(target, (2, 0, 1))

        return torch.from_numpy(img), torch.from_numpy(target)


# --- 3. Model: MobileNetV3-UNet (5-Channel Input) ---
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
        
        center = self.context_bot(e4)
        
        d1 = self.up1(center, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3)
        d5 = self.up5(d4)
        
        # Concatenate grid AND edges before final layer
        d5 = torch.cat([d5, grid, edges], dim=1)
        
        out = self.final(d5)
        return out # Return logits

# --- 4. Loss with Gradients ---
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        for x in range(4): self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18): self.slice3.add_module(str(x), vgg[x])
        
        self.slice1.eval()
        self.slice2.eval()
        self.slice3.eval()
        for p in self.parameters(): p.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_img, target_img):
        # Normalize
        in_norm = (input_img - self.mean) / self.std
        tg_norm = (target_img - self.mean) / self.std
        
        h1 = self.slice1(in_norm); h2 = self.slice1(tg_norm)
        h3 = self.slice2(h1); h4 = self.slice2(h2)
        h5 = self.slice3(h3); h6 = self.slice3(h4)
        
        loss = F.l1_loss(h1, h2) + F.l1_loss(h3, h4) + F.l1_loss(h5, h6)
        return loss

class GeometricLoss(nn.Module):
    def __init__(self, device, scales=None, weights_dict=None):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss() 
        
        # Only instantiate VGG if its weight is greater than 0
        self.weights_dict = weights_dict if weights_dict is not None else weights
        if self.weights_dict.get("VGG", 0) > 0:
            self.perceptual = PerceptualLoss(device).to(device)
        else:
            self.perceptual = None
            
        self.scales = scales if scales is not None else {k: 1.0 for k in weights}
        self.weights_dict = weights_dict if weights_dict is not None else weights
        
    def gradient_loss(self, pred_uv, target_uv, mask):
        # Dy
        dy_pred = torch.abs(pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :])
        dy_gt = torch.abs(target_uv[:, :, 1:, :] - target_uv[:, :, :-1, :])
        # Dx
        dx_pred = torch.abs(pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1])
        dx_gt = torch.abs(target_uv[:, :, :, 1:] - target_uv[:, :, :, :-1])
        
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        loss_dy = torch.abs(dy_pred - dy_gt) * mask_y
        loss_dx = torch.abs(dx_pred - dx_gt) * mask_x
        
        return (loss_dy.sum() + loss_dx.sum()) / (mask_y.sum() + mask_x.sum() + 1e-6)

    def curvature_loss(self, pred_uv, mask):
        dx = pred_uv[:, :, :, 1:] - pred_uv[:, :, :, :-1]
        dx2 = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        dy = pred_uv[:, :, 1:, :] - pred_uv[:, :, :-1, :]
        dy2 = dy[:, :, 1:, :] - dy[:, :, :-1, :]
        
        m_x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
        m_y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
        
        loss_x = torch.mean(torch.abs(dx2) * m_x)
        loss_y = torch.mean(torch.abs(dy2) * m_y)
        return loss_x + loss_y

    def tv_shading_loss(self, shading, mask):
        dh = torch.abs(shading[:, :, 1:, :] - shading[:, :, :-1, :])
        dw = torch.abs(shading[:, :, :, 1:] - shading[:, :, :, :-1])
        
        mask_h = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_w = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        loss_h = (dh * mask_h).sum() / (mask_h.sum() + 1e-6)
        loss_w = (dw * mask_w).sum() / (mask_w.sum() + 1e-6)
        return loss_h + loss_w

    def jacobian_rigidity_loss(self, pred_uv, mask):
        du_dx = pred_uv[:, 0:1, :, 1:] - pred_uv[:, 0:1, :, :-1]
        du_dy = pred_uv[:, 0:1, 1:, :] - pred_uv[:, 0:1, :-1, :]
        dv_dx = pred_uv[:, 1:2, :, 1:] - pred_uv[:, 1:2, :, :-1]
        dv_dy = pred_uv[:, 1:2, 1:, :] - pred_uv[:, 1:2, :-1, :]
        
        diff_u_x = du_dx[:, :, :, 1:] - du_dx[:, :, :, :-1]
        diff_u_y = du_dy[:, :, 1:, :] - du_dy[:, :, :-1, :]
        diff_v_x = dv_dx[:, :, :, 1:] - dv_dx[:, :, :, :-1]
        diff_v_y = dv_dy[:, :, 1:, :] - dv_dy[:, :, :-1, :]
        
        m_x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
        m_y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
        
        loss = (diff_u_x.abs() * m_x).mean() + (diff_u_y.abs() * m_y).mean() + \
               (diff_v_x.abs() * m_x).mean() + (diff_v_y.abs() * m_y).mean()
        return loss

    def structure_loss(self, grid_pred, flat_edge_map, wraped_edge_map, target_mask):
        pred_wraped_edges = F.grid_sample(flat_edge_map, grid_pred, align_corners=True, mode='bilinear')
        loss = self.l1(pred_wraped_edges, wraped_edge_map)
        loss = (loss * target_mask.unsqueeze(1)).sum() / (target_mask.unsqueeze(1).sum() + 1e-6)
        return loss

    def forward(self, pred_logits, target, input_img):
        pred = torch.sigmoid(pred_logits)
        pred_uv = pred[:, :2, :, :]
        pred_shading = pred[:, 3:4, :, :]
        pred_mask_logits = pred_logits[:, 2, :, :]
        
        # Unpack Target
        target_uv = target[:, :2, :, :]
        target_mask = target[:, 2, :, :]
        target_shading = target[:, 3:4, :, :]
        target_flat_img = target[:, 4:7, :, :]
        target_flat_edge_map = target[:, 7, :, :].unsqueeze(1)
        warped_canny_map = target[:, 8, :, :].unsqueeze(1)
        target_clean_warped = target[:, 9:12, :, :]
        
        # 1. Mask Loss
        loss_mask = self.bce(pred_mask_logits, target_mask)
        
        mask_expanded = target_mask.unsqueeze(1).repeat(1, 2, 1, 1)
        
        # 2. UV Loss
        edge_weights = 1.0 + 5.0 * warped_canny_map
        edge_weights_uv = edge_weights.repeat(1, 2, 1, 1)
        loss_uv = self.l1(pred_uv, target_uv)
        loss_uv = (loss_uv * mask_expanded * edge_weights_uv).sum() / (mask_expanded.sum() + 1e-6)
        
        # 3. Grad Loss
        loss_grad_1 = self.gradient_loss(pred_uv, target_uv, mask_expanded)
        pred_uv_2 = F.interpolate(pred_uv, scale_factor=0.5, mode='bilinear', align_corners=True)
        target_uv_2 = F.interpolate(target_uv, scale_factor=0.5, mode='bilinear', align_corners=True)
        mask_2 = F.interpolate(mask_expanded, scale_factor=0.5, mode='nearest')
        loss_grad_2 = self.gradient_loss(pred_uv_2, target_uv_2, mask_2)
        
        pred_uv_4 = F.interpolate(pred_uv, scale_factor=0.25, mode='bilinear', align_corners=True)
        target_uv_4 = F.interpolate(target_uv, scale_factor=0.25, mode='bilinear', align_corners=True)
        mask_4 = F.interpolate(mask_expanded, scale_factor=0.25, mode='nearest')
        loss_grad_4 = self.gradient_loss(pred_uv_4, target_uv_4, mask_4)
        
        loss_grad_total = loss_grad_1 + loss_grad_2 + loss_grad_4
        
        # 4. Shading Loss
        mask_expanded_shading = target_mask.unsqueeze(1)
        loss_shading = self.l1(pred_shading, target_shading)
        loss_shading = (loss_shading * mask_expanded_shading).sum() / (mask_expanded_shading.sum() + 1e-6)
        
        # 5. Reproj & VGG (Clean-to-Clean)
        grid_pred = pred_uv.permute(0, 2, 3, 1) * 2.0 - 1.0 
        recon_input = F.grid_sample(target_flat_img, grid_pred, align_corners=True, mode='bilinear')
        
        mask_3ch = target_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        recon_masked = recon_input * mask_3ch
        input_masked = target_clean_warped * mask_3ch
        
        loss_reproj = (self.l1(recon_masked, input_masked)).sum() / (mask_3ch.sum() + 1e-6)
        
        if self.perceptual is not None:
            loss_vgg = self.perceptual(recon_masked, input_masked)
        else:
            loss_vgg = torch.tensor(0.0, device=pred_logits.device)
        
        # 6. Stiffness & TV & Rigid
        loss_stiffness = self.curvature_loss(pred_uv, mask_expanded)
        loss_tv_shading = self.tv_shading_loss(pred_shading, target_mask.unsqueeze(1))
        loss_rigidity = self.jacobian_rigidity_loss(pred_uv, target_mask.unsqueeze(1))

        # 7. Structure Loss
        loss_structure = self.structure_loss(grid_pred, target_flat_edge_map, warped_canny_map, target_mask.unsqueeze(1))

        component_tensors = {
            "Mask": loss_mask,
            "UV": loss_uv,
            "Grad": loss_grad_total,
            "Shading": loss_shading,
            "Stiffness": loss_stiffness,
            "TV": loss_tv_shading,
            "Rigid": loss_rigidity,
            "Reproj": loss_reproj,
            "VGG": loss_vgg,
            "Structure": loss_structure
        }

        components = {k: v.item() for k, v in component_tensors.items()}

        total_loss = sum(w * self.scales.get(k, 1.0) * component_tensors[k] for k, w in self.weights_dict.items()) / sum(self.weights_dict.values())
        
        debug_info = {
            "recon_masked": recon_masked,
            "pred_uv": pred_uv,
            "target_mask": target_mask
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

def calibrate_loss_scales(model, train_loader, device, num_steps=1000, adaptive=False, threshold=0.02, max_steps=5000):
    import copy
    from collections import deque
    
    print(f"Starting Calibration Phase (min steps: {num_steps}, adaptive: {adaptive})...")
    temp_model = copy.deepcopy(model).to(device)
    temp_model.train()
    optimizer = optim.AdamW(temp_model.parameters(), lr=1e-3)
    
    # Use 1.0 weights for all components during calibration
    # to ensure each component contributes equally to gradient updates.
    calib_weights = {k: 1.0 if weights[k] > 0 else 0.0 for k in weights.keys()}
    criterion = GeometricLoss(device=device, scales=None, weights_dict=calib_weights).to(device)
    
    sums = {k: 0.0 for k in weights.keys()}
    count = 0
    
    # History for stability tracking (every 100 steps)
    chunk_size = 100
    history = deque(maxlen=5) 
    
    pbar = tqdm(total=num_steps if not adaptive else None, desc="Calibrating")
    step = 0
    
    stop = False
    while not stop:
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = temp_model(imgs)
            loss, components, _ = criterion(preds, targets, imgs)
            loss.backward()
            optimizer.step()
            
            for k, v in components.items():
                sums[k] += v
            
            count += 1
            step += 1
            pbar.update(1)
            
            # Check for stability every chunk_size steps
            if step % chunk_size == 0:
                # Running average of means
                current_means = {k: v / count for k, v in sums.items()}
                
                if adaptive and step >= num_steps:
                    history.append(current_means)
                    if len(history) >= 2:
                        # Compare normalized scales (relative importance)
                        def get_norm_scales(m):
                            s = {k: 1.0 / (val + 1e-8) for k, val in m.items()}
                            total = sum(s.values())
                            return {k: v/total for k,v in s.items()}
                        
                        s_prev = get_norm_scales(history[-2])
                        s_curr = get_norm_scales(history[-1])
                        
                        # Calculate max relative change across all components
                        max_change = max(abs(s_curr[k] - s_prev[k]) / (s_prev[k] + 1e-8) for k in s_curr)
                        pbar.set_postfix({"rel_change": f"{max_change:.4f}"})
                        
                        if max_change < threshold:
                            print(f"\nStable! Max relative change {max_change:.4f} < {threshold} after {step} steps.")
                            stop = True
                            break
                
                # Fixed mode or max safety limit
                if not adaptive and step >= num_steps:
                    stop = True
                    break
                if step >= max_steps:
                    print(f"\nReached max_steps ({max_steps}) limit.")
                    stop = True
                    break
        if stop: break

    pbar.close()
    
    final_means = {k: v / count for k, v in sums.items()}
    scales = {k: 1.0 / (v + 1e-8) for k, v in final_means.items()}
    print(f"Calibration complete after {step} steps.")
    
    # Explicitly free memory
    del temp_model
    del optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return scales

def train_unet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Handle VGG disable
    if args.no_vgg:
        weights["VGG"] = 0.0
        print("VGG Perceptual loss disabled.")

    # 1. Prepare Transforms
    transforms_train = get_train_transforms(args)

    # 2. Instantiate Datasets
    # A. UVDoc (Existing)
    uvdoc_ds_train = UVDocUNetDataset(args.data_dir, 'train', transforms=transforms_train)
    uvdoc_ds_val = UVDocUNetDataset(args.data_dir, 'val')
    
    datasets_train = [uvdoc_ds_train]
    datasets_val = [uvdoc_ds_val]
    
    # B. Doc3D (New - Check if exists)
    doc3d_root = Path(args.doc3d_dir)
    if doc3d_root.exists():
        print(f"Found Doc3D Dataset at {doc3d_root}. merging...")
        doc3d_ds_train = Doc3DDataset(doc3d_root, 'train', transforms=transforms_train)
        doc3d_ds_val = Doc3DDataset(doc3d_root, 'val')
        
        datasets_train.append(doc3d_ds_train)
        datasets_val.append(doc3d_ds_val)
    else:
        print(f"Doc3D not found at {doc3d_root}. Using UVDoc only.")

    # 3. Concat
    train_ds = ConcatDataset(datasets_train)
    val_ds = ConcatDataset(datasets_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    model = MobileNetV3UNet(num_classes=4).to(device)
    
    # --- Calibration / Scaling ---
    scales_path = Path(args.scales_file)
    if args.calibrate or args.adaptive_calibrate or not scales_path.exists():
        scales = calibrate_loss_scales(model, train_loader, device, num_steps=1000, adaptive=args.adaptive_calibrate)
        with open(scales_path, 'w') as f:
            json.dump(scales, f, indent=4)
    else:
        with open(scales_path, 'r') as f:
            scales = json.load(f)

    for k, v in weights.items():
        print(f"  {k}: Scale={scales[k]:.4f} * Weight={v:.1f} = {scales[k]*v:.4f}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = GeometricLoss(device=device, scales=scales, weights_dict=weights).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    print("------ Training Config ------\n" \
        f"Train Size: {len(train_ds)} (UVDoc + Doc3D)\n" \
        f"Val Size: {len(val_ds)}\n" \
        f"Batch Size: {args.batch}\n" \
        f"Accumulation: {args.accumulate}\n" \
        )
    print(f"Training for {args.epochs} epochs...")
    best_val = float('inf')
    
    scaler = torch.cuda.amp.GradScaler() 
    global_step = 0
    accumulated_components = {}
    verbose_dir = None

    if args.verbose:
        verbose_dir = Path(args.verbose_dir)
        verbose_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss, components, debug_info = criterion(preds, targets, imgs)
                accum_loss = loss / args.accumulate
            
            scaler.scale(accum_loss).backward()
            
            if (i + 1) % args.accumulate == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            for k, v in components.items():
                accumulated_components[k] = accumulated_components.get(k, 0.0) + v
            
            global_step += 1
            if global_step % args.log_interval == 0:
                avg_comp = {k: v / args.log_interval for k, v in accumulated_components.items()}
                log_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_comp.items()])
                print(f"Step {global_step} | {log_str}")
                accumulated_components = {}
                
                if args.verbose and verbose_dir is not None:
                    # Viz: Input | UV | Recon
                    input_img_vis = imgs[0, :3].detach().cpu().numpy().transpose(1, 2, 0)
                    recon = debug_info["recon_masked"][0].detach().cpu().numpy().transpose(1, 2, 0)
                    pred_uv_vis = debug_info["pred_uv"][0].detach().cpu().numpy()
                    mask_vis = debug_info["target_mask"][0].detach().cpu().numpy()
                    
                    uv_img = np.zeros_like(input_img_vis)
                    uv_img[..., 0] = pred_uv_vis[0] * mask_vis
                    uv_img[..., 1] = pred_uv_vis[1] * mask_vis
                    
                    vis = np.concatenate([input_img_vis, uv_img, recon], axis=1)
                    vis = (vis * 255).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(str(verbose_dir / f"step_{global_step:06d}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
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
    parser.add_argument("--doc3d_dir", default="dataset_doc3d", help="Path to processed Doc3D dataset")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_vgg", action='store_true', help="Disable VGG perceptual loss to save VRAM and speed up training")
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--aug_color", type=float, default=0.2)
    parser.add_argument("--aug_noise_std", type=float, default=0.2)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--calibrate", action='store_true')
    parser.add_argument("--adaptive_calibrate", action='store_true', help="Calibrate until loss scales stabilize")
    parser.add_argument("--scales_file", default="loss_scales.json")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--verbose_dir", default="train_vis_debug")
    args = parser.parse_args()
    
    train_unet(args)
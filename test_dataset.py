import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from train_unet import UVDocUNetDataset, get_train_transforms

def visualize_dataset(data_dir, split, n_samples, args=None):
    transforms = None
    if args and args.augment:
        transforms = get_train_transforms(args)
        
    dataset = UVDocUNetDataset(data_dir, split=split, transforms=transforms)
    
    if len(dataset) == 0:
        print(f"No samples found in {data_dir}/{split}")
        return

    indices = np.random.choice(len(dataset), min(len(dataset), n_samples), replace=False)
    
    for i, idx in enumerate(indices):
        print(f"Visualizing sample {idx}...")
        img_tensor, target_tensor = dataset[idx]
        
        # Process Image: (5, H, W) -> (H, W, 3) RGB
        # Channel 0,1,2 are RGB. 3,4 are CoordConv.
        img = img_tensor[:3, :, :].permute(1, 2, 0).numpy()
        # It was normalized / 255.0 in dataset.
        
        # Process Target: (9, H, W) -> (H, W, C)
        # Channels: 0,1=UV, 2=Mask, 3=Shading(Luma), 4,5,6=FlatRGB, 7=FlatCanny, 8=WarpedCanny
        target = target_tensor.permute(1, 2, 0).numpy()
        
        uv_map = target[..., :2]
        mask = target[..., 2]
        shading = target[..., 3:4]
        flat_img = target[..., 4:7]
        flat_canny = target[..., 7]
        warped_canny = target[..., 8]
        
        # Visualize UV Map (encoded as colors for visibility)
        # Standard UV visualization: Red=U, Green=V, Blue=0
        uv_vis = np.zeros((*shading.shape[:2], 3), dtype=np.float32)
        uv_vis[..., 0] = uv_map[..., 0]
        uv_vis[..., 1] = uv_map[..., 1]
        
        # Calculate Albedo
        # Avoid division by zero
        shading_safe = np.clip(shading, 0.05, 1.0)
        albedo = img / shading_safe
        albedo = np.clip(albedo, 0, 1)
        
        # Mask Albedo/Shading/UV with the validity mask
        valid_mask = (mask > 0.5)
        
        uv_vis[~valid_mask] = 0
        shading_vis = shading.copy()
        shading_vis[~valid_mask] = 0
        albedo[~valid_mask] = 0
        
        # Plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        axes[0].imshow(np.clip(img, 0, 1))
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        axes[1].imshow(np.clip(uv_vis, 0, 1))
        axes[1].set_title("GT UV Map (R=U, G=V)")
        axes[1].axis('off')
        
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("GT Mask")
        axes[2].axis('off')
        
        axes[3].imshow(np.clip(shading_vis[..., 0], 0, 1), cmap='gray')
        axes[3].set_title("GT Shading (Luma)")
        axes[3].axis('off')
        
        axes[4].imshow(flat_canny, cmap='magma')
        axes[4].set_title("Flat Canny Map")
        axes[4].axis('off')
        
        axes[5].imshow(warped_canny, cmap='magma')
        axes[5].set_title("Warped Canny Map")
        axes[5].axis('off')
        
        axes[6].imshow(np.clip(albedo, 0, 1))
        axes[6].set_title("Derived Albedo")
        axes[6].axis('off')
        
        axes[7].imshow(np.clip(flat_img, 0, 1))
        axes[7].set_title("GT Flat Texture")
        axes[7].axis('off')
        
        out_path = f"test_sample_{i}_{split}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_unet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=3)
    
    # Augmentation args
    parser.add_argument("--augment", action="store_true", help="Apply augmentations")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--aug_color", type=float, default=0.2, help="Strength of color jitter")
    parser.add_argument("--aug_noise_std", type=float, default=0.2, help="Max standard deviation for Gaussian noise (0-1 range)")
    
    args = parser.parse_args()
    
    visualize_dataset(args.data_dir, args.split, args.n, args)


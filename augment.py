"""
Augmentations for backward-map training.

Spatial augmentation must transform the backward-map *values*, not just move label pixels:
if the image is warped by a homography H (old normalized coords -> new normalized coords),
then img_new(p) = img_old(H^-1 p) and bm_new = H(bm_old).
Page-space tensors (flat, flat_edges) are unchanged.
"""
import math

import albumentations as A
import torch
import torch.nn.functional as F

from models import coord_grid

# Batch keys that live in the warped image space
IMAGE_SPACE_KEYS = ("img", "warped", "warped_edges")


def get_color_transforms(aug_color: float, aug_noise_std: float, p: float) -> A.Compose:
    """Photometric augmentation of the input image only (uint8 HWC)."""
    return A.Compose([
        A.ColorJitter(brightness=aug_color, contrast=aug_color, saturation=aug_color, hue=aug_color / 2, p=p),
        A.GaussNoise(std_range=(0.0, aug_noise_std), mean_range=(0.0, 0.0), p=p),
    ])


def random_homography(b: int, device, p: float = 0.5, max_rot_deg: float = 15.0, scale=(0.9, 1.1),
                      max_trans: float = 0.125, max_persp: float = 0.1) -> torch.Tensor:
    """(B, 3, 3) homographies in normalized [-1, 1] coords; identity with probability 1 - p."""
    def u(lo, hi):
        return torch.empty(b, device=device).uniform_(lo, hi)

    ang = u(-1, 1) * math.radians(max_rot_deg)
    s = u(*scale)
    tx, ty = u(-max_trans, max_trans), u(-max_trans, max_trans)
    px, py = u(-max_persp, max_persp), u(-max_persp, max_persp)

    H = torch.zeros(b, 3, 3, device=device)
    H[:, 0, 0], H[:, 0, 1], H[:, 0, 2] = s * torch.cos(ang), -s * torch.sin(ang), tx
    H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = s * torch.sin(ang), s * torch.cos(ang), ty
    H[:, 2, 0], H[:, 2, 1], H[:, 2, 2] = px, py, 1.0

    keep = torch.rand(b, device=device) >= p
    H[keep] = torch.eye(3, device=device)
    return H


def warp_points(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Applies (B, 3, 3) homographies to points (B, 2, ...) in normalized coords."""
    shape = pts.shape
    flat = pts.reshape(shape[0], 2, -1)
    homo = torch.cat([flat, torch.ones_like(flat[:, :1])], dim=1)
    out = H @ homo
    return (out[:, :2] / out[:, 2:3].clamp(min=1e-6)).reshape(shape)


def apply_homography(batch: dict, H: torch.Tensor) -> dict:
    """Returns a new batch with image-space tensors warped and the backward map transformed."""
    out = dict(batch)
    b, _, h, w = batch["img"].shape
    src = warp_points(torch.linalg.inv(H), coord_grid(b, h, w, H.device).contiguous())
    grid = src.permute(0, 2, 3, 1)
    for k in IMAGE_SPACE_KEYS:
        out[k] = F.grid_sample(batch[k], grid.to(batch[k].dtype), mode="bilinear",
                               padding_mode="zeros", align_corners=True)
    out["bm"] = warp_points(H, batch["bm"])
    return out

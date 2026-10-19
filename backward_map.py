"""
Backward-map (BM) conventions shared by dataset building, training, evaluation and inference.

A backward map has shape (2, GH, GW). Entry [:, i, j] is the position in the *warped input image*
of the flat-page point (u = j / (GW - 1), v = i / (GH - 1)).
Channel 0 is x, channel 1 is y, both normalized to [-1, 1] with align_corners=True semantics,
i.e. -1 is the center of the first pixel and +1 the center of the last pixel.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import griddata

BM_SIZE = (32, 32)  # (rows, cols)


def resize_bm(bm: np.ndarray, size: Tuple[int, int] = BM_SIZE) -> np.ndarray:
    """Resizes a (2, h, w) backward map with corner-preserving bilinear interpolation."""
    if bm.shape[1:] == tuple(size):
        return bm.astype(np.float32)
    t = torch.from_numpy(bm.astype(np.float32)).unsqueeze(0)
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=True)
    return t.squeeze(0).numpy()


def bm_from_point_grid(points_xy: np.ndarray, img_w: int, img_h: int, size: Tuple[int, int] = BM_SIZE) -> np.ndarray:
    """
    points_xy: (gh, gw, 2) pixel coordinates in an image of size (img_w, img_h),
    laid out on a regular flat-page grid (e.g. UVDoc grid2d).
    """
    x = points_xy[..., 0] / (img_w - 1) * 2.0 - 1.0
    y = points_xy[..., 1] / (img_h - 1) * 2.0 - 1.0
    return resize_bm(np.stack([x, y], axis=0), size)


PageAffine = Tuple[float, float, float, float]

# Doc3D's bm grid covers a slightly smaller page range than its UV texture coordinates.
# Fitted against original Doc3D bm (u' = a*u + b, v' = c*v + d): reduces the inversion error
# from 1.24px to 0.32px (256px image); re-check with build_backward_maps.py --verify_doc3d N.
DOC3D_PAGE_AFFINE: PageAffine = (0.99604, 0.00688, 0.99577, 0.00629)


def _extrapolate_border(bm: np.ndarray) -> None:
    """In-place linear extrapolation of missing outer rows/cols of a (GH, GW, 2) grid."""
    def fill(dst, a, b):
        m = np.isnan(dst[..., 0]) & ~np.isnan(a[..., 0]) & ~np.isnan(b[..., 0])
        dst[m] = 2.0 * a[m] - b[m]

    fill(bm[0], bm[1], bm[2])
    fill(bm[-1], bm[-2], bm[-3])
    fill(bm[:, 0], bm[:, 1], bm[:, 2])
    fill(bm[:, -1], bm[:, -2], bm[:, -3])


def invert_forward_uv(u: np.ndarray, v: np.ndarray, mask: np.ndarray, size: Tuple[int, int] = BM_SIZE,
                      stride: int = 2, page_affine: PageAffine = (1.0, 0.0, 1.0, 0.0)) -> Optional[np.ndarray]:
    """
    Inverts a dense forward map (image pixel -> page uv in [0, 1]) into a backward map.
    Boundary pixels are dropped (eroded mask) because resized UV maps blend with the background there.
    Sampled uv never reaches exactly 0 / 1, so the outer grid ring lies outside the convex hull of the
    samples; it is linearly extrapolated, remaining holes get their nearest neighbour.
    page_affine maps the bm page grid into the dataset's uv convention.
    """
    h, w = mask.shape
    valid = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    sub = np.zeros_like(valid)
    sub[::stride, ::stride] = valid[::stride, ::stride]
    ys, xs = np.nonzero(sub)
    if len(xs) < 16:
        return None

    points = np.stack([u[ys, xs], v[ys, xs]], axis=1).astype(np.float64)
    values = np.stack([xs / (w - 1) * 2.0 - 1.0, ys / (h - 1) * 2.0 - 1.0], axis=1)

    a, b, c, d = page_affine
    gu, gv = np.meshgrid(a * np.linspace(0, 1, size[1]) + b, c * np.linspace(0, 1, size[0]) + d)
    try:
        bm = griddata(points, values, (gu, gv), method="linear")
        _extrapolate_border(bm)
        holes = np.isnan(bm[..., 0])
        if holes.any():
            bm[holes] = griddata(points, values, (gu[holes], gv[holes]), method="nearest")
    except Exception:
        return None
    return bm.transpose(2, 0, 1).astype(np.float32)


def bm_to_dense(bm: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """(B, 2, GH, GW) -> sampling grid (B, H, W, 2) for F.grid_sample(align_corners=True)."""
    dense = F.interpolate(bm, size=size, mode="bilinear", align_corners=True)
    return dense.permute(0, 2, 3, 1)


def dewarp(img: torch.Tensor, bm: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Samples the flat page from img (B, C, H, W) using bm (B, 2, GH, GW)."""
    size = size or tuple(img.shape[-2:])
    grid = bm_to_dense(bm.to(img.dtype), size)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def bm_to_remap(bm: np.ndarray, out_w: int, out_h: int, src_w: int, src_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """(2, GH, GW) backward map -> cv2.remap pixel maps of size (out_h, out_w) into a (src_w, src_h) image."""
    dense = bm_to_dense(torch.from_numpy(bm.astype(np.float32)).unsqueeze(0), (out_h, out_w))[0].numpy()
    map_x = (dense[..., 0] + 1.0) * 0.5 * (src_w - 1)
    map_y = (dense[..., 1] + 1.0) * 0.5 * (src_h - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def bm_outline(bm: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """Page border of a (2, GH, GW) backward map as a closed (N, 2) pixel polygon in the source image."""
    x = (bm[0] + 1.0) * 0.5 * (src_w - 1)
    y = (bm[1] + 1.0) * 0.5 * (src_h - 1)
    pts = np.stack([x, y], axis=-1)
    border = np.concatenate([pts[0, :], pts[1:, -1], pts[-1, -2::-1], pts[-2:0:-1, 0]], axis=0)
    return border.astype(np.float32)


def grid_error_px(pred: torch.Tensor, target: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """Per-sample mean euclidean grid error in pixels of an (H, W) image."""
    h, w = img_size
    scale = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], device=pred.device, dtype=pred.dtype).view(1, 2, 1, 1)
    return torch.linalg.norm((pred - target) * scale, dim=1).mean(dim=(1, 2))

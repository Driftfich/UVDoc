import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from train_unet import Doc3DDataset, UVDocUNetDataset, get_train_transforms


class LabeledDataset(Dataset):
    """Wrap a dataset to expose source name and sample id."""

    def __init__(self, base_dataset: Dataset, source_name: str):
        self.base = base_dataset
        self.source_name = source_name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        sample_id = str(idx)
        if hasattr(self.base, "ids") and idx < len(self.base.ids):
            sample_id = self.base.ids[idx]
        return x, y, self.source_name, sample_id


def to_u8_gray(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def to_u8_rgb(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def colorize_uv(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    uv_vis = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    uv_vis[..., 0] = to_u8_gray(u)  # R <- U
    uv_vis[..., 1] = to_u8_gray(v)  # G <- V
    uv_vis[..., 2] = 255 - to_u8_gray(0.5 * (u + v))
    uv_vis[mask < 0.5] = 0
    return uv_vis


def put_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def tile_bgr(img_u8_rgb: np.ndarray, title: str) -> np.ndarray:
    return put_title(cv2.cvtColor(img_u8_rgb, cv2.COLOR_RGB2BGR), title)


def tile_gray_bgr(gray_u8: np.ndarray, title: str) -> np.ndarray:
    return put_title(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), title)


def make_sample_panel(
    sample_x: torch.Tensor,
    sample_y: torch.Tensor,
    source: str,
    sample_id: str,
) -> tuple[np.ndarray, dict]:
    # Input: x -> (5, H, W): RGB + CoordX + CoordY
    # Target: y -> (12, H, W)
    x_np = sample_x.cpu().numpy()
    y_np = sample_y.cpu().numpy()

    rgb = np.transpose(x_np[:3], (1, 2, 0))
    coord_x = x_np[3]
    coord_y = x_np[4]

    uv_u = y_np[0]
    uv_v = y_np[1]
    mask = y_np[2]
    shading = y_np[3]
    flat = np.transpose(y_np[4:7], (1, 2, 0))
    flat_edge = y_np[7]
    warped_edge = y_np[8]
    clean_warped = np.transpose(y_np[9:12], (1, 2, 0))

    # Same reprojection operation used in training loss:
    # grid_pred = target_uv * 2 - 1, recon = grid_sample(target_flat, grid_pred)
    flat_t = torch.from_numpy(np.transpose(flat, (2, 0, 1))).unsqueeze(0).float()
    uv_t = torch.from_numpy(np.stack([uv_u, uv_v], axis=0)).unsqueeze(0).float()
    grid = uv_t.permute(0, 2, 3, 1) * 2.0 - 1.0
    recon = F.grid_sample(flat_t, grid, align_corners=True, mode="bilinear")
    recon = recon.squeeze(0).permute(1, 2, 0).numpy()
    recon_masked = recon * mask[..., None]
    clean_masked = clean_warped * mask[..., None]
    recon_diff = np.mean(np.abs(recon_masked - clean_masked), axis=2)

    coord_x_vis = (coord_x + 1.0) * 0.5
    coord_y_vis = (coord_y + 1.0) * 0.5
    uv_vis = colorize_uv(uv_u, uv_v, mask)

    tiles = [
        tile_bgr(to_u8_rgb(rgb), "Input RGB"),
        tile_gray_bgr(to_u8_gray(coord_x_vis), "Coord X"),
        tile_gray_bgr(to_u8_gray(coord_y_vis), "Coord Y"),
        tile_gray_bgr(to_u8_gray(mask), "Mask"),
        tile_bgr(uv_vis, "UV (U/G, V/RGB)"),
        tile_gray_bgr(to_u8_gray(shading), "Shading"),
        tile_bgr(to_u8_rgb(flat), "Flat GT"),
        tile_gray_bgr(to_u8_gray(flat_edge), "Flat Edge"),
        tile_gray_bgr(to_u8_gray(warped_edge), "Warped Edge"),
        tile_bgr(to_u8_rgb(clean_warped), "Clean Warped GT"),
        tile_bgr(to_u8_rgb(recon_masked), "Reproj(Flat, UV)"),
        tile_gray_bgr(to_u8_gray(np.clip(recon_diff * 4.0, 0.0, 1.0)), "Reproj AbsDiff x4"),
    ]

    h, w = tiles[0].shape[:2]
    top = np.hstack(tiles[:6])
    bot = np.hstack(tiles[6:12])
    panel = np.vstack([top, bot])
    panel = put_title(panel, f"source={source} | id={sample_id}")

    stats = {
        "source": source,
        "sample_id": sample_id,
        "mask_mean": float(mask.mean()),
        "uv_u_minmax": [float(uv_u.min()), float(uv_u.max())],
        "uv_v_minmax": [float(uv_v.min()), float(uv_v.max())],
        "reproj_mae_masked": float(recon_diff[mask > 0.5].mean()) if np.any(mask > 0.5) else None,
    }
    return panel, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvdoc_dir", default="dataset_unet")
    parser.add_argument("--doc3d_dir", default="dataset_doc3d")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="train_step_data_debug")
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--aug_color", type=float, default=0.2)
    parser.add_argument("--aug_noise_std", type=float, default=0.2)
    parser.add_argument("--no_aug", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transforms_train = None
    if args.split == "train" and not args.no_aug:
        transforms_train = get_train_transforms(args)

    datasets = []
    uvdoc = UVDocUNetDataset(args.uvdoc_dir, args.split, transforms=transforms_train)
    datasets.append(LabeledDataset(uvdoc, "uvdoc"))

    doc3d_root = Path(args.doc3d_dir)
    if doc3d_root.exists():
        doc3d = Doc3DDataset(args.doc3d_dir, args.split, transforms=transforms_train)
        datasets.append(LabeledDataset(doc3d, "doc3d"))

    concat_ds = ConcatDataset(datasets)
    loader = DataLoader(
        concat_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    batch = next(iter(loader))
    xs, ys, sources, sample_ids = batch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {
        "split": args.split,
        "batch_size": int(xs.shape[0]),
        "samples": [],
    }

    for i in range(xs.shape[0]):
        panel, stats = make_sample_panel(xs[i], ys[i], str(sources[i]), str(sample_ids[i]))
        out_path = out_dir / f"step_sample_{i:02d}_{sources[i]}_{sample_ids[i]}.png"
        cv2.imwrite(str(out_path), panel)
        all_stats["samples"].append(stats)

    with open(out_dir / "step_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"Saved {len(all_stats['samples'])} panels to: {out_dir}")
    print(f"Saved metadata to: {out_dir / 'step_metadata.json'}")


if __name__ == "__main__":
    main()

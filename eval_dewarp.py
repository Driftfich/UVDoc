"""
Evaluates dewarping quality on the validation splits.

Metrics per dataset:
  grid_px  mean euclidean backward-grid error in pixels (256px input)
  recon    L1 between dewarped clean page and flat page
  msssim   MS-SSIM between dewarped clean page and flat page

  python eval_dewarp.py --model unet_bm_best.pth
  python eval_dewarp.py --model unet_best.pth --legacy     # old forward-UV model, inverted with griddata
"""
import argparse
from collections import defaultdict

import torch
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Subset

from backward_map import BM_SIZE, dewarp, grid_error_px, invert_forward_uv
from datasets import Doc3DDataset, Inv3DDataset, UVDocDataset
from models import load_fastvit_unet


@torch.no_grad()
def predict_bm(model, img: torch.Tensor, legacy: bool) -> torch.Tensor:
    pred = model(img).float()
    if not legacy:
        return pred
    # Forward UV + mask -> backward map (same inversion as used for the GT targets)
    prob = torch.sigmoid(pred).cpu().numpy()
    bms = []
    for p in prob:
        bm = invert_forward_uv(p[0], p[1], p[2] > 0.5, BM_SIZE)
        if bm is None:
            bm = torch.zeros(2, *BM_SIZE).numpy()
        bms.append(torch.from_numpy(bm))
    return torch.stack(bms).to(img.device)


def evaluate(model, dataset, device, legacy: bool, batch: int):
    sums, count, flat_count = defaultdict(float), 0, 0
    for b in DataLoader(dataset, batch_size=batch, num_workers=4):
        b = {k: v.to(device) for k, v in b.items()}
        bm = predict_bm(model, b["img"], legacy)
        sums["grid_px"] += grid_error_px(bm, b["bm"], tuple(b["img"].shape[-2:])).sum().item()
        count += len(bm)

        has = b["has_flat"] > 0
        if has.any():
            rec = dewarp(b["warped"][has], bm[has]).clamp(0, 1)
            flat = b["flat"][has]
            sums["recon"] += (rec - flat).abs().flatten(1).mean(1).sum().item()
            sums["msssim"] += ms_ssim(rec, flat, data_range=1.0, size_average=False).sum().item()
            flat_count += int(has.sum())
    res = {"n": count, "grid_px": sums["grid_px"] / max(count, 1)}
    if flat_count:
        res["recon"] = sums["recon"] / flat_count
        res["msssim"] = sums["msssim"] / flat_count
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet_bm_best.pth")
    parser.add_argument("--legacy", action="store_true", help="Checkpoint is the old forward-UV model")
    parser.add_argument("--data_dir", default="dataset_unet")
    parser.add_argument("--doc3d_dir", default="dataset_doc3d")
    parser.add_argument("--inv3d_dir", default="dataset_inv3d")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_samples", type=int, default=500, help="Per dataset (0 = all)")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fastvit_unet(args.model, backward_map=not args.legacy, device=device)

    datasets = {
        "UVDoc": (UVDocDataset, args.data_dir),
        "Doc3D": (Doc3DDataset, args.doc3d_dir),
        "Inv3D": (Inv3DDataset, args.inv3d_dir),
    }
    g = torch.Generator().manual_seed(0)
    for name, (cls, root) in datasets.items():
        try:
            ds = cls(root, args.split)
        except FileNotFoundError:
            continue
        if len(ds) == 0:
            continue
        if args.max_samples and len(ds) > args.max_samples:
            ds = Subset(ds, torch.randperm(len(ds), generator=g)[:args.max_samples].tolist())
        res = evaluate(model, ds, device, args.legacy, args.batch)
        print(f"{name:6s} " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))


if __name__ == "__main__":
    main()

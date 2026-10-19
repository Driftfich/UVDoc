"""
Trains FastViTUNet to regress the backward map (see backward_map.py).

  python build_backward_maps.py        # once, creates bm/ targets
  python train_unet.py --epochs 120
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from augment import apply_homography, get_color_transforms, random_homography
from backward_map import dewarp, grid_error_px
from datasets import Doc3DDataset, Inv3DDataset, UVDocDataset
from losses import DEFAULT_WEIGHTS, BackwardMapLoss
from models import FastViTUNet


def build_datasets(args, split, color_transforms=None):
    candidates = [
        (UVDocDataset, args.data_dir),
        (Doc3DDataset, args.doc3d_dir),
        (Inv3DDataset, args.inv3d_dir),
    ]
    datasets = []
    for cls, root in candidates:
        if not Path(root).exists():
            print(f"{cls.__name__}: {root} not found, skipped")
            continue
        ds = cls(root, split, color_transforms=color_transforms)
        print(f"{cls.__name__} [{split}]: {len(ds)} samples")
        if len(ds) > 0:
            datasets.append(ds)
    return ConcatDataset(datasets)


def lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 1e-3):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return f


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, err_sum, n = 0.0, 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(batch["img"])
        loss, _ = criterion(pred, batch)
        bs = batch["img"].shape[0]
        loss_sum += loss.item() * bs
        err_sum += grid_error_px(pred.float(), batch["bm"], tuple(batch["img"].shape[-2:])).sum().item()
        n += bs
    return loss_sum / max(n, 1), err_sum / max(n, 1)


def save_debug(batch, pred, path: Path):
    """Input | dewarped input (prediction) | flat target"""
    img = batch["img"][:1].float()
    rec = dewarp(img, pred[:1].float())
    row = torch.cat([img, rec, batch["flat"][:1].float()], dim=3)[0]
    vis = (row.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    color = get_color_transforms(args.aug_color, args.aug_noise_std, args.aug_prob)
    train_ds = build_datasets(args, "train", color)
    val_ds = build_datasets(args, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    model = FastViTUNet(backward_map=True).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.accumulate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    weights = dict(DEFAULT_WEIGHTS)
    for k in weights:
        override = getattr(args, f"w_{k.lower()}")
        if override is not None:
            weights[k] = override
    criterion = BackwardMapLoss(weights)

    out_path = Path(args.out)
    last_path = out_path.with_name(out_path.stem + "_last.pth")
    start_epoch, best_err = 0, float("inf")
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_err = ckpt["epoch"] + 1, ckpt["best_err"]
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    debug_dir = Path(args.verbose_dir) if args.verbose else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {args.batch}x{args.accumulate} | Weights: {weights}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        recon_scale = min(1.0, (epoch + 1) / max(1, args.recon_warmup_epochs))
        running = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            batch = to_device(batch, device)
            if not args.no_spatial_aug:
                H = random_homography(batch["img"].shape[0], device, p=args.aug_prob)
                batch = apply_homography(batch, H)

            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(batch["img"])
            loss, comps = criterion(pred, batch, recon_scale)
            scaler.scale(loss / args.accumulate).backward()

            if (i + 1) % args.accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update_parameters(model)

            for k, v in comps.items():
                running[k] = running.get(k, 0.0) + v
            if (i + 1) % args.log_interval == 0:
                pbar.set_postfix({k: f"{v / args.log_interval:.4f}" for k, v in running.items()})
                running = {}
                if debug_dir:
                    save_debug(batch, pred, debug_dir / f"e{epoch + 1:03d}_s{i + 1:06d}.png")

        val_loss, val_err = validate(ema.module, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: val loss {val_loss:.4f} | val grid error {val_err:.2f}px "
              f"| lr {scheduler.get_last_lr()[0]:.2e}")

        if val_err < best_err:
            best_err = val_err
            torch.save(ema.module.state_dict(), out_path)
            print(f"  -> saved EMA weights to {out_path}")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_err": best_err}, last_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_unet")
    parser.add_argument("--doc3d_dir", default="dataset_doc3d")
    parser.add_argument("--inv3d_dir", default="dataset_inv3d")
    parser.add_argument("--out", default="unet_bm_best.pth", help="Best EMA weights (selected by val grid error)")
    parser.add_argument("--resume", action="store_true", help="Resume from <out>_last.pth")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--recon_warmup_epochs", type=int, default=3, help="Linear ramp of Recon/Edge losses")
    parser.add_argument("--batch", type=int, default=56)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--no_spatial_aug", action="store_true", help="Disable homography augmentation")
    parser.add_argument("--aug_color", type=float, default=0.2)
    parser.add_argument("--aug_noise_std", type=float, default=0.2)
    for k in DEFAULT_WEIGHTS:
        parser.add_argument(f"--w_{k.lower()}", type=float, default=None, help=f"Loss weight {k} (default {DEFAULT_WEIGHTS[k]})")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--verbose", action="store_true", help="Save input | prediction | flat debug images")
    parser.add_argument("--verbose_dir", default="train_vis_debug")
    train(parser.parse_args())
    
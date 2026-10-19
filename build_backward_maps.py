"""
Creates backward-map (BM) targets (2, 32, 32) by inverting the stored forward UV maps.

  Doc3D : dataset_doc3d/uv/<id>.exr          -> dataset_doc3d/bm/<id>.npy
  Inv3D : dataset_inv3d/<split>/uv/<id>.npy  -> dataset_inv3d/<split>/bm/<id>.npy
  UVDoc : dataset_unet/<split>/masks/<id>.npy -> dataset_unet/<split>/bm/<id>.npy
          (only fallback: build_uvdoc_unet.py writes the exact BM from grid2d; samples with
           flat_tex/ come from that build and are never overwritten here)

--sanity N        writes N ground-truth dewarps per dataset (clean warped | GT dewarp | flat).
--verify_doc3d N  compares N inverted Doc3D maps against the original Doc3D bm, streamed from
                  Hugging Face (only the needed files of bm_1.zip are read, no full download).
"""
import argparse
import io
import os
import random
import zipfile
from multiprocessing import Pool
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm

from backward_map import BM_SIZE, DOC3D_PAGE_AFFINE, dewarp, invert_forward_uv, resize_bm

IDENTITY_AFFINE = (1.0, 0.0, 1.0, 0.0)


def load_uv_doc3d(path: Path):
    # build convention: ch0 = U, ch1 = V (-1 background), ch2 = document mask
    uv = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return uv[..., 0], uv[..., 1], (uv[..., 2] > 0.5) & (uv[..., 0] >= 0)


def load_uv_inv3d(path: Path):
    # build convention: [U, V, mask]
    uv = np.load(str(path))
    return uv[..., 0], uv[..., 1], uv[..., 2] > 0.5


def load_uv_uvdoc(path: Path):
    # ch0 = U, ch1 = V, -1 marks background
    uv = np.load(str(path))
    return uv[..., 0], uv[..., 1], uv[..., 0] >= 0


def _invert_job(job):
    uv_path, bm_path, loader, page_affine = job
    try:
        u, v, mask = loader(uv_path)
        bm = invert_forward_uv(u, v, mask, BM_SIZE, page_affine=page_affine)
    except Exception:
        bm = None
    if bm is None:
        return False
    np.save(str(bm_path), bm)
    return True


def collect_jobs(name, uv_dir: Path, bm_dir: Path, pattern: str, loader, page_affine, overwrite: bool,
                 protected=lambda stem: False):
    bm_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for uv_path in sorted(uv_dir.glob(pattern)):
        bm_path = bm_dir / f"{uv_path.stem}.npy"
        if not bm_path.exists() or (overwrite and not protected(uv_path.stem)):
            jobs.append((uv_path, bm_path, loader, page_affine))
    print(f"{name}: {len(jobs)} backward maps to compute in {bm_dir}")
    return jobs


def run_jobs(jobs, workers: int):
    if not jobs:
        return
    with Pool(workers) as pool:
        ok = sum(tqdm(pool.imap_unordered(_invert_job, jobs, chunksize=16), total=len(jobs), desc="Inverting"))
    print(f"  {ok}/{len(jobs)} succeeded")


def write_sanity(name, bm_dir: Path, warped_dir: Path, flat_dir: Path, out_dir: Path, n: int):
    """GT dewarp of the clean warped image must look like the flat image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bm_paths = sorted(bm_dir.glob("*.npy"))
    rng = np.random.default_rng(0)
    for bm_path in rng.choice(bm_paths, min(n, len(bm_paths)), replace=False) if bm_paths else []:
        stem = bm_path.stem
        if not (warped_dir / f"{stem}.png").exists() or not (flat_dir / f"{stem}.png").exists():
            continue
        warped = cv2.imread(str(warped_dir / f"{stem}.png"))
        flat = cv2.imread(str(flat_dir / f"{stem}.png"))
        if warped is None or flat is None:
            continue
        bm = torch.from_numpy(np.load(str(bm_path))).unsqueeze(0)
        img = torch.from_numpy(warped).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        rec = dewarp(img, bm)[0].permute(1, 2, 0).numpy()
        rec = (rec * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{name}_{stem}.png"), np.hstack([warped, rec, flat]))


def _load_doc3d_original_bm(data: bytes) -> np.ndarray:
    """Original Doc3D bm .mat (pixel coords of the 448px image) -> (2, h, w) in [-1, 1]."""
    import h5py
    import scipy.io
    try:
        mat = scipy.io.loadmat(io.BytesIO(data))
        arr = mat[[k for k in mat if not k.startswith("__")][0]]
        arr = np.transpose(arr, (1, 2, 0)) if arr.shape[0] == 2 else arr
    except NotImplementedError:
        with h5py.File(io.BytesIO(data), "r") as f:
            arr = np.array(f[list(f.keys())[0]])
        arr = np.transpose(arr, (2, 1, 0)) if arr.shape[0] == 2 else arr
    h, w = arr.shape[:2]
    return np.stack([arr[..., 0] / (w - 1) * 2 - 1, arr[..., 1] / (h - 1) * 2 - 1], 0).astype(np.float32)


def verify_doc3d(doc3d: Path, n: int, repo: str = "StonyBrook-CVLab/doc3D-dataset"):
    from huggingface_hub import HfFileSystem
    f = HfFileSystem().open(f"datasets/{repo}/doc3d/bm_1.zip", "rb", block_size=4 * 2**20)
    zf = zipfile.ZipFile(f)
    ours = {p.stem for p in (doc3d / "bm").glob("*.npy")}
    members = [m for m in zf.infolist() if m.filename.endswith(".mat") and Path(m.filename).stem in ours]
    random.seed(0)
    members = random.sample(members, min(n, len(members)))

    errs, rec_orig, rec_ours = [], [], []
    for m in tqdm(members, desc="Verifying Doc3D"):
        stem = Path(m.filename).stem
        orig = resize_bm(_load_doc3d_original_bm(zf.read(m)))
        inv = np.load(str(doc3d / "bm" / f"{stem}.npy"))
        errs.append(np.linalg.norm((orig - inv) * 127.5, axis=0).mean())
        alb = cv2.imread(str(doc3d / "alb" / f"{stem}.png"))
        flat = cv2.imread(str(doc3d / "flat_albedo" / f"{stem}.png"))
        if alb is None or flat is None:
            continue
        img = torch.from_numpy(alb).permute(2, 0, 1).float()[None] / 255
        for grid, acc in [(orig, rec_orig), (inv, rec_ours)]:
            r = dewarp(img, torch.from_numpy(grid)[None])[0].permute(1, 2, 0).numpy()
            acc.append(np.abs(r - flat / 255).mean())
    e = np.array(errs)
    print(f"Inverted vs original Doc3D bm (px @256): mean {e.mean():.3f} | median {np.median(e):.3f} | "
          f"p95 {np.percentile(e, 95):.3f} | max {e.max():.3f}")
    print(f"Recon L1 vs flat_albedo: original {np.mean(rec_orig):.4f} | inverted {np.mean(rec_ours):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvdoc_dir", default="dataset_unet")
    parser.add_argument("--doc3d_dir", default="dataset_doc3d")
    parser.add_argument("--inv3d_dir", default="dataset_inv3d")
    parser.add_argument("--datasets", nargs="+", default=["doc3d", "inv3d", "uvdoc"], choices=["doc3d", "inv3d", "uvdoc"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing maps (exact UVDoc maps are kept)")
    parser.add_argument("--sanity", type=int, default=8, help="GT dewarp samples per dataset (0 = off)")
    parser.add_argument("--sanity_dir", default="bm_sanity")
    parser.add_argument("--verify_doc3d", type=int, default=0, help="Compare N Doc3D maps against the original bm")
    args = parser.parse_args()

    doc3d, inv3d, uvdoc = Path(args.doc3d_dir), Path(args.inv3d_dir), Path(args.uvdoc_dir)
    sanity = Path(args.sanity_dir)

    jobs = []
    if "doc3d" in args.datasets and (doc3d / "uv").exists():
        jobs += collect_jobs("Doc3D", doc3d / "uv", doc3d / "bm", "*.exr", load_uv_doc3d,
                             DOC3D_PAGE_AFFINE, args.overwrite)
    for split in ["train", "val", "test"]:
        if "inv3d" in args.datasets and (inv3d / split / "uv").exists():
            jobs += collect_jobs(f"Inv3D/{split}", inv3d / split / "uv", inv3d / split / "bm", "*.npy",
                                 load_uv_inv3d, IDENTITY_AFFINE, args.overwrite)
        if "uvdoc" in args.datasets and (uvdoc / split / "masks").exists():
            flat_dir = uvdoc / split / "flat_tex"
            jobs += collect_jobs(f"UVDoc/{split}", uvdoc / split / "masks", uvdoc / split / "bm", "*.npy",
                                 load_uv_uvdoc, IDENTITY_AFFINE, args.overwrite,
                                 protected=lambda stem, d=flat_dir: (d / f"{stem}.png").exists())
    run_jobs(jobs, args.workers)

    if args.sanity > 0:
        write_sanity("doc3d", doc3d / "bm", doc3d / "alb", doc3d / "flat_albedo", sanity, args.sanity)
        for split in ["train", "val", "test"]:
            write_sanity(f"inv3d_{split}", inv3d / split / "bm", inv3d / split / "warped_alb",
                         inv3d / split / "flat_alb", sanity, args.sanity)
            write_sanity(f"uvdoc_{split}", uvdoc / split / "bm", uvdoc / split / "warped_tex",
                         uvdoc / split / "flat_tex", sanity, args.sanity)
        print(f"Sanity images (warped | GT dewarp | flat) in {sanity}/")

    if args.verify_doc3d > 0:
        verify_doc3d(doc3d, args.verify_doc3d)


if __name__ == "__main__":
    main()

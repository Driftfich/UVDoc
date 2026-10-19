"""
Backward-map training datasets (UVDoc, Doc3D, Inv3D).

Every sample is a dict:
  img          (3, H, W)  RGB input in [0, 1]
  bm           (2, GH, GW) backward map (see backward_map.py)
  flat         (3, H, W)  flat page (albedo / texture)
  warped       (3, H, W)  clean (shading-free) warped page, source for the reconstruction loss
  flat_edges   (1, H, W)  edge weights of the flat page
  warped_edges (1, H, W)  edge weights of the warped page
  has_flat     ()         1 if flat and warped exist, else these are zeros and must be ignored
  has_edges    ()         same for the edge maps
"""
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import orjson
import torch
from torch.utils.data import Dataset

from backward_map import BM_SIZE, resize_bm


def _read_rgb(path: Path, size) -> Optional[np.ndarray]:
    if not path.exists():  # optional targets (e.g. UVDoc textures) are missing by design; avoid OpenCV warnings
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != tuple(size):
        img = cv2.resize(img, (size[1], size[0]))
    return img


def _read_gray(path: Path, size) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape[:2] != tuple(size):
        img = cv2.resize(img, (size[1], size[0]))
    return img.astype(np.float32) / 255.0


def _chw(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class BackwardMapDataset(Dataset):
    """
    Subclasses provide the sample ids and a mapping id -> file paths with the keys
    img, bm, flat, warped, flat_edges, warped_edges. Samples without a bm file are skipped.
    """
    def __init__(self, ids: List[str], size=(256, 256), color_transforms=None):
        self.size = size
        self.color_transforms = color_transforms
        self.ids = [i for i in ids if self.paths(i)["bm"].exists()]
        if len(self.ids) < len(ids):
            print(f"{type(self).__name__}: {len(ids) - len(self.ids)} of {len(ids)} samples skipped (no bm)")

    def paths(self, img_id: str) -> Dict[str, Path]:
        raise NotImplementedError

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        p = self.paths(self.ids[idx])
        h, w = self.size

        img = _read_rgb(p["img"], self.size)
        if img is None:
            img = np.zeros((h, w, 3), np.uint8)
        if self.color_transforms is not None:
            img = self.color_transforms(image=img)["image"]

        bm = resize_bm(np.load(str(p["bm"])), BM_SIZE)

        flat, warped = _read_rgb(p["flat"], self.size), _read_rgb(p["warped"], self.size)
        has_flat = flat is not None and warped is not None
        if not has_flat:
            flat = warped = np.zeros((h, w, 3), np.uint8)

        flat_edges, warped_edges = _read_gray(p["flat_edges"], self.size), _read_gray(p["warped_edges"], self.size)
        has_edges = flat_edges is not None and warped_edges is not None
        if not has_edges:
            flat_edges = warped_edges = np.zeros((h, w), np.float32)

        return {
            "img": _chw(img.astype(np.float32) / 255.0),
            "bm": torch.from_numpy(bm),
            "flat": _chw(flat.astype(np.float32) / 255.0),
            "warped": _chw(warped.astype(np.float32) / 255.0),
            "flat_edges": torch.from_numpy(flat_edges)[None],
            "warped_edges": torch.from_numpy(warped_edges)[None],
            "has_flat": torch.tensor(float(has_flat)),
            "has_edges": torch.tensor(float(has_edges)),
        }


class UVDocDataset(BackwardMapDataset):
    """dataset_unet/<split>/{images, bm, flat_tex, warped_tex, flat_edges, warped_edges}"""
    def __init__(self, root_dir, split='train', **kwargs):
        self.root = Path(root_dir) / split
        super().__init__(sorted(f.stem for f in (self.root / "images").glob("*.jpg")), **kwargs)

    def paths(self, i):
        r = self.root
        return {
            "img": r / "images" / f"{i}.jpg",
            "bm": r / "bm" / f"{i}.npy",
            "flat": r / "flat_tex" / f"{i}.png",
            "warped": r / "warped_tex" / f"{i}.png",
            "flat_edges": r / "flat_edges" / f"{i}.png",
            "warped_edges": r / "warped_edges" / f"{i}.png",
        }


class Doc3DDataset(BackwardMapDataset):
    """dataset_doc3d/{img, bm, flat_albedo, alb, flat_edges, warped_edges} + split.json"""
    def __init__(self, root_dir, split='train', **kwargs):
        self.root = Path(root_dir)
        split_file = self.root / "split.json"
        if split_file.exists():
            ids = orjson.loads(split_file.read_bytes()).get(split, [])
        else:
            print(f"Warning: Split file not found at {split_file}. Loading all images.")
            ids = sorted(f.stem for f in (self.root / "img").glob("*.png"))
        super().__init__(ids, **kwargs)

    def paths(self, i):
        r = self.root
        return {
            "img": r / "img" / f"{i}.png",
            "bm": r / "bm" / f"{i}.npy",
            "flat": r / "flat_albedo" / f"{i}.png",
            "warped": r / "alb" / f"{i}.png",
            "flat_edges": r / "flat_edges" / f"{i}.png",
            "warped_edges": r / "warped_edges" / f"{i}.png",
        }


class Inv3DDataset(BackwardMapDataset):
    """dataset_inv3d/<split>/{img, bm, flat_alb, warped_alb, flat_edges, warped_edges}"""
    def __init__(self, root_dir, split='train', **kwargs):
        self.root = Path(root_dir) / split
        super().__init__(sorted(f.stem for f in (self.root / "img").glob("*.png")), **kwargs)

    def paths(self, i):
        r = self.root
        return {
            "img": r / "img" / f"{i}.png",
            "bm": r / "bm" / f"{i}.npy",
            "flat": r / "flat_alb" / f"{i}.png",
            "warped": r / "warped_alb" / f"{i}.png",
            "flat_edges": r / "flat_edges" / f"{i}.png",
            "warped_edges": r / "warped_edges" / f"{i}.png",
        }

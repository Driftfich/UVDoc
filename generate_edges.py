import argparse
import json
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import scipy.io
import h5py
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F

# Configuration
TARGET_SIZE = (256, 256)

def load_mat_file(path):
    """Robustly loads a .mat file."""
    try:
        mat = scipy.io.loadmat(str(path))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys: return None, None
        key = keys[0]
        if 'grid2d' in keys: key = 'grid2d'
        elif 'uv' in keys: key = 'uv'
        return mat[key], 'scipy'
    except:
        try:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                key = keys[0]
                if 'grid2d' in keys: key = 'grid2d'
                elif 'uv' in keys: key = 'uv'
                return np.array(f[key]), 'h5py'
        except:
            return None, None

def compute_edge_map_from_image(img, size=TARGET_SIZE):
    """
    Computes distance-weighted edge map from a numpy image.
    img: Numpy array (H, W) or (H, W, 3)
    """
    if img is None: return None
    
    # Resize first
    img_resized = cv2.resize(img, (size[1], size[0]))
    
    # Convert to Gray if needed
    if len(img_resized.shape) == 3:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_resized

    # Text/Edge Detection pipeline
    # 1. Blur slightly to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. Thresholding (Otsu) to separate ink from paper
    # Invert so ink is bright
    gray_inv = cv2.bitwise_not(blurred)
    _, bin_img = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphological closing to connect characters into lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) # Horizontal text
    lines_h = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_h)
    
    # 4. Canny Edge Detection
    edges = cv2.Canny(lines_h, 50, 150)
    
    # 5. Distance Transform
    # We want a smooth gradient falling off from the edges
    dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
    weight_map = np.exp(-0.2 * dist).astype(np.float32)
    
    return weight_map

def generate_uvdoc_edges(base_dir: Path):
    """base_dir: one split of the built UVDoc dataset (e.g. dataset_unet/train)."""
    print(f"\n--- Generating UVDoc Edges in {base_dir} ---")
    src_flat_dir = base_dir / "flat_tex"
    src_uv_dir = base_dir / "masks"

    dest_flat_dir = base_dir / "flat_edges"
    dest_warped_dir = base_dir / "warped_edges"

    dest_flat_dir.mkdir(exist_ok=True, parents=True)
    dest_warped_dir.mkdir(exist_ok=True, parents=True)

    if not src_flat_dir.exists():
        print(f"UVDoc 'flat_tex' directory not found in {base_dir}. Rebuild with build_uvdoc_unet.py. Skipping.")
        return

    for flat_path in tqdm(list(src_flat_dir.glob("*.png")), desc="UVDoc"):
        img_id = flat_path.stem
        dest_flat_path = dest_flat_dir / f"{img_id}.png"
        dest_warped_path = dest_warped_dir / f"{img_id}.png"

        if dest_flat_path.exists() and dest_warped_path.exists():
            continue

        flat_map = compute_edge_map_from_image(cv2.imread(str(flat_path)))
        if flat_map is None: continue
        cv2.imwrite(str(dest_flat_path), (flat_map * 255).astype(np.uint8))

        # Forward UV (ch0, ch1), -1 marks background
        uv_path = src_uv_dir / f"{img_id}.npy"
        if not uv_path.exists(): continue
        uv = np.load(str(uv_path))[..., :2]

        h_flat, w_flat = flat_map.shape
        doc = uv[..., 0] >= 0
        map_x = np.where(doc, uv[..., 0] * (w_flat - 1), -1.0).astype(np.float32)
        map_y = np.where(doc, uv[..., 1] * (h_flat - 1), -1.0).astype(np.float32)

        warped_map = cv2.remap(flat_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(dest_warped_path), (warped_map * 255).astype(np.uint8))

def generate_doc3d_edges(base_dir: Path):
    print(f"\n--- Generating Doc3D Edges in {base_dir} ---")
    
    src_flat_dir = base_dir / "flat_albedo"
    src_uv_dir = base_dir / "uv"
    
    dest_flat_dir = base_dir / "flat_edges"
    dest_warped_dir = base_dir / "warped_edges"
    
    dest_flat_dir.mkdir(exist_ok=True, parents=True)
    dest_warped_dir.mkdir(exist_ok=True, parents=True)
    
    if not src_flat_dir.exists():
        print(f"Doc3D 'flat_albedo' directory not found in {base_dir}. Skipping.")
        return
        
    flat_files = list(src_flat_dir.glob("*.png"))
    print(f"Found {len(flat_files)} Doc3D samples.")
    
    # Enable EXR for OpenCV
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    
    for flat_path in tqdm(flat_files, desc="Doc3D"):
        img_id = flat_path.stem
        dest_flat_path = dest_flat_dir / f"{img_id}.png"
        dest_warped_path = dest_warped_dir / f"{img_id}.png"
        
        # Optimization
        if dest_flat_path.exists() and dest_warped_path.exists():
            continue
            
        # 1. Flat Edges
        flat_alb = cv2.imread(str(flat_path))
        if flat_alb is None: continue
        
        flat_map = compute_edge_map_from_image(flat_alb)
        if flat_map is None: continue
        
        cv2.imwrite(str(dest_flat_path), (flat_map * 255).astype(np.uint8))
        
        # 2. Warped Edges
        # Load UV map (build convention: ch0=U, ch1=V in [0,1], ch2=doc mask)
        # Background pixels use -1 sentinel in ch0/ch1.
        uv_path = src_uv_dir / f"{img_id}.exr"
        if not uv_path.exists(): continue

        uv = cv2.imread(str(uv_path), cv2.IMREAD_UNCHANGED)
        if uv is None: continue

        if uv.shape[:2] != TARGET_SIZE:
            uv = cv2.resize(uv, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)

        h_flat, w_flat = flat_map.shape

        # Use ch2 mask to send background pixels out-of-bounds (→ borderValue=0)
        if uv.shape[2] >= 3:
            doc = uv[..., 2] > 0.5
        else:
            doc = uv[..., 0] >= 0  # fallback: -1 sentinel

        map_x = np.where(doc, uv[..., 0] * (w_flat - 1), -1.0).astype(np.float32)
        map_y = np.where(doc, uv[..., 1] * (h_flat - 1), -1.0).astype(np.float32)

        warped_map = cv2.remap(flat_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(dest_warped_path), (warped_map * 255).astype(np.uint8))

def generate_inv3d_edges(base_dir: Path):
    print(f"\n--- Generating Inv3D Edges in {base_dir} ---")
    src_flat_dir = base_dir / "flat_alb"
    src_warped_dir = base_dir / "warped_alb"
    src_img_dir = base_dir / "img"
    src_uv_dir = base_dir / "uv"
    
    dest_flat_dir = base_dir / "flat_edges"
    dest_warped_dir = base_dir / "warped_edges"

    dest_flat_dir.mkdir(exist_ok=True, parents=True)
    dest_warped_dir.mkdir(exist_ok=True, parents=True)

    if not src_flat_dir.exists():
        print(f"Inv3D 'flat_alb' directory not found in {base_dir}. Skipping.")
        return
    
    flat_files = list(src_flat_dir.glob("*.png"))
    print(f"Found {len(flat_files)} Inv3D samples.")

    # Enable EXR for OpenCV
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    for flat_path in tqdm(flat_files, desc="Inv3D"):
        img_id = flat_path.stem
        dest_flat_path = dest_flat_dir / f"{img_id}.png"
        dest_warped_path = dest_warped_dir / f"{img_id}.png"

        # Optimization
        if dest_flat_path.exists() and dest_warped_path.exists():
            continue

        # 1. Flat Edges
        flat_alb = cv2.imread(str(flat_path))
        if flat_alb is None: continue

        flat_map = compute_edge_map_from_image(flat_alb)
        if flat_map is None: continue

        cv2.imwrite(str(dest_flat_path), (flat_map * 255).astype(np.uint8))

        # load uv map
        uv_path = src_uv_dir / f"{img_id}.npy"
        if not uv_path.exists(): continue

        uv = np.load(str(uv_path))
        if uv is None: continue

        if uv.shape[:2] != TARGET_SIZE:
            uv = cv2.resize(uv, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)

        h_flat, w_flat = flat_map.shape

        u, v, mask = uv[..., 0], uv[..., 1], uv[..., 2]

        # use mask to send background pixels out-of-bounds (→ borderValue=0)
        map_x = np.where(mask > 0.5, u * (w_flat - 1), -1.0).astype(np.float32)
        map_y = np.where(mask > 0.5, v * (h_flat - 1), -1.0).astype(np.float32)

        warped_map = cv2.remap(flat_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(dest_warped_path), (warped_map * 255).astype(np.uint8))

    print("Done generating Inv3D edges.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uvdoc_dir", default="dataset_unet", help="Built UVDoc dataset (with train/val splits)")
    args, _ = parser.parse_known_args()

    uvdoc_path = Path(args.uvdoc_dir)
    for split in ["train", "val"]:
        if (uvdoc_path / split).exists():
            generate_uvdoc_edges(uvdoc_path / split)

    doc3d_dataset_path = Path("dataset_doc3d")
    if (doc3d_dataset_path / "flat_albedo").exists():
        generate_doc3d_edges(doc3d_dataset_path)
    else:
        print("Doc3D flat_albedo not found in dataset_doc3d. Please prepare flat albedos first.")

    inv3d_path = Path("dataset_inv3d")
    if inv3d_path.exists():
        for split in ["train", "val", "test"]:
            split_path = inv3d_path / split
            if split_path.exists():
                generate_inv3d_edges(split_path)

if __name__ == "__main__":
    main()

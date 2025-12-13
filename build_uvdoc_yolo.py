import os
import sys
import zipfile
import requests
import numpy as np
import scipy.io
import cv2
import yaml
import argparse
import json
import shutil
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import List
from scipy.ndimage import map_coordinates

# --- Configuration ---
# REPLACE THIS WITH THE ACTUAL URL IF KNOWN, OR DOWNLOAD MANUALLY
DATASET_URL = "https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip" 
DOWNLOAD_DIR = Path("downloads")
DATASET_DIR = Path("dataset_yolo")
RAW_DATA_DIR = Path("UVDoc_final") # The directory inside the zip
GRID_SIZE = 10 # N x N grid
IMG_SIZE = (640, 640) # YOLO training image size

# Ensure directories exist
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, destination_path):
    """
    Downloads a file with resume capability.
    """
    if destination_path.exists():
        current_size = destination_path.stat().st_size
    else:
        current_size = 0

    headers = {'Range': f'bytes={current_size}-'}
    
    print(f"Downloading {url} to {destination_path}...")
    try:
        response = requests.get(url, headers=headers, stream=True)
        
        mode = 'ab' if response.status_code == 206 else 'wb'
        if mode == 'wb':
            print("Server does not support resume or file is new. Downloading from start.")
            current_size = 0
            
        total_size = int(response.headers.get('content-length', 0)) + current_size
        
        block_size = 1024 * 1024 # 1MB
        
        with open(destination_path, mode) as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, initial=current_size, desc=destination_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Please download the 'UVDoc_final.zip' manually and place it in the 'downloads' folder.")

def unzip_file(zip_path, extract_to, skip_dirs: List[str] = ["__MACOSX/"]):
    """
    Unzips the archive.
    """
    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        for member in tqdm(zip_ref.infolist(), total=total_files, desc="Extracting"):
            name = member.filename.lstrip("./")
            if any(name.startswith(dir) for dir in skip_dirs):
                continue
            zip_ref.extract(member, extract_to)

def load_mat_file(path):
    """
    Robustly loads a .mat file, handling both v7 (scipy) and v7.3 (h5py).
    Returns the data array found.
    """
    try:
        mat = scipy.io.loadmat(str(path))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys: return None
        
        # Heuristic for key
        key = keys[0]
        if 'grid2d' in keys: key = 'grid2d'
        elif 'uv' in keys: key = 'uv'
        
        data = mat[key]
        return data, 'scipy'
        
    except NotImplementedError:
        # v7.3 HDF5
        try:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                if not keys: return None
                
                key = keys[0]
                if 'grid2d' in keys: key = 'grid2d'
                elif 'uv' in keys: key = 'uv'
                
                # h5py reads as (C, W, H) or similar because of column-major
                data = np.array(f[key])
                return data, 'h5py'
        except Exception:
            return None
    except Exception:
        return None

def process_dataset(raw_dir: Path, output_dir: Path, grid_n: int):
    """
    Converts UVDoc data to YOLO-Pose format.
    """
    
    # Define subdirectories
    img_dir_train = output_dir / "images" / "train"
    img_dir_val = output_dir / "images" / "val"
    lbl_dir_train = output_dir / "labels" / "train"
    lbl_dir_val = output_dir / "labels" / "val"
    
    for d in [img_dir_train, img_dir_val, lbl_dir_train, lbl_dir_val]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Load split.json if available
    split_file = raw_dir / "split.json"
    train_ids = None
    val_ids = None
    
    if split_file.exists():
        try:
            with open(split_file, 'r') as f:
                splits = yaml.safe_load(f)
                if 'train' in splits or 'val' in splits:
                    train_ids = set(splits.get('train', []))
                    val_ids = set(splits.get('val', []))
                else:
                    print("split.json found but missing 'train'/'val' keys. Using random split.")
        except Exception:
            print("Error reading split.json. Using random split.")
    else:
        print("split.json not found. Using random split (80/20).")

    # Directories
    src_img_dir = raw_dir / "img"
    src_meta_dir = raw_dir / "metadata_sample"
    src_grid_dir = raw_dir / "grid2d"
        
    if not src_img_dir.exists():
        print(f"Error: Image directory {src_img_dir} not found.")
        return

    images = list(src_img_dir.glob("*.png")) + list(src_img_dir.glob("*.jpg"))
    
    print(f"Found {len(images)} images. Processing...")
    
    for img_path in tqdm(images, desc="Processing Images"):
        img_id = img_path.stem
        
        # 1. Load Metadata to find Geometry file
        meta_path = src_meta_dir / f"{img_id}.json"
        if not meta_path.exists():
            continue
            
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            geom_name = meta.get("geom_name")
            if not geom_name:
                continue
        except Exception:
            continue
            
        # 2. Determine split
        is_train = True
        if train_ids is not None:
            if img_id in val_ids:
                is_train = False
            elif img_id not in train_ids:
                pass
        else:
            if hash(img_id) % 5 == 0: 
                is_train = False
                
        dest_img_dir = img_dir_train if is_train else img_dir_val
        dest_lbl_dir = lbl_dir_train if is_train else lbl_dir_val
        
        # 3. Read Image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # 4. Read Ground Truth Grid
        mat_path = src_grid_dir / f"{geom_name}.mat"
        if not mat_path.exists():
            continue
            
        result = load_mat_file(mat_path)
        if result is None:
            continue
            
        map_data, source = result
        
        # Normalize shape to (H, W, 2)
        # HDF5 often reads as (2, W, H) or (2, H, W)
        # We need to ensure last dim is 2 (x, y)
        
        if source == 'h5py':
            # Typical HDF5 from MATLAB: (2, W, H) -> need transpose to (H, W, 2)
            # Or (2, H, W) -> (W, H, 2)
            # Let's assume (2, X_dim, Y_dim) -> transpose(2, 1, 0)
            if map_data.shape[0] == 2:
                map_data = np.transpose(map_data, (2, 1, 0))
            # If shape is (W, H, 2), it might be correct or transposed.
            # Usually HDF5 reverses dims. 
            
        elif source == 'scipy':
            if map_data.ndim == 3 and map_data.shape[0] == 2:
                map_data = np.transpose(map_data, (1, 2, 0))
        
        map_h, map_w = map_data.shape[:2]
        
        # Sampling coordinates
        xs = np.linspace(0, map_w - 1, grid_n)
        ys = np.linspace(0, map_h - 1, grid_n)
        
        grid_x_indices, grid_y_indices = np.meshgrid(xs, ys)
        
        # Flatten for mapping (2, N*N) -> [[y0, y1...], [x0, x1...]]
        sample_coords = np.vstack((grid_y_indices.flatten(), grid_x_indices.flatten()))
        
        try:
            # Interpolate X and Y coordinates
            interp_x = map_coordinates(map_data[:,:,0], sample_coords, order=1, mode='nearest')
            interp_y = map_coordinates(map_data[:,:,1], sample_coords, order=1, mode='nearest')
            
            norm_x = interp_x / w
            norm_y = interp_y / h
            
            keypoints = []
            for i in range(len(norm_x)):
                px = norm_x[i]
                py = norm_y[i]
                px = np.clip(px, 0, 1)
                py = np.clip(py, 0, 1)
                keypoints.extend([px, py, 2.0])
            
            # 5. Create Label
            min_x = max(0, min(1, np.min(norm_x)))
            max_x = max(0, min(1, np.max(norm_x)))
            min_y = max(0, min(1, np.min(norm_y)))
            max_y = max(0, min(1, np.max(norm_y)))
            
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            bbox_cx = min_x + bbox_w / 2
            bbox_cy = min_y + bbox_h / 2
            
            line = [0, bbox_cx, bbox_cy, bbox_w, bbox_h] + keypoints
            line_str = " ".join(map(str, line))
            
            with open(dest_lbl_dir / f"{img_id}.txt", 'w') as f:
                f.write(line_str)
                
            # 6. Copy Image
            shutil.copy2(img_path, dest_img_dir / f"{img_id}.png")
            
        except Exception:
            continue

def generate_yaml(output_dir, grid_n):
    """
    Creates the dataset.yaml for YOLO.
    """
    # Calculate flip_idx
    # Horizontal flip: (i, j) -> (i, N-1-j)
    flip_idx = []
    kpt_names = []
    for i in range(grid_n):
        for j in range(grid_n):
            # current_idx = i * grid_n + j
            flipped_j = grid_n - 1 - j
            flipped_idx = i * grid_n + flipped_j
            flip_idx.append(flipped_idx)
            kpt_names.append(f"pt_{i}_{j}")

    # Manual YAML construction to match specific comments and structure
    
    yaml_content = f"""# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# UVDoc-pose dataset
# Generated by build_uvdoc_yolo.py

# Train/val/test sets
path: {output_dir.absolute()} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
test: # test images (optional)

# Keypoints
kpt_shape: [{grid_n * grid_n}, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: {flip_idx}

# Classes
names:
  0: document

# Keypoint names per class
kpt_names:
  0: {kpt_names}
"""
    
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"Generated dataset.yaml at {output_dir / 'dataset.yaml'}")

def main():
    parser = argparse.ArgumentParser(description="Build UVDoc YOLO Dataset")
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE, help="Size of the keypoint grid (N x N)")
    args = parser.parse_args()
    
    zip_path = DOWNLOAD_DIR / "UVDoc_final.zip"
    
    # 1. Download
    if not zip_path.exists():
        download_file(DATASET_URL, zip_path)
    
    if not zip_path.exists():
        print("Dataset zip not found. Exiting.")
        return

    # 2. Extract
    extract_path = DOWNLOAD_DIR / "extracted"
    if not extract_path.exists():
        unzip_file(zip_path, extract_path)
    
    # Find the data root inside extracted folder
    data_root = extract_path / "UVDoc_final"
    if not data_root.exists():
        # Fallback
        if (extract_path / "img").exists():
            data_root = extract_path
        else:
            found = list(extract_path.rglob("img"))
            if found:
                data_root = found[0].parent
            else:
                print("Could not find data structure in zip.")
                return

    # 3. Process
    process_dataset(data_root, DATASET_DIR, args.grid_size)
    
    # 4. Config
    generate_yaml(DATASET_DIR, args.grid_size)
    
    print("Done! Dataset is ready.")

if __name__ == "__main__":
    main()

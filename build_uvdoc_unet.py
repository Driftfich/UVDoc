import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import cv2
import scipy.io
import h5py
from tqdm import tqdm
from scipy.interpolate import griddata
import requests
import zipfile
from typing import List
import orjson

# --- Configuration ---
# 256x256 is standard for mobile UNets (fast). 
# You can go higher (512) if needed, but start small.
TARGET_SIZE = (256, 256) 

DATASET_URL = "https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip" 
DOWNLOAD_DIR = Path("downloads")
DATASET_DIR = Path("dataset_unet")

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
    """Robustly loads a .mat file."""
    try:
        mat = scipy.io.loadmat(str(path))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys: return None
        
        key = keys[0]
        if 'grid2d' in keys: key = 'grid2d'
        elif 'uv' in keys: key = 'uv'
        
        data = mat[key]
        return data, 'scipy'
    except NotImplementedError:
        try:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                key = keys[0]
                if 'grid2d' in keys: key = 'grid2d'
                elif 'uv' in keys: key = 'uv'
                data = np.array(f[key])
                return data, 'h5py'
        except Exception:
            return None
    except Exception:
        return None

def compute_texture_edge_map(tex_path, target_size):
    """
    Computes a structural edge map from the raw texture.
    Targeting text lines and document borders.
    """
    try:
        if not tex_path.exists():
            return np.zeros(target_size, dtype=np.float32)
            
        tex = cv2.imread(str(tex_path), cv2.IMREAD_GRAYSCALE)
        if tex is None:
            return np.zeros(target_size, dtype=np.float32)
            
        # Resize to target size
        tex_resized = cv2.resize(tex, (target_size[1], target_size[0]))
        
        # Invert so text is bright
        tex_inv = cv2.bitwise_not(tex_resized)
        # Threshold to get binary text/content
        _, bin_tex = cv2.threshold(tex_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # A. Detect Text Lines (Horizontal Structure)
        # Kernel width=20, height=1 -> Fuse characters into lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        lines_h = cv2.morphologyEx(bin_tex, cv2.MORPH_CLOSE, kernel_h)
        edges_h = cv2.Canny(lines_h, 50, 150)
        
        # B. Detect Vertical Separators/Borders
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        lines_v = cv2.morphologyEx(bin_tex, cv2.MORPH_CLOSE, kernel_v)
        edges_v = cv2.Canny(lines_v, 50, 150)
        
        # C. Standard Canny for strong image borders (non-text)
        edges_img = cv2.Canny(tex_resized, 100, 200)
        
        # Combine
        combined_edges = cv2.bitwise_or(edges_h, edges_v)
        combined_edges = cv2.bitwise_or(combined_edges, edges_img)
        
        # Distance Transform
        dist = cv2.distanceTransform(cv2.bitwise_not(combined_edges), cv2.DIST_L2, 5)
        # Invert distance: 1.0 at edge, falling off
        weight_map = np.exp(-0.2 * dist)
        
        return weight_map.astype(np.float32)
    except Exception:
        return np.zeros(target_size, dtype=np.float32)

def process_unet_dataset(raw_dir: Path, output_dir: Path):
    """
    Generates (Image, UV_Map) pairs for UNet training.
    UV Map: (H, W, 2) array where ch0=u, ch1=v (normalized 0-1).
    """
    
    img_dir_train = output_dir / "train" / "images"
    mask_dir_train = output_dir / "train" / "masks" # UV maps
    img_dir_val = output_dir / "val" / "images"
    mask_dir_val = output_dir / "val" / "masks"
    
    for d in [img_dir_train, mask_dir_train, img_dir_val, mask_dir_val]:
        d.mkdir(parents=True, exist_ok=True)

    src_img_dir = raw_dir / "img"
    src_meta_dir = raw_dir / "metadata_sample"
    src_img_geom_dir = raw_dir / "img_geom"
    src_grid_dir = raw_dir / "grid2d"
    src_tex_dir = raw_dir / "textures"
    
    if not src_img_dir.exists():
        print(f"Error: {src_img_dir} not found.")
        return

    images = list(src_img_dir.glob("*.png")) + list(src_img_dir.glob("*.jpg"))
    print(f"Found {len(images)} images.")
    
    # Simple split based on hash
    for img_path in tqdm(images, desc="Processing"):
        img_id = img_path.stem
        
        # 1. Load Metadata
        meta_path = src_meta_dir / f"{img_id}.json"
        if not meta_path.exists():
            print(f"Error: {meta_path} not found.")
            continue
        
        try:
            meta = orjson.loads(meta_path.read_text())
        except:
            print(f"Error: {meta_path} not found.")
            continue

        # load geometry image
        geom_name: str = meta.get("geom_name")
        if not geom_name:
            print(f"Error: {geom_name} not found.")
            continue

        geo_img_pth = src_img_geom_dir / f"{geom_name}.png"
        if not geo_img_pth.exists():
            print(f"Error: {geo_img_pth} not found.")
            continue

        geo_img = cv2.imread(str(geo_img_pth))
        if geo_img is None:
            print(f"Error: {geo_img_pth} not found.")
            continue

        # resize geometry image to target size
        geo_img_resized = cv2.resize(geo_img, TARGET_SIZE)

        # Filter Green Background from Geometry Image
        # Convert to HSV to detect green
        hsv = cv2.cvtColor(geo_img_resized, cv2.COLOR_BGR2HSV)
        # Green is typically around 60. Range [35, 85] covers most greens.
        # Saturation/Value threshold ensures we don't mask out white/grey paper.
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Refine mask: Dilate the green mask slightly to catch anti-aliased edges
        kernel = np.ones((3,3), np.uint8)
        green_mask_dilated = cv2.dilate(green_mask, kernel, iterations=1)
        
        # Keep non-green areas
        paper_mask = cv2.bitwise_not(green_mask_dilated)
        
        # Apply mask to geometry image
        geo_img_clean = cv2.bitwise_and(geo_img_resized, geo_img_resized, mask=paper_mask)

        # create shading image
        shading_img = geo_img_clean.astype(np.float32) / 255.0
        
        # 1b. Load/Compute Texture Edge Map
        tex_rel_path = meta.get("texture_name")
        edge_map = np.zeros(TARGET_SIZE, dtype=np.float32)
        if tex_rel_path:
            tex_name = Path(tex_rel_path).name
            tex_path = src_tex_dir / tex_name
            # If not found directly, try recursive search or just assume correct if structure is known
            if not tex_path.exists():
                # Fallback: maybe it's inside a subdir in textures/ ?
                # For now assume flat or rely on find. 
                # Given ls output earlier, it seems flat.
                pass
            
            edge_map = compute_texture_edge_map(tex_path, TARGET_SIZE)
            
        # 2. Load GT Grid
        mat_path = src_grid_dir / f"{geom_name}.mat"
        if not mat_path.exists(): continue
        
        res = load_mat_file(mat_path)
        if res is None: continue
        map_data, source = res  
        
        # Normalize shape to (H_grid, W_grid, 2)
        if source == 'h5py':
            if map_data.shape[0] == 2:
                map_data = np.transpose(map_data, (2, 1, 0)) # Often (2, W, H) -> (H, W, 2) ? 
                # Actually H5py is typically (2, W, H) -> transpose(2, 1, 0) gives (H, W, 2)
                # Let's verify dimensions later
        elif source == 'scipy':
            if map_data.ndim == 3 and map_data.shape[0] == 2:
                map_data = np.transpose(map_data, (1, 2, 0))

        # The map_data contains the (x,y) coordinates in the *original image* 
        # corresponding to the regular grid points of the flattened document.
        # BUT UNet needs the inverse: For every pixel in the *input image*, what is its (u,v) coord?
        # This is hard because the grid is sparse (e.g. 50x50) relative to image pixels.
        #
        # ALTERNATIVE: Backward mapping.
        # The .mat usually gives us: Grid[i,j] -> (x,y) in image.
        # We want: Image[x,y] -> (u,v) where u=i/W, v=j/H.
        # This requires scattered interpolation (griddata).
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        # Resize image to target size
        img_resized = cv2.resize(img, TARGET_SIZE)
        
        # Prepare sparse points for interpolation
        # map_data is (GridH, GridW, 2) containing (x,y)
        grid_h, grid_w, _ = map_data.shape
        
        # Normalize grid (x,y) to (0,1) relative to original image dimensions
        # Note: .mat coords might be 1-based or 0-based. Assuming typical CV.
        points_x = map_data[..., 0].flatten() / w_img
        points_y = map_data[..., 1].flatten() / h_img
        
        # Target values (u,v)
        # u varies 0..1 along columns, v varies 0..1 along rows
        u_grid, v_grid = np.meshgrid(np.linspace(0, 1, grid_w), np.linspace(0, 1, grid_h))
        values_u = u_grid.flatten()
        values_v = v_grid.flatten()
        
        # Valid points mask (map_data might have NaNs or zeros for background?)
        # UVDoc usually full grid.
        
        # Create dense UV map for the TARGET_SIZE image
        # We need to query at every pixel of the resized image
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, TARGET_SIZE[0]), 
            np.linspace(0, 1, TARGET_SIZE[1])
        )
        
        # Interpolate: (x,y) -> u
        # This is "Forward Mapping" (Image -> UV).
        # Since we have sparse (x,y) -> (u,v), we can use griddata (scipy).
        # Input points: (x,y) from map_data. Values: u and v.
        # Query points: regular grid of the resized image.
        
        points = np.stack([points_x, points_y], axis=1) # (N, 2)
        
        try:
            # Linear is faster than cubic but smoother than nearest
            dense_u = griddata(points, values_u, (grid_x, grid_y), method='linear', fill_value=-1)
            dense_v = griddata(points, values_v, (grid_x, grid_y), method='linear', fill_value=-1)
        except Exception as e:
            # print(f"Interpolation failed: {e}")
            continue
            
        # Stack to (H, W, 2)
        uv_map = np.stack([dense_u, dense_v], axis=-1).astype(np.float32)

        # stack shading image and uv map AND edge map
        # uv_map: (H,W,2), shading_img: (H,W,3), edge_map: (H,W) -> (H,W,6)
        combined_label = np.concatenate([
            uv_map, 
            shading_img, 
            edge_map[..., None]
        ], axis=-1).astype(np.float32)
        
        # -1 indicates background (outside document hull).
        # We can treat this as a mask for loss calculation.
        
        # Save
        is_train = (hash(img_id) % 5 != 0)
        dest_img_dir = img_dir_train if is_train else img_dir_val
        dest_mask_dir = mask_dir_train if is_train else mask_dir_val
        
        # Save Image
        cv2.imwrite(str(dest_img_dir / f"{img_id}.jpg"), img_resized)
        
        # Save UV Map as .npy (float precision is crucial for dewarping)
        np.save(str(dest_mask_dir / f"{img_id}.npy"), combined_label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=DOWNLOAD_DIR)
    parser.add_argument("--out_dir", default=DATASET_DIR)
    parser.add_argument("--url", default=DATASET_URL)

    args = parser.parse_args()
    
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    url = args.url
    
    # Ensure directories exist
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Download the dataset and unzip if UVDoc_final dir not present
    if not (raw / "UVDoc_final").exists():
        if not (raw / "UVDoc_final.zip").exists():
            print("UVDoc_final.zip not found. Downloading...")
            download_file(url, raw / "UVDoc_final.zip")
        
        unzip_file(raw / "UVDoc_final.zip", raw)
        print("UVDoc_final dir downloaded and unzipped.")
    else:
        print("UVDoc_final dir already exists. Skipping download and unzip.")

    # Process the dataset
    process_unet_dataset(raw / "UVDoc_final", out)
    print("Done.")

if __name__ == "__main__":
    main()

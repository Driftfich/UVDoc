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
import tarfile
from typing import List, Tuple
import orjson
import torch
from huggingface_hub import snapshot_download
import torch.nn.functional as F
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# --- Configuration ---
# 256x256 is standard for mobile UNets (fast). 
# You can go higher (512) if needed, but start small.
TARGET_SIZE = (256, 256) 

DATASET_URL = "https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip" 
DOWNLOAD_DIR = Path("downloads")
UVDOC_DATASET_DIR = Path("dataset_unet")
DOC3D_DATASET_DIR = Path("dataset_doc3d")
DOC3D_DIR = "StonyBrook-CVLab/doc3D-dataset"
INV3D_DATASET_DIR = Path("dataset_inv3d")
INV3D_URLS_TRAIN = [
    "https://radar.kit.edu/radar-backend/archives/JnvMMjtvIjxpUpGz/versions/1/content",
    "https://radar.kit.edu/radar-backend/archives/ihpFPDjGMMpcPcRF/versions/1/content",
    "https://radar.kit.edu/radar-backend/archives/gbiTkBlCTWttcdlv/versions/1/content",
    "https://radar.kit.edu/radar-backend/archives/loetxlbsNkFYeCjI/versions/1/content"
]
INV3D_URLS_VAL = ["https://radar.kit.edu/radar-backend/archives/GPweoqzKRRZGOAEE/versions/1/content"]
INV3D_URLS_TEST = ["https://radar.kit.edu/radar-backend/archives/lvwLUTAbBhcjdTbB/versions/1/content"]

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

def unzip_file(zip_path, extract_to, skip_dirs: List[str] = ["__MACOSX/"], delete_after: bool = True):
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
    if delete_after:
        os.remove(zip_path)
        print(f"Deleted {zip_path}")

def untar_file(tar_path, extract_to, skip_dirs: List[str] = ["__MACOSX/"], delete_after: bool = True):
    """
    Untars the archive.
    """
    print(f"Untaring {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar_ref:
        members = tar_ref.getmembers()
        for member in tqdm(members, total=len(members), desc="Extracting"):
            name = member.name.lstrip("./")
            if any(name.startswith(dir) for dir in skip_dirs):
                continue
            tar_ref.extract(member, extract_to)
    if delete_after:
        os.remove(tar_path)
        print(f"Deleted {tar_path}")

def load_mat_file(path):
    """Robustly loads a .mat file."""
    try:
        mat = scipy.io.loadmat(str(path))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys: return None, None
        
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
            return None, None
    except Exception:
        return None, None

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

def process_uvdoc_dataset(raw_dir: Path, output_dir: Path):
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
    existing_train_img_ids = np.array([img_path.stem for img_path in img_dir_train.glob("*.jpg")])
    existing_train_mask_ids = np.array([img_path.stem for img_path in mask_dir_train.glob("*.npy")])
    existing_val_img_ids = np.array([img_path.stem for img_path in img_dir_val.glob("*.jpg")])
    existing_val_mask_ids = np.array([img_path.stem for img_path in mask_dir_val.glob("*.npy")])

    # an id exists if its present in train or val img and mask
    existing_train_ids = np.intersect1d(existing_train_img_ids, existing_train_mask_ids)
    existing_val_ids = np.intersect1d(existing_val_img_ids, existing_val_mask_ids)
    existing_ids = np.concatenate([existing_train_ids, existing_val_ids])

    images = [img_path for img_path in images if img_path.stem not in existing_ids]
    print(f"Found {len(images)} new images. Already existing processed images: {len(existing_ids)}")
    
    # Simple split based on hash
    for img_path in tqdm(images, desc="Processing"):
        img_id = img_path.stem

        if Path(img_dir_train / f"{img_id}.jpg").exists() and Path(img_dir_val / f"{img_id}.jpg").exists() and Path(mask_dir_train / f"{img_id}.npy").exists() and Path(mask_dir_val / f"{img_id}.npy").exists():
            continue
        
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
        
        map_data, source = load_mat_file(mat_path)
        if map_data is None or source is None: continue

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

def reconstruct_flat_albedo_tensor(alb, bm):
    """
    Torch-based reconstruction for speed.
    alb: numpy array (H, W, 3) BGR [0-255]
    bm: numpy array (H, W, 2) [0-1] or [pixels]
    """
    # 1. Explicitly move to RGB workspace for processing
    alb_rgb = cv2.cvtColor(alb, cv2.COLOR_BGR2RGB)
    alb_t = torch.from_numpy(alb_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0 # (1, 3, H, W)
    # Sample a source-validity mask with the same grid to remove black-edge bleed
    # from bilinear interpolation across wrapped-image background.
    src_valid = (np.max(alb_rgb, axis=2) > 1).astype(np.float32)
    src_valid_t = torch.from_numpy(src_valid).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    # Convert BM to Tensor & Normalize to [-1, 1]
    # Doc3D BM is typically in range [0, 448] (pixel coords of the *input* image)
    # But F.grid_sample expects [-1, 1] referring to the source texture dimensions.
    # The BM maps Flat_Pos -> Warped_Pos.
    # So we want to sample from 'alb' at positions given by 'bm'.
    
    # Doc3D BM is often normalized 0..1 in the .mat file
    # Let's assume 0..1 based on typical Doc3D usage
    bm_norm = bm.copy()
    
    # Check if BM is in pixel coordinates or normalized
    if bm_norm.max() > 1.0: # Likely pixel coordinates
        bm_norm[..., 0] /= (alb.shape[1] - 1)
        bm_norm[..., 1] /= (alb.shape[0] - 1)

    bm_norm = bm_norm * 2.0 - 1.0 # (H, W, 2)
    grid = torch.from_numpy(bm_norm).unsqueeze(0).float() # (1, H, W, 2)
    
    # Grid Sample
    # Note: grid_sample(input, grid)
    # input = Wrapped Albedo
    # grid = Backward Map (Flat -> Warped Coords)
    flat_bilinear = F.grid_sample(alb_t, grid, align_corners=True, mode='bilinear', padding_mode='border')
    flat_nearest = F.grid_sample(alb_t, grid, align_corners=True, mode='nearest', padding_mode='border')
    sampled_valid = F.grid_sample(src_valid_t, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
    sampled_valid = sampled_valid.clamp(min=1e-4)
    flat_unpremul = (flat_bilinear / sampled_valid).clamp(0.0, 1.0)

    # Boundary polish:
    # - strong support -> use unpremultiplied bilinear (smooth and accurate)
    # - weak support (edge/intersection with background) -> fall back to nearest
    #   to avoid black interpolation streaks.
    low, high = 0.20, 0.85
    conf = ((sampled_valid - low) / (high - low)).clamp(0.0, 1.0)
    flat_albedo = conf * flat_unpremul + (1.0 - conf) * flat_nearest
    
    # 2. Get RGB result
    flat_albedo = flat_albedo.squeeze(0).permute(1, 2, 0).numpy() # (H, W, 3) RGB 0-1
    flat_albedo_u8 = (flat_albedo * 255).astype(np.uint8)

    # 3. Convert back to BGR for cv2.imwrite compatibility
    return cv2.cvtColor(flat_albedo_u8, cv2.COLOR_RGB2BGR)


def process_doc3d_dataset(raw_dir: Path, output_dir: Path, max_partitions: int = 3, clean: bool = True):
    """
    1. Unzips the Doc3D dataset into the unzipped dir (raw_dir / "doc3d_raw")
    2. flattens out the partitions into the category dirs (unzipped_dir / "img", "alb", "bm", "uv")
    3. resizes the images to target size (256, 256)
    4. creates the flat albedo and shading images
    5. create train/val split file with random 80/20 split
    6. normalizes uv map shape
    clean: if True, delete high resolution data not needed for training or testing
    """

    print("Processing Doc3D dataset...")

    # dir for unzipped doc3d files
    unzipped_root = raw_dir / "doc3d_raw"
    # create split file with random 80/20 split
    split_path = output_dir / "split.json"

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # unzip the files for each data type
    for category in tqdm(["img", "alb", "bm", "uv"], desc="Unzipping Doc3D Data Types"):
        # create subdir
        cat_dir = unzipped_root / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # unzip
        zips = list((raw_dir / "doc3d").glob(f"{category}_*.zip"))
        zips = [zip_path for zip_path in zips if int(zip_path.stem.split("_")[1]) <= max_partitions]
        for zip_partition in tqdm(zips, desc=f"Unzipping {category} Partitions"):
            unzip_file(zip_partition, cat_dir, skip_dirs=[])

        # Flatten nested directories
        for sib in list(cat_dir.iterdir()):
            if sib.is_dir():
                # Move all files from subdir to root cat_dir
                for f in sib.rglob("*"):
                    if f.is_file():
                        shutil.move(str(f), str(cat_dir / f.name))
                # Now it is safe to remove the directory
                shutil.rmtree(sib)

    print("Done unzipping Doc3D data types.")

    out_img = output_dir / "img" # raw input img
    out_alb = output_dir / "alb" # wrapped albedo img
    out_shading = output_dir / "shading" # wrapped shading img
    out_uv = output_dir / "uv" # uv map
    out_flat = output_dir / "flat_albedo" # flat albedo img
    
    for d in [out_flat, out_shading, out_uv, out_img, out_alb]:
        d.mkdir(parents=True, exist_ok=True)

    img_files = list((unzipped_root / "img").glob("*.png"))
    print(f"Found {len(img_files)} Doc3D images to process.")

    for img_path in tqdm(img_files, desc="Processing Doc3D Images"):
        img_id = img_path.stem

        alb_path = unzipped_root / "alb" / f"{img_id}.png"
        bm_path = unzipped_root / "bm" / f"{img_id}.mat"
        uv_path = unzipped_root / "uv" / f"{img_id}.exr"

        if not (alb_path.exists() and bm_path.exists() and uv_path.exists()):
            print(f"Error: {alb_path}, {bm_path}, or {uv_path} not found.")
            continue

        ###### RESIZING ######
        # resize inp img
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: {img_path} not found.")
            continue
        img_resized = cv2.resize(img, TARGET_SIZE)
        # output to out_img dir
        cv2.imwrite(str(out_img / f"{img_id}.png"), img_resized)

        # resize uv (EXR)
        uv = cv2.imread(str(uv_path), cv2.IMREAD_UNCHANGED)
        if uv is None:
            print(f"Error: {uv_path} not found.")
            continue
        
        # Doc3D UV EXR has 3 channels with the following semantics:
        #   ch0 = document mask (~1.0 inside document, 0 in background)
        #   ch1 = V coordinate in [0, 1] (needs flipping: V_out = 1 - ch1)
        #   ch2 = U coordinate in [0, 1] (no flip needed: U_out = ch2)
        #
        # Standardize to the same final encoding used by UVDoc targets:
        #   stored ch0 = U in [0, 1], stored ch1 = V in [0, 1]
        #   stored ch2 = document mask
        # Background pixels use -1.0 sentinel so cv2.remap with
        # BORDER_CONSTANT returns borderValue (black) instead of sampling
        # the flat image corner (white paper).
        # Downstream remap usage:
        #   map_x = uv[..., 0] * (W - 1), map_y = uv[..., 1] * (H - 1)
        if uv.shape[-1] >= 3:
            mask = uv[..., 0]
            # Create a clean mask for the document area
            doc = mask > 0.1
            
            # 1. Fill background with 0.0 before resizing to avoid sentinel bleeding
            u_chan = np.where(doc, np.clip(uv[..., 2], 0.0, 1.0), 0.0)
            v_chan = np.where(doc, np.clip(1.0 - uv[..., 1], 0.0, 1.0), 0.0)
            
            # Stack U, V, and Mask for a single resize operation
            uv_stack = np.stack([u_chan, v_chan, mask], axis=-1).astype(np.float32)
            uv_resized = cv2.resize(uv_stack, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            
            # 2. Re-apply the -1.0 sentinel to the resized UV map
            # We threshold the interpolated mask to determine the new document boundary
            res_mask = uv_resized[..., 2]
            final_doc_mask = res_mask > 0.5
            
            # Set background to -1.0 sentinel
            uv_resized[..., 0] = np.where(final_doc_mask, uv_resized[..., 0], -1.0)
            uv_resized[..., 1] = np.where(final_doc_mask, uv_resized[..., 1], -1.0)
            # Channel 2 remains the continuous [0, 1] mask
        elif uv.shape[-1] == 2:
            # Fallback for 2-channel case (less common in Doc3D)
            u_chan = np.clip(1.0 - uv[..., 1], 0.0, 1.0)
            v_chan = np.clip(uv[..., 0], 0.0, 1.0)
            uv_stack = np.stack([u_chan, v_chan], axis=-1).astype(np.float32)
            uv_resized = cv2.resize(uv_stack, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            # Pad with a zero mask channel
            uv_resized = np.pad(uv_resized, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        
        # Ensure we have exactly 3 channels for EXR saving
        if uv_resized.shape[-1] > 3:
            uv_resized = uv_resized[..., :3]
        elif uv_resized.shape[-1] == 2:
            uv_resized = np.pad(uv_resized, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)

        cv2.imwrite(str(out_uv / f"{img_id}.exr"), uv_resized)

        bm_data, source = load_mat_file(bm_path)
        if bm_data is None or source is None:
            print(f"Error: {bm_path} not found.")
            continue
        
        # Normalize BM using ORIGINAL image dimensions
        # Doc3D BM is in pixel coordinates of the original image (448x448)
        if bm_data.ndim == 3 and bm_data.shape[0] == 2: # Check Transpose
            if source == 'h5py':
                 # h5py reads (C, W, H). We want (H, W, C).
                 # This avoids the spatial transpose (rotation + flip).
                 bm_data = np.transpose(bm_data, (2, 1, 0))
            else:
                 # scipy reads (C, H, W). We want (H, W, C).
                 bm_data = np.transpose(bm_data, (1, 2, 0))
            
        # Get original dimensions from img (loaded earlier)
        h_orig, w_orig = img.shape[:2]
        
        # Normalize to [0, 1]
        if bm_data.max() > 1.0:
            bm_data = bm_data.astype(np.float32)
            # Channel 0 is X, Channel 1 is Y
            bm_data[..., 0] /= (w_orig - 1)
            bm_data[..., 1] /= (h_orig - 1)
            
        # Now resize the normalized grid
        # bm_resized = cv2.resize(bm_data, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        alb = cv2.imread(str(alb_path))
        if alb is None: 
            print(f"Error: {alb_path} not found.")
            continue
        alb_resized = cv2.resize(alb, TARGET_SIZE)
        cv2.imwrite(str(out_alb / f"{img_id}.png"), alb_resized)

        # create shading img from wrapped alb img and original img
        # We need to handle the background (where alb is black/zero or img is background)
        # Using the UV map to determine valid document area is robust.
        
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_alb = cv2.cvtColor(alb_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Mask: Valid pixels where UV map is non-zero
        # Background in Doc3D UV maps is (0,0,0).
        # We use a small threshold to handle interpolation artifacts at boundaries.
        mask = (np.max(uv_resized, axis=2) > 1e-6).astype(np.float32)
        
        # Calculate shading: Image = Albedo * Shading => Shading = Image / Albedo
        # Add epsilon to avoid division by zero
        shading = gray_img / (gray_alb + 1e-6)
        
        # Clip to [0, 1]
        shading = np.clip(shading, 0, 1)
        
        # Apply mask: Set background to 0
        shading = shading * mask
        
        shading_u8 = (shading * 255).astype(np.uint8)
        cv2.imwrite(str(out_shading / f"{img_id}.png"), shading_u8)
        
        # create flat albedo img from wrapped alb img and backward mapping
        if not bm_path.exists():
            print(f"Error: {bm_path} not found.")
            continue

        flat_albedo = reconstruct_flat_albedo_tensor(alb, cv2.resize(bm_data, alb.shape[:2], interpolation=cv2.INTER_LINEAR))
        flat_albedo = cv2.resize(flat_albedo, TARGET_SIZE)
        cv2.imwrite(str(out_flat / f"{img_id}.png"), flat_albedo)

    print("Done processing Doc3D images.")

    # create train/val split file with random 80/20 split
    train_val_split = {
        "train": [],
        "val": []
    }
    for img_path in tqdm(list((out_img).glob("*.png")), desc="Creating Train/Val Split"):
        img_id = img_path.stem
        if np.random.random() < 0.8:
            train_val_split["train"].append(img_id)
        else:
            train_val_split["val"].append(img_id)

    # save train/val split file
    with open(split_path, "w") as f:
        f.write(orjson.dumps(train_val_split).decode("utf-8"))

    if clean:
        # delete doc3d_raw bm and uv
        shutil.rmtree(raw_dir / "bm")
        shutil.rmtree(raw_dir / "uv")

    print("Done creating train/val split file.")

def uncapsulate_inv3d_data(raw_dir: Path, mid_dir: Path):
    """
    1. Locates all the zip files in the raw_dir OR nested within mid_dir (after untarring)
    2. Unzips the zip files into the mid_dir
    3. Within the mid_dir, locate all dirs with format "\d{5}" (e.g. 00000)
    4. For each dir, copy flat document (albedo), warped albedo, warped document (img) pngs and warped_UV.npz files 
    5. Drop the doc_dirs and keep the mid_dirs
    """
    print("Uncapsulating INV3D data...")
    
    # 1. Check for zips in raw_dir (manual downloads)
    zip_folders = list(raw_dir.rglob("*.zip"))
    for zip_folder in tqdm(zip_folders, desc=f"Unzipping INV3D Zip Files into {raw_dir}"):
        unzip_file(zip_folder, mid_dir, skip_dirs=[])
    
    # in paths/names
    flat_alb_name = "flat_document.png"
    warped_alb_name = "warped_albedo.png"
    warped_img_name = "warped_document.png"
    warped_UV_name = "warped_UV.npz"

    # create dirs in mid_dir for saving raw data/img (needed for testing)
    mid_flat_alb = mid_dir / "flat_alb"
    mid_warped_alb = mid_dir / "warped_alb"
    mid_img = mid_dir / "img"
    mid_uv = mid_dir / "uv"

    mid_flat_alb.mkdir(parents=True, exist_ok=True)
    mid_warped_alb.mkdir(parents=True, exist_ok=True)
    mid_img.mkdir(parents=True, exist_ok=True)
    mid_uv.mkdir(parents=True, exist_ok=True)
    
    # Locate all dirs within mid_dir with format 5 digits and copy the data to the mid_dirs
    # Using glob pattern for 5 digits [0-9][0-9][0-9][0-9][0-9]
    # search recursively in case unzip didn't flatten them
    doc_dirs = list(mid_dir.rglob("[0-9][0-9][0-9][0-9][0-9]"))
    for doc_dir in tqdm(doc_dirs, desc="Processing INV3D Document Dirs"):
        # create paths
        flat_alb_path = doc_dir / flat_alb_name
        warped_alb_path = doc_dir / warped_alb_name
        warped_img_path = doc_dir / warped_img_name
        warped_UV_path = doc_dir / warped_UV_name

        if all([flat_alb_path.exists(), warped_alb_path.exists(), warped_img_path.exists(), warped_UV_path.exists()]):
            # copy the data to the mid_dirs
            shutil.copy(str(flat_alb_path), str(mid_flat_alb / f"{doc_dir.name}.png"))
            shutil.copy(str(warped_alb_path), str(mid_warped_alb / f"{doc_dir.name}.png"))
            shutil.copy(str(warped_img_path), str(mid_img / f"{doc_dir.name}.png"))
            shutil.copy(str(warped_UV_path), str(mid_uv / f"{doc_dir.name}.npz"))

    # drop the doc_dirs and keep the mid_dirs
    # Be careful not to delete mid_dirs themselves if they match the pattern (unlikely)
    for d in doc_dirs:
        if d.is_dir() and d.name.isdigit():
            shutil.rmtree(d)

    print("Done uncapsulating INV3D data.")

def process_inv3d_data(mid_dir: Path, output_dir: Path, clean: bool = True):
    """
    Process the data in mid_dir and save the processed data to the output_dir
    clean: if True, delete high resolution data not needed for training or testing
    """
    print("Processing INV3D data...")
    
    # create dirs in output_dir for saving processed data
    out_flat_alb = output_dir / "flat_alb"
    out_warped_alb = output_dir / "warped_alb"
    out_img = output_dir / "img"
    out_uv = output_dir / "uv"
    out_shading = output_dir / "shading"

    for d in [out_flat_alb, out_warped_alb, out_img, out_uv, out_shading]:
        d.mkdir(parents=True, exist_ok=True)

    mid_flat_alb = mid_dir / "flat_alb"
    mid_warped_alb = mid_dir / "warped_alb"
    mid_img = mid_dir / "img"
    mid_uv = mid_dir / "uv"

    flat_alb_files = np.array([f.stem for f in mid_flat_alb.glob("*.png")])
    warped_alb_files = np.array([f.stem for f in mid_warped_alb.glob("*.png")])
    img_files = np.array([f.stem for f in mid_img.glob("*.png")])
    uv_files = np.array([f.stem for f in mid_uv.glob("*.npz")])

    stems = np.intersect1d(np.intersect1d(np.intersect1d(flat_alb_files, warped_alb_files), img_files), uv_files)

    for stem in tqdm(stems, desc="Processing INV3D Data"):
        flat_alb_path = mid_flat_alb / f"{stem}.png"
        warped_alb_path = mid_warped_alb / f"{stem}.png"
        img_path = mid_img / f"{stem}.png"
        uv_path = mid_uv / f"{stem}.npz"

        flat_alb = cv2.imread(str(flat_alb_path))
        warped_alb = cv2.imread(str(warped_alb_path))
        img = cv2.imread(str(img_path))
        
        # .npz contains multiple arrays, usually 'uv' or similar. 
        # But if you saved it as .npy, then np.load is enough. 
        # If it's a .npz, you need to access the array inside.
        with np.load(str(uv_path)) as data:
            # According to Inv3D, it's 'warped_UV.npz'
            # Let's assume the key is 'uv' or 'arr_0' or 'warped_UV'
            # Based on common practices, it might be the first file.
            uv = data[data.files[0]]

        # resize the images to TARGET_SIZE
        flat_alb = cv2.resize(flat_alb, TARGET_SIZE)
        warped_alb = cv2.resize(warped_alb, TARGET_SIZE)
        img = cv2.resize(img, TARGET_SIZE)
        uv = cv2.resize(uv, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

        # Standardize INV3D UV [mask, V_flipped, U] to [U, V, mask]
        # To match Doc3D/UVDoc encoding used in this script
        inv_mask = uv[..., 0]
        inv_v_flipped = uv[..., 1]
        inv_u = uv[..., 2]
        
        v = 1.0 - inv_v_flipped
        u = inv_u
        
        # New UV: [U, V, mask]
        uv_standard = np.stack([u, v, inv_mask], axis=-1)

        # create shading img from wrapped alb img and original img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_alb = cv2.cvtColor(warped_alb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Mask: Valid pixels where UV map is non-zero
        mask = inv_mask.astype(np.float32)
        
        # Calculate shading: Image = Albedo * Shading => Shading = Image / Albedo + 1e-6
        shading = gray_img / (gray_alb + 1e-6)
        
        # Clip to [0, 1]
        shading = np.clip(shading, 0, 1)
        
        # Apply mask: Set background to 0
        shading = shading * mask

        shading_u8 = (shading * 255).astype(np.uint8)

        cv2.imwrite(str(out_shading / f"{stem}.png"), shading_u8)
        cv2.imwrite(str(out_img / f"{stem}.png"), img)
        cv2.imwrite(str(out_warped_alb / f"{stem}.png"), warped_alb)
        cv2.imwrite(str(out_flat_alb / f"{stem}.png"), flat_alb)
        
        # save uv as npy
        np.save(str(out_uv / f"{stem}.npy"), uv_standard)

    if clean:
        # delete mid_flat_alb, mid_warped_alb, mid_uv (mid_img is needed for testing)
        shutil.rmtree(mid_flat_alb)
        shutil.rmtree(mid_warped_alb)
        shutil.rmtree(mid_uv)

    print("Done processing INV3D data.")

def create_inv3d_dataset(raw_dir: Path, mid_dir: Path, output_dir: Path):
    """
    1. Uncapsulate the INV3D data into the mid_dir in a clear structure
    2. Process the data in mid_dir and save the processed data to the output_dir
    """

    print("Create INV3D dataset...")

    # uncapsulate the INV3D data into the mid_dir in a clear structure
    uncapsulate_inv3d_data(raw_dir, mid_dir)

    process_inv3d_data(mid_dir, output_dir)

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=DOWNLOAD_DIR)
    parser.add_argument("--uvdoc_out_dir", default=UVDOC_DATASET_DIR)
    parser.add_argument("--doc3d_out_dir", default=DOC3D_DATASET_DIR)
    parser.add_argument("--inv3d_out_dir", default=INV3D_DATASET_DIR)
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--doc3d_dir", default=DOC3D_DIR)
    parser.add_argument("--skip_doc3d", action='store_true')

    args = parser.parse_args()
    
    raw = Path(args.raw_dir)
    uvdoc_out_dir = Path(args.uvdoc_out_dir)
    urls = args.url
    
    # Ensure directories exist
    raw.mkdir(parents=True, exist_ok=True)
    uvdoc_out_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset and unzip if UVDoc_final dir not present
    if not (raw / "UVDoc_final").exists():
        if not (raw / "UVDoc_final.zip").exists():
            print("UVDoc_final.zip not found. Downloading...")
            download_file(urls, raw / "UVDoc_final.zip")
        
        unzip_file(raw / "UVDoc_final.zip", raw)
        print("UVDoc_final dir downloaded and unzipped.")
    else:
        print("UVDoc_final dir already exists. Skipping download and unzip.")

    # Process the dataset
    process_uvdoc_dataset(raw / "UVDoc_final", uvdoc_out_dir)
    print("Done with UVDoc dataset.")

    # download Doc3D from huggingface
    if not args.skip_doc3d:
        print(f"Downloading Doc3D from huggingface {args.doc3d_dir}...")
        # number of partitions to download from all available 21 Doc3D partitions
        max_partitions = 8
        patterns = []
        for cat in ["img", "alb", "bm", "uv"]:
            for i in range(1, max_partitions + 1):
                patterns.append(f"doc3d/{cat}_{i}.zip")
            
        # downloads max partitions of img_n.zip, alb_n.zip, bm_n.zip, uv_n.zip
        print(f"Downloading {patterns} from {args.doc3d_dir}...")
        doc3d_dir = snapshot_download(repo_id=args.doc3d_dir, local_dir=raw, allow_patterns=patterns, repo_type="dataset")

        process_doc3d_dataset(raw, Path(args.doc3d_out_dir), max_partitions)
    else:
        print("Skipping Doc3D dataset.")

    raw = Path("/media/driftfich/Backup-Volume/downloads")
    raw.mkdir(parents=True, exist_ok=True)
    download_dir = raw / INV3D_DATASET_DIR / "raw"
    download_train_dir = download_dir / "train"
    download_val_dir = download_dir / "val"
    download_test_dir = download_dir / "test"
    download_train_dir.mkdir(parents=True, exist_ok=True)
    download_val_dir.mkdir(parents=True, exist_ok=True)
    download_test_dir.mkdir(parents=True, exist_ok=True)

    unzipped_dir = raw / INV3D_DATASET_DIR / "unzipped"
    unzipped_train_dir = unzipped_dir / "train"
    unzipped_val_dir = unzipped_dir / "val"
    unzipped_test_dir = unzipped_dir / "test"
    unzipped_train_dir.mkdir(parents=True, exist_ok=True)
    unzipped_val_dir.mkdir(parents=True, exist_ok=True)
    unzipped_test_dir.mkdir(parents=True, exist_ok=True)

    inv3d_train_partitions: int = 1
    inv3d_val_partitions: int = 0
    inv3d_test_partitions: int = 0

    for part, count, urls in zip(["train", "val", "test"], [inv3d_train_partitions, inv3d_val_partitions, inv3d_test_partitions], [INV3D_URLS_TRAIN, INV3D_URLS_VAL, INV3D_URLS_TEST]):
        for i in range(min(count, len(urls))):
            download_file(urls[i], raw / INV3D_DATASET_DIR / "raw" / part / f"inv3d_{part}_{i}.tar")
            untar_file(raw / INV3D_DATASET_DIR / "raw" / part / f"inv3d_{part}_{i}.tar", raw / INV3D_DATASET_DIR / "unzipped" / part / f"inv3d_{part}_{i}")
            create_inv3d_dataset(raw / INV3D_DATASET_DIR / "unzipped" / part / f"inv3d_{part}_{i}", raw / INV3D_DATASET_DIR / "processed" / part / f"inv3d_{part}_{i}", INV3D_DATASET_DIR / part)
            # remove the content of the unzipped dir
            shutil.rmtree(raw / INV3D_DATASET_DIR / "unzipped" / part / f"inv3d_{part}_{i}")

if __name__ == "__main__":
    main()

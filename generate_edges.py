import argparse
import json
import cv2
import numpy as np
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

def reconstruct_flat_albedo_cv(alb, bm, target_size):
    """
    Unwarps Doc3D albedo using Backward Map to get Flat Texture.
    Uses Torch for the grid_sample operation (faster/easier).
    """
    # 1. Explicitly move to RGB workspace
    alb_rgb = cv2.cvtColor(alb, cv2.COLOR_BGR2RGB)
    alb_t = torch.from_numpy(alb_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Prepare BM
    # BM is (H, W, 2) in pixels or normalized 0-1
    # We need to resize BM to the target size first if it isn't already
    bm_resized = cv2.resize(bm, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize BM to [-1, 1] for grid_sample
    # If max value > 1.0, assume pixel coords
    bm_norm = bm_resized.copy()
    if bm_norm.max() > 1.0:
        bm_norm[..., 0] /= (alb.shape[1] - 1) # width
        bm_norm[..., 1] /= (alb.shape[0] - 1) # height
        
    bm_norm = bm_norm * 2.0 - 1.0
    grid = torch.from_numpy(bm_norm).unsqueeze(0).float()
    
    # 2. Unwarp in RGB space
    flat = F.grid_sample(alb_t, grid, align_corners=True, mode='bilinear')
    flat = flat.squeeze(0).permute(1, 2, 0).numpy() # (H, W, 3) 0-1
    flat_u8 = (flat * 255).astype(np.uint8)
    
    # 3. Convert back to BGR for cv2.imwrite compatibility
    return cv2.cvtColor(flat_u8, cv2.COLOR_RGB2BGR)

def generate_uvdoc_edges(raw_dir: Path):
    print(f"\n--- Generating UVDoc Edges in {raw_dir} ---")
    src_meta_dir = raw_dir / "metadata_sample"
    src_tex_dir = raw_dir / "textures"
    src_img_dir = raw_dir / "img"
    src_grid_dir = raw_dir / "grid2d"
    
    dest_flat_dir = raw_dir / "flat_edges"
    dest_warped_dir = raw_dir / "warped_edges"
    
    dest_flat_dir.mkdir(exist_ok=True, parents=True)
    dest_warped_dir.mkdir(exist_ok=True, parents=True)
    
    if not src_meta_dir.exists():
        print("UVDoc metadata not found. Skipping.")
        return

    meta_files = list(src_meta_dir.glob("*.json"))
    flat_cache = {}
    
    for meta_path in tqdm(meta_files, desc="UVDoc"):
        img_id = meta_path.stem
        dest_flat_path = dest_flat_dir / f"{img_id}.png"
        dest_warped_path = dest_warped_dir / f"{img_id}.png"
        
        # Optimization: Skip if done
        if dest_flat_path.exists() and dest_warped_path.exists():
            continue
            
        try:
            with open(meta_path, 'r') as f: meta = json.load(f)
        except: continue
            
        # 1. Flat Edge Map
        tex_rel_path = meta.get("texture_name")
        if not tex_rel_path: continue
        tex_name = Path(tex_rel_path).name
        
        if tex_name in flat_cache:
            flat_map = flat_cache[tex_name]
        else:
            tex_path = src_tex_dir / tex_name
            if not tex_path.exists(): continue
            tex_img = cv2.imread(str(tex_path))
            flat_map = compute_edge_map_from_image(tex_img)
            if flat_map is not None: flat_cache[tex_name] = flat_map
        
        if flat_map is None: continue
        cv2.imwrite(str(dest_flat_path), (flat_map * 255).astype(np.uint8))

        # 2. Warped Edge Map (via Sparse Grid Interpolation)
        geom_name = meta.get("geom_name")
        mat_path = src_grid_dir / f"{geom_name}.mat"
        if not mat_path.exists(): continue
        
        map_data, source = load_mat_file(mat_path)
        if map_data is None: continue
        
        if source == 'h5py' and map_data.shape[0] == 2: 
            map_data = np.transpose(map_data, (2, 1, 0))
        elif source == 'scipy' and map_data.ndim == 3 and map_data.shape[0] == 2: 
            map_data = np.transpose(map_data, (1, 2, 0))

        # Get original image dims
        img_path = src_img_dir / f"{img_id}.jpg"
        if not img_path.exists(): img_path = src_img_dir / f"{img_id}.png"
        if not img_path.exists(): continue
        
        # We need dims to normalize grid points
        # Using PIL or just cv2 shape is fast enough
        img_temp = cv2.imread(str(img_path))
        if img_temp is None: continue
        h_img, w_img = img_temp.shape[:2]

        # Interpolate Flat Map -> Warped Domain
        grid_h, grid_w, _ = map_data.shape
        points_x = map_data[..., 0].flatten() / w_img
        points_y = map_data[..., 1].flatten() / h_img
        
        u_grid, v_grid = np.meshgrid(np.linspace(0, 1, grid_w), np.linspace(0, 1, grid_h))
        
        points = np.stack([points_x, points_y], axis=1)
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, TARGET_SIZE[1]), np.linspace(0, 1, TARGET_SIZE[0]))
        
        # Inverse mapping: Where does each pixel in Warped Image come from in Flat UV?
        try:
            dense_u = griddata(points, u_grid.flatten(), (grid_x, grid_y), method='linear', fill_value=-1)
            dense_v = griddata(points, v_grid.flatten(), (grid_x, grid_y), method='linear', fill_value=-1)
        except: continue
        
        map_x = (dense_u * (TARGET_SIZE[1] - 1)).astype(np.float32)
        map_y = (dense_v * (TARGET_SIZE[0] - 1)).astype(np.float32)
        
        warped_map = cv2.remap(flat_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(dest_warped_path), (warped_map * 255).astype(np.uint8))

def generate_doc3d_edges(raw_dir: Path, output_dir: Path):
    print(f"\n--- Generating Doc3D Edges in {raw_dir} ---")
    
    # We look for the 'doc3d_raw' folder usually created by the build script
    # Or strict 'raw_dir' if passed directly
    if (raw_dir / "doc3d_raw").exists():
        root = raw_dir / "doc3d_raw"
    else:
        root = raw_dir # Assume we are inside doc3d_raw

    src_alb = root / "alb"
    src_bm = root / "bm"
    
    if not src_alb.exists():
        print(f"Doc3D 'alb' directory not found in {root}. Skipping.")
        return

    dest_flat_dir = output_dir / "flat_edges"
    dest_warped_dir = output_dir / "warped_edges"
    
    dest_flat_dir.mkdir(exist_ok=True, parents=True)
    dest_warped_dir.mkdir(exist_ok=True, parents=True)
    
    alb_files = list(src_alb.glob("*.png"))
    print(f"Found {len(alb_files)} Doc3D samples.")
    
    for alb_path in tqdm(alb_files, desc="Doc3D"):
        img_id = alb_path.stem
        dest_flat_path = dest_flat_dir / f"{img_id}.png"
        dest_warped_path = dest_warped_dir / f"{img_id}.png"
        
        # Optimization
        if dest_flat_path.exists() and dest_warped_path.exists():
            continue
            
        # 1. Warped Edges
        # Doc3D 'alb' IS the clean texture warped. We just run edge detection on it.
        # This is equivalent to UVDoc's synthetic warp logic but 100x faster.
        alb = cv2.imread(str(alb_path))
        if alb is None: continue
        
        warped_map = compute_edge_map_from_image(alb)
        if warped_map is not None:
            cv2.imwrite(str(dest_warped_path), (warped_map * 255).astype(np.uint8))
            
        # 2. Flat Edges
        # We need to unwarp 'alb' using 'bm' to get the flat texture
        bm_path = src_bm / f"{img_id}.mat"
        if not bm_path.exists(): continue
        
        bm_data, _ = load_mat_file(bm_path)
        if bm_data is None: continue
        
        # Ensure BM is (H, W, 2)
        if bm_data.ndim == 3 and bm_data.shape[0] == 2:
            bm_data = np.transpose(bm_data, (1, 2, 0))
            
        # Unwarp
        flat_alb = reconstruct_flat_albedo_cv(alb, bm_data, TARGET_SIZE)
        
        # Compute edges on flat
        flat_map = compute_edge_map_from_image(flat_alb)
        if flat_map is not None:
            cv2.imwrite(str(dest_flat_path), (flat_map * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="downloads", help="Root downloads folder containing UVDoc_final and/or doc3d_raw")
    args = parser.parse_args()
    
    root_path = Path(args.raw_dir)
    
    # Check for UVDoc
    uvdoc_path = root_path / "UVDoc_final"
    if uvdoc_path.exists():
        generate_uvdoc_edges(uvdoc_path)
    elif (root_path / "img").exists() and (root_path / "grid2d").exists():
        # Maybe raw_dir points directly to UVDoc
        generate_uvdoc_edges(root_path)
        
    # Check for Doc3D
    # The build script puts it in downloads/doc3d_raw
    output_dir = Path("dataset_doc3d")
    if (root_path / "doc3d_raw").exists():
        generate_doc3d_edges(root_path, output_dir) # Pass root, function handles subdir
    elif (root_path / "alb").exists() and (root_path / "bm").exists():
         # Maybe raw_dir points directly to Doc3D
        generate_doc3d_edges(root_path, output_dir)
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.io
import h5py
from scipy.interpolate import griddata

def load_mat_file(path):
    """Robustly loads a .mat file."""
    try:
        mat = scipy.io.loadmat(str(path))
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys: return None
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
            return None

def compute_canny_map(img_path, target_size=(256, 256)):
    """Computes a Distance Transform map from Canny edges."""
    try:
        if not img_path.exists(): return None
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
        return np.exp(-0.2 * dist).astype(np.float32)
    except:
        return None

def generate_edge_maps(raw_dir: Path):
    src_meta_dir = raw_dir / "metadata_sample"
    src_tex_dir = raw_dir / "textures"
    src_img_dir = raw_dir / "img"
    src_grid_dir = raw_dir / "grid2d"
    
    dest_flat_dir = raw_dir / "flat_edges"
    dest_warped_dir = raw_dir / "warped_edges"
    
    dest_flat_dir.mkdir(exist_ok=True, parents=True)
    dest_warped_dir.mkdir(exist_ok=True, parents=True)
    
    meta_files = list(src_meta_dir.glob("*.json"))
    print(f"Found {len(meta_files)} samples.")
    
    flat_cache = {}
    target_size = (256, 256)
    
    for meta_path in tqdm(meta_files, desc="Generating Synthetic Maps"):
        img_id = meta_path.stem
        dest_flat_path = dest_flat_dir / f"{img_id}.png"
        dest_warped_path = dest_warped_dir / f"{img_id}.png"
        
        if dest_flat_path.exists() and dest_warped_path.exists():
            continue
            
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except:
            continue
            
        # 1. Process/Cache Flat Edge Map
        tex_rel_path = meta.get("texture_name")
        if not tex_rel_path: continue
        tex_name = Path(tex_rel_path).name
        
        if tex_name in flat_cache:
            flat_map = flat_cache[tex_name]
        else:
            tex_path = src_tex_dir / tex_name
            flat_map = compute_canny_map(tex_path, target_size)
            if flat_map is not None:
                flat_cache[tex_name] = flat_map
        
        if flat_map is None: continue
        cv2.imwrite(str(dest_flat_path), (flat_map * 255).astype(np.uint8))

        # 2. Create Synthetic Warped Map using GT Grid
        geom_name = meta.get("geom_name")
        mat_path = src_grid_dir / f"{geom_name}.mat"
        if not mat_path.exists(): continue
        
        res = load_mat_file(mat_path)
        if res is None: continue
        map_data, source = res
        
        if source == 'h5py':
            if map_data.shape[0] == 2: map_data = np.transpose(map_data, (2, 1, 0))
        elif source == 'scipy':
            if map_data.ndim == 3 and map_data.shape[0] == 2: map_data = np.transpose(map_data, (1, 2, 0))

        # Get original image size for normalization
        img_path = src_img_dir / f"{img_id}.jpg"
        if not img_path.exists(): img_path = src_img_dir / f"{img_id}.png"
        img_temp = cv2.imread(str(img_path))
        if img_temp is None: continue
        h_img, w_img = img_temp.shape[:2]

        # Normalization and Scattered Interpolation (Scatter Points -> UV Map)
        grid_h, grid_w, _ = map_data.shape
        points_x = map_data[..., 0].flatten() / w_img
        points_y = map_data[..., 1].flatten() / h_img
        u_grid, v_grid = np.meshgrid(np.linspace(0, 1, grid_w), np.linspace(0, 1, grid_h))
        
        points = np.stack([points_x, points_y], axis=1)
        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, target_size[1]), np.linspace(0, 1, target_size[0]))
        
        # This creates the inverse map (Image -> UV)
        dense_u = griddata(points, u_grid.flatten(), (grid_x, grid_y), method='linear', fill_value=-1)
        dense_v = griddata(points, v_grid.flatten(), (grid_x, grid_y), method='linear', fill_value=-1)
        
        # Warp the flat_map to the image domain
        # remap expects coordinates in pixel units of the source image (flat_map)
        map_x = (dense_u * (target_size[1] - 1)).astype(np.float32)
        map_y = (dense_v * (target_size[0] - 1)).astype(np.float32)
        
        warped_map = cv2.remap(flat_map, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        cv2.imwrite(str(dest_warped_path), (warped_map * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="downloads/UVDoc_final")
    args = parser.parse_args()
    
    generate_edge_maps(Path(args.raw_dir))


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

# --- Configuration ---
# 256x256 is standard for mobile UNets (fast). 
# You can go higher (512) if needed, but start small.
TARGET_SIZE = (256, 256) 

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
    src_grid_dir = raw_dir / "grid2d"
    
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
        if not meta_path.exists(): continue
        
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            geom_name = meta.get("geom_name")
            if not geom_name: continue
        except: continue
            
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
        
        # -1 indicates background (outside document hull).
        # We can treat this as a mask for loss calculation.
        
        # Save
        is_train = (hash(img_id) % 5 != 0)
        dest_img_dir = img_dir_train if is_train else img_dir_val
        dest_mask_dir = mask_dir_train if is_train else mask_dir_val
        
        # Save Image
        cv2.imwrite(str(dest_img_dir / f"{img_id}.jpg"), img_resized)
        
        # Save UV Map as .npy (float precision is crucial for dewarping)
        np.save(str(dest_mask_dir / f"{img_id}.npy"), uv_map)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="downloads/extracted/UVDoc_final")
    parser.add_argument("--out_dir", default="dataset_unet")
    args = parser.parse_args()
    
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    
    if not raw.exists():
        # Fallback search
        found = list(Path("downloads").rglob("UVDoc_final"))
        if found: raw = found[0]
        
    process_unet_dataset(raw, out)
    print("Done.")

if __name__ == "__main__":
    main()

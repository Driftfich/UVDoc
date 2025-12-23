import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import orjson

def compute_canny_map(img_path, target_size=(256, 256), is_warped=False):
    """
    Computes a Distance Transform map from Canny edges.
    If is_warped=True, it suppresses edges at the document boundary.
    """
    try:
        if not img_path.exists():
            return None
            
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Resize to target size
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        
        # 1. Canny Edges
        blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 2. Boundary Suppression (Only for warped textures)
        if is_warped:
            # Create a mask of the document (where it's not black)
            _, mask = cv2.threshold(img_resized, 10, 255, cv2.THRESH_BINARY)
            # Erode the mask to ignore the outermost edges of the paper
            kernel = np.ones((5, 5), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=2)
            # Remove edges that fall on the boundary
            edges = cv2.bitwise_and(edges, mask_eroded)

        # 3. Distance Transform
        dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
        
        # 4. Gaussian-like falloff
        weight_map = np.exp(-0.2 * dist).astype(np.float32)
        
        return weight_map
        
    except Exception as e:
        print(f"Error computing Canny for {img_path}: {e}")
        return None

def generate_edge_maps(raw_dir: Path):
    src_meta_dir = raw_dir / "metadata_sample"
    src_tex_dir = raw_dir / "textures"
    src_warped_dir = raw_dir / "warped_textures"
    
    dest_flat_dir = raw_dir / "flat_edges" # Flat Canny (Source)
    dest_warped_dir = raw_dir / "warped_edges" # Warped Canny (Target)
    
    dest_flat_dir.mkdir(exist_ok=True, parents=True)
    dest_warped_dir.mkdir(exist_ok=True, parents=True)
    
    meta_files = list(src_meta_dir.glob("*.json"))
    print(f"Found {len(meta_files)} metadata files.")
    
    # Cache for flat textures (since many samples share the same texture)
    flat_cache = {}
    
    for meta_path in tqdm(meta_files, desc="Generating Maps"):
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
            
        # 1. Process Flat Texture
        tex_rel_path = meta.get("texture_name")
        if not tex_rel_path:
            continue
            
        tex_name = Path(tex_rel_path).name
        
        if tex_name in flat_cache:
            flat_map = flat_cache[tex_name]
        else:
            tex_path = src_tex_dir / tex_name
            flat_map = compute_canny_map(tex_path)
            if flat_map is not None:
                flat_cache[tex_name] = flat_map
        
        # 2. Process Warped Texture
        warped_path = src_warped_dir / f"{img_id}.png"
        warped_map = compute_canny_map(warped_path, is_warped=True)
        
        # 3. Save
        if flat_map is not None:
            cv2.imwrite(str(dest_flat_path), (flat_map * 255).astype(np.uint8))
        if warped_map is not None:
            cv2.imwrite(str(dest_warped_path), (warped_map * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="downloads/UVDoc_final")
    args = parser.parse_args()
    
    generate_edge_maps(Path(args.raw_dir))


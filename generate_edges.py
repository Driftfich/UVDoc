import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import orjson

def compute_maps(tex_path, target_size=(256, 256)):
    """
    Computes both Structural Weight Map (Morph+Canny) and Sobel Target (Gradient).
    """
    try:
        if not tex_path.exists():
            return None, None
            
        tex = cv2.imread(str(tex_path), cv2.IMREAD_GRAYSCALE)
        if tex is None:
            return None, None
            
        # Resize to target size
        tex_resized = cv2.resize(tex, (target_size[1], target_size[0]))
        
        # --- 1. Weight Map (Morphological Lines) ---
        # Invert so text is bright
        tex_inv = cv2.bitwise_not(tex_resized)
        _, bin_tex = cv2.threshold(tex_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        lines_h = cv2.morphologyEx(bin_tex, cv2.MORPH_CLOSE, kernel_h)
        edges_h = cv2.Canny(lines_h, 50, 150)
        
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        lines_v = cv2.morphologyEx(bin_tex, cv2.MORPH_CLOSE, kernel_v)
        edges_v = cv2.Canny(lines_v, 50, 150)
        
        edges_img = cv2.Canny(tex_resized, 100, 200)
        
        combined_edges = cv2.bitwise_or(edges_h, edges_v)
        combined_edges = cv2.bitwise_or(combined_edges, edges_img)
        
        dist = cv2.distanceTransform(cv2.bitwise_not(combined_edges), cv2.DIST_L2, 5)
        weight_map = np.exp(-0.2 * dist).astype(np.float32)
        
        # --- 2. Sobel Target (Visual Structure) ---
        # Standard Sobel Magnitude
        gx = cv2.Sobel(tex_resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(tex_resized, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        sobel_map = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        return weight_map, sobel_map
        
    except Exception as e:
        print(f"Error computing edges for {tex_path}: {e}")
        return None, None

def generate_edge_maps(raw_dir: Path):
    src_meta_dir = raw_dir / "metadata_sample"
    src_tex_dir = raw_dir / "textures"
    
    dest_weight_dir = raw_dir / "texture_edges" # Weights
    dest_sobel_dir = raw_dir / "texture_sobel" # Targets
    
    dest_weight_dir.mkdir(exist_ok=True, parents=True)
    dest_sobel_dir.mkdir(exist_ok=True, parents=True)
    
    meta_files = list(src_meta_dir.glob("*.json"))
    print(f"Found {len(meta_files)} metadata files.")
    
    texture_cache = {}
    
    for meta_path in tqdm(meta_files, desc="Generating Maps"):
        img_id = meta_path.stem
        
        dest_weight_path = dest_weight_dir / f"{img_id}.png"
        dest_sobel_path = dest_sobel_dir / f"{img_id}.png"
        
        if dest_weight_path.exists() and dest_sobel_path.exists():
            continue
            
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except:
            continue
            
        tex_rel_path = meta.get("texture_name")
        if not tex_rel_path:
            continue
            
        tex_name = Path(tex_rel_path).name
        
        if tex_name in texture_cache:
            weight_map, sobel_map = texture_cache[tex_name]
        else:
            tex_path = src_tex_dir / tex_name
            weight_map, sobel_map = compute_maps(tex_path)
            if weight_map is not None:
                texture_cache[tex_name] = (weight_map, sobel_map)
        
        if weight_map is not None:
            cv2.imwrite(str(dest_weight_path), (weight_map * 255).astype(np.uint8))
            cv2.imwrite(str(dest_sobel_path), (sobel_map * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="downloads/UVDoc_final")
    args = parser.parse_args()
    
    generate_edge_maps(Path(args.raw_dir))


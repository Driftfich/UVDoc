import cv2
import numpy as np
import os
from pathlib import Path

# Enable EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def test_uv_warp(sample_id, base_dir=Path("dataset_doc3d")):
    print(f"--- Testing UV Warp for ID: {sample_id} ---")
    
    # 1. Load the files
    flat_path = base_dir / "flat_albedo" / f"{sample_id}.png"
    uv_path = base_dir / "uv" / f"{sample_id}.exr"
    alb_path = base_dir / "alb" / f"{sample_id}.png"
    
    for p in [flat_path, uv_path, alb_path]:
        if not p.exists():
            print(f"Missing file: {p}")
            return

    flat = cv2.imread(str(flat_path))
    uv = cv2.imread(str(uv_path), cv2.IMREAD_UNCHANGED)
    alb_gt = cv2.imread(str(alb_path))
    
    # 2. Prepare mapping
    # uv[..., 0] is U (horizontal/columns), uv[..., 1] is V (vertical/rows)
    # Note: build_uvdoc_unet.py now standardizes UV to (U, V) and flips V for OpenCV.
    h_flat, w_flat = flat.shape[:2]
    
    # map_x = u * (W-1), map_y = v * (H-1)
    # uv[..., 0] -> U, uv[..., 1] -> V (already flipped for OpenCV)
    map_x = (uv[..., 0] * (w_flat - 1)).astype(np.float32)
    map_y = (uv[..., 1] * (h_flat - 1)).astype(np.float32)
    
    # 3. Perform Warp (Flat -> Wrapped)
    wrapped_recon = cv2.remap(flat, map_x, map_y, 
                              interpolation=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=0)
    
    # 4. Save results for visual inspection
    cv2.imwrite("test_uv_flat_input.png", flat)
    cv2.imwrite("test_uv_wrapped_recon.png", wrapped_recon)
    cv2.imwrite("test_uv_wrapped_gt.png", alb_gt)
    
    # 5. Calculate Difference
    diff = cv2.absdiff(wrapped_recon, alb_gt)
    mean_diff = np.mean(diff)
    print(f"Reconstruction Mean Pixel Difference: {mean_diff:.4f}")
    
    if mean_diff < 5.0: # Threshold for "mostly equal"
        print("SUCCESS: The UV map correctly warps the flat albedo back to the wrapped state.")
    else:
        print("FAILURE: Significant difference found. Check if flat_albedo is still rotated or if UV channels are swapped.")

if __name__ == "__main__":
    # Pick a sample ID from your directory
    sample_id = "1_1_1-pr_Page_141-PZU0001" # Your example ID
    test_uv_warp(sample_id)
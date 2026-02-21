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
    # Processed .exr stores: ch0 = U, ch1 = V, ch2 = document mask.
    # Document pixels: U,V ∈ [0,1]. Background pixels: U=V=-1 (sentinel).
    # cv2.remap with BORDER_CONSTANT/borderValue=0: any map_x/map_y outside
    # [0, W-1] or [0, H-1] returns black — correctly handles background.
    # Older processed files may have background=0.0 instead of -1.0; the
    # mask channel (ch2) is used to explicitly zero those out.
    h_flat, w_flat = flat.shape[:2]
    doc_mask = (uv[..., 2] > 0.05).astype(np.uint8)
    map_x = (uv[..., 0] * (w_flat - 1)).astype(np.float32)
    map_y = (uv[..., 1] * (h_flat - 1)).astype(np.float32)
    # Force background map coords out of bounds so remap returns borderValue
    map_x = np.where(doc_mask, map_x, -1.0).astype(np.float32)
    map_y = np.where(doc_mask, map_y, -1.0).astype(np.float32)

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
    # pick a random sample id from the dataset_doc3d/flat_albedo directory
    flat_albedo_dir = Path("dataset_doc3d/flat_albedo")
    # sample_id = np.random.choice(list(flat_albedo_dir.glob("*.png")))
    # sample_id = sample_id.stem
    sample_id = "1_1_1-pr_Page_141-PZU0001"
    test_uv_warp(sample_id)
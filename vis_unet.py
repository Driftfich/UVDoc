import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from train_unet import MobileNetV3UNet, UVDocUNetDataset
from scipy.interpolate import griddata

def infer_high_res(model, img_path, device='cpu', num_points=2500):
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"Could not load {img_path}")
        return None, None, None
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_orig.shape[:2]
    
    # 2. Resize for Model (256x256)
    img_resized = cv2.resize(img_orig, (256, 256))
    
    # 3. Prep Input (Normalize + CoordConv)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_ch = np.transpose(img_norm, (2, 0, 1)) # (3, 256, 256)
    
    # Generate Coord Grids (256x256)
    h, w = 256, 256
    xx_ones = np.ones((h, w), dtype=np.int32)
    xx_range = np.arange(w, dtype=np.int32)
    xx = xx_ones * xx_range[None, :]
    yy_ones = np.ones((h, w), dtype=np.int32)
    yy_range = np.arange(h, dtype=np.int32)
    yy = yy_range[:, None] * yy_ones
    
    xx = xx / (w - 1) * 2 - 1
    yy = yy / (h - 1) * 2 - 1
    grid = np.stack([xx, yy], axis=0).astype(np.float32)
    
    input_np = np.concatenate([img_ch, grid], axis=0)
    input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)
    
    # 4. Inference
    with torch.no_grad():
        pred = model(input_tensor)
        
    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    uv_pred_low = pred[..., :2]
    mask_pred_low = pred[..., 2]
    
    # 5. Smooth High-Quality Backward Map Generation using Scipy GridData
    
    target_h = int(max(h_orig, w_orig))
    target_w = int(target_h / 1.414)
    
    # Filter valid points from prediction
    valid_mask = mask_pred_low > 0.5
    
    y_idx, x_idx = np.where(valid_mask)
    src_y = y_idx / (h - 1) # 0-1
    src_x = x_idx / (w - 1) # 0-1
    
    dst_u = uv_pred_low[valid_mask, 0]
    dst_v = uv_pred_low[valid_mask, 1]
    
    # Subsample Control Points for Speed & Smoothness
    # num_points default 2500 (50x50 grid equivalent)
    total_valid = len(src_x)
    if total_valid > num_points:
        step = total_valid // num_points
        src_x = src_x[::step]
        src_y = src_y[::step]
        dst_u = dst_u[::step]
        dst_v = dst_v[::step]
        
    points = np.stack([dst_u, dst_v], axis=1) # Input: Target UV
    values_x = src_x # Output: Source X
    values_y = src_y # Output: Source Y
    
    # Create regular grid for target image
    # Optimization: Interpolate on a smaller grid (e.g. 512x512) then resize to full
    # This is much faster than griddata on 2000x2000
    interp_h, interp_w = 512, int(512 / 1.414)
    
    grid_u, grid_v = np.meshgrid(
        np.linspace(0, 1, interp_w),
        np.linspace(0, 1, interp_h)
    )
    
    # Interpolate Surface (Backward Map)
    # Cubic is smoother for warping
    try:
        back_map_x_low = griddata(points, values_x, (grid_u, grid_v), method='cubic', fill_value=-1)
        back_map_y_low = griddata(points, values_y, (grid_u, grid_v), method='cubic', fill_value=-1)
    except Exception as e:
        print(f"Interpolation failed: {e}")
        return img_orig, None, None
    
    # Resize map to full resolution
    back_map_x = cv2.resize(back_map_x_low, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    back_map_y = cv2.resize(back_map_y_low, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert normalized back_map to pixel coordinates
    map_x = (back_map_x * (w_orig - 1)).astype(np.float32)
    map_y = (back_map_y * (h_orig - 1)).astype(np.float32)
    
    # Mask invalid regions (outside hull)
    # Check for -1 (or near -1 due to cubic overshoot)
    mask_valid = (back_map_x > -0.1) & (back_map_y > -0.1) & (back_map_x < 1.1) & (back_map_y < 1.1)
    
    # Remap
    dewarped = cv2.remap(img_orig, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    dewarped[~mask_valid] = 0
    
    # --- Raw Splat for Comparison ---
    dewarped_raw = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    uv_pred_high = cv2.resize(uv_pred_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    mask_pred_high = cv2.resize(mask_pred_low, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    valid_mask_high = mask_pred_high > 0.5
    y_high, x_high = np.where(valid_mask_high)
    u_high = uv_pred_high[valid_mask_high, 0]
    v_high = uv_pred_high[valid_mask_high, 1]
    u_t = np.clip((u_high * (target_w - 1)).astype(int), 0, target_w - 1)
    v_t = np.clip((v_high * (target_h - 1)).astype(int), 0, target_h - 1)
    dewarped_raw[v_t, u_t] = img_orig[y_high, x_high]
    
    return img_orig, dewarped, dewarped_raw

def visualize_results(model_path, data_dir, raw_dir, num_samples=3, device='cpu', points=2500):
    model = MobileNetV3UNet(num_classes=3)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    model.to(device)
    model.eval()
    
    val_names = [p.name for p in Path(data_dir).joinpath("val/images").glob("*.jpg")]
    
    if not val_names:
        print("No validation images found.")
        return
        
    selected_names = np.random.choice(val_names, min(len(val_names), num_samples), replace=False)
    
    for name in selected_names:
        stem = Path(name).stem
        matches = list(Path(raw_dir).glob(f"{stem}.*"))
        if not matches: continue
        raw_img_path = matches[0]
        
        print(f"Processing {raw_img_path.name}...")
        orig, res, raw = infer_high_res(model, raw_img_path, device, points)
        
        if res is not None:
            out_name = f"vis_highres_{stem}.png"
            h, w = orig.shape[:2]
            scale = h / res.shape[0]
            new_w = int(res.shape[1] * scale)
            
            res_disp = cv2.resize(res, (new_w, h))
            raw_disp = cv2.resize(raw, (new_w, h))
            
            combined = np.hstack([orig, raw_disp, res_disp])
            cv2.imwrite(out_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"Saved {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet_best.pth")
    parser.add_argument("--data", default="dataset_unet")
    parser.add_argument("--raw", default="downloads/extracted/UVDoc_final/img")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--points", type=int, default=2500, help="Number of control points for interpolation")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_results(args.model, args.data, args.raw, args.n, device, args.points)

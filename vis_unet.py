import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from train_unet import MobileNetV3UNet, FastViTUNet, UVDocUNetDataset
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Union
import orjson 

class TPSSolver:
    def __init__(self, device='cpu'):
        self.device = device

    def fit_transform(self, sources, targets, grid_shape):
        """
        sources: (N, 2) points in target space (u, v) -> [0, 1]
        targets: (N, 2) points in source image space (x, y) -> normalized [-1, 1] usually, or 0-1
        grid_shape: (H, W) for output dense grid
        """
        # We want to find f: Source(uv) -> Target(xy)
        # So inputs to TPS are (u,v), outputs are (x,y)
        
        P = torch.tensor(sources, dtype=torch.float32, device=self.device) # (N, 2)
        V = torch.tensor(targets, dtype=torch.float32, device=self.device) # (N, 2)
        
        N = P.shape[0]
        
        # Construct K matrix
        # K_ij = phi(|x_i - x_j|)
        # phi(r) = r^2 * log(r)
        
        # P: (N, 2)
        # P_dist: (N, N) pairwise distances
        P_norm = (P ** 2).sum(1).view(-1, 1)
        dist_sq = P_norm + P_norm.view(1, -1) - 2.0 * torch.mm(P, P.t())
        dist_sq = dist_sq.clamp(min=1e-9) # Avoid 0 for log
        dist = torch.sqrt(dist_sq)
        
        K = dist_sq * torch.log(dist)
        # Handle diagonal (r=0) -> 0 * -inf = NaN. Limit is 0.
        # But we clamped min, so we have epsilon^2 * log(epsilon) which is fine small number.
        
        # Regularization lambda
        lmbda = 0.01
        I = torch.eye(N, device=self.device)
        
        # Build System:
        # [K + lI   P   1] [W] = [V]
        # [P^T      0   0] [A]   [0]
        # [1^T      0   0] [b]   [0]
        
        # Augmented P matrix: [P, 1] -> (N, 3)
        ones = torch.ones(N, 1, device=self.device)
        P_aug = torch.cat([P, ones], dim=1)
        
        # Top Left: K + lI
        TL = K + lmbda * I
        
        # Top Right: P_aug
        TR = P_aug
        
        # Bottom Left: P_aug^T
        BL = P_aug.t()
        
        # Bottom Right: Zeros (3, 3)
        BR = torch.zeros(3, 3, device=self.device)
        
        # Full Matrix L: (N+3, N+3)
        L_top = torch.cat([TL, TR], dim=1)
        L_bot = torch.cat([BL, BR], dim=1)
        L = torch.cat([L_top, L_bot], dim=0)
        
        # RHS Y: (N+3, 2)
        # V is (N, 2)
        Y_top = V
        Y_bot = torch.zeros(3, 2, device=self.device)
        Y = torch.cat([Y_top, Y_bot], dim=0)
        
        # Solve
        try:
            # Cholesky/LU might fail if not positive definite, use lstsq or pinv
            # L is symmetric? K is symmetric.
            # L is indefinite (saddle point problem).
            # Use torch.linalg.solve (LU) or lstsq
            weights = torch.linalg.solve(L, Y)
        except RuntimeError:
            weights = torch.linalg.lstsq(L, Y).solution
            
        self.weights = weights[:N] # (N, 2)
        self.affine = weights[N:]  # (3, 2) -> (a1, a2, b) for x and y
        self.control_points = P
        
        return self.evaluate_grid(grid_shape)

    def evaluate_grid(self, shape):
        h, w = shape
        
        # Create dense grid
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, h, device=self.device),
            torch.linspace(0, 1, w, device=self.device),
            indexing='ij'
        )
        grid_points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1) # (H*W, 2)
        
        # Chunking to avoid OOM with large grids against large N
        # Matrix is (HW, N). If HW=250k, N=2500 -> 625M floats -> 2.5GB. OK for GPU.
        # But if N=5000 or grid is larger, might be issue.
        
        batch_size = 10000
        n_points = grid_points.shape[0]
        outputs = []
        
        for i in range(0, n_points, batch_size):
            batch = grid_points[i:i+batch_size]
            
            # Distance to all control points
            # batch: (B, 2), ctrl: (N, 2)
            # dist: (B, N)
            
            b_sq = (batch ** 2).sum(1).view(-1, 1)
            c_sq = (self.control_points ** 2).sum(1).view(1, -1)
            d_sq = b_sq + c_sq - 2.0 * torch.mm(batch, self.control_points.t())
            d_sq = d_sq.clamp(min=1e-9)
            d = torch.sqrt(d_sq)
            
            K_batch = d_sq * torch.log(d)
            
            # Affine part
            # [batch, 1] * affine
            ones = torch.ones(batch.shape[0], 1, device=self.device)
            batch_aug = torch.cat([batch, ones], dim=1)
            affine_part = torch.mm(batch_aug, self.affine)
            
            # TPS part
            tps_part = torch.mm(K_batch, self.weights)
            
            outputs.append(affine_part + tps_part)
            
        res = torch.cat(outputs, dim=0)
        res = res.view(h, w, 2) # (H, W, 2) -> (x, y)
        
        return res.cpu().numpy()

def axis_alignment_preprocessing(model, img_orig, device, expand_ratio=0.05):
    """
    Richtet das Dokument vor der eigentlichen Entzerrung grob aus und schneidet es zu.
    Basiert auf der vorhergesagten Maske und dem Minimum-Area-Rectangle.
    """
    h_orig, w_orig = img_orig.shape[:2]
    
    # Schneller Forward-Pass für die Maske
    img_resized = cv2.resize(img_orig, (256, 256))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_ch = np.transpose(img_norm, (2, 0, 1))

    h, w = 256, 256
    xx = np.arange(w, dtype=np.float32)
    yy = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)
    xx = xx / (w - 1) * 2 - 1
    yy = yy / (h - 1) * 2 - 1
    grid = np.stack([xx, yy], axis=0)
    
    input_np = np.concatenate([img_ch, grid], axis=0)
    input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_logits = model(input_tensor)
        pred = torch.sigmoid(pred_logits)
    
    # Maske extrahieren und auf 0-255 skalieren
    mask_pred = pred[0, 2].cpu().numpy()
    mask_pred_u8 = (mask_pred * 255).astype(np.uint8)
    
    # Guided Filter anwenden (Guide-Bild ist der graue 256x256 Input)
    guide = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    refined_mask = cv2.ximgproc.guidedFilter(guide=guide, src=mask_pred_u8, radius=8, eps=50)
    
    # Die verfeinerte Maske binarisieren
    mask_bin = (refined_mask > 127).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_orig # Fallback, falls keine Kontur gefunden wird
        
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return img_orig
        
    # Minimum Area Rectangle (rotiertes Rechteck)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    
    # Eckpunkte wieder auf die Originalauflösung hochskalieren
    box[:, 0] *= (w_orig / 256.0)
    box[:, 1] *= (h_orig / 256.0)
    
    # Punkte sortieren: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    rect_pts = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect_pts[0] = box[np.argmin(s)]
    rect_pts[2] = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    rect_pts[1] = box[np.argmin(diff)]
    rect_pts[3] = box[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect_pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Etwas Padding hinzufügen (expand_ratio), damit keine Ecken abgeschnitten werden
    pad_x = int(maxWidth * expand_ratio)
    pad_y = int(maxHeight * expand_ratio)
    
    dst_padded = np.array([
        [pad_x, pad_y],
        [maxWidth + pad_x - 1, pad_y],
        [maxWidth + pad_x - 1, maxHeight + pad_y - 1],
        [pad_x, maxHeight + pad_y - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect_pts, dst_padded)
    warped = cv2.warpPerspective(img_orig, M, (maxWidth + 2*pad_x, maxHeight + 2*pad_y))
    
    return warped

def refine_lattice(back_map_x, back_map_y, mask_pred_bin, method='gaussian'):
    """
    Refines the backward mapping (H, W).
    1. Smooths high-freq jitter.
    2. Aligns to borders using mask signal.
    """
    h, w = back_map_x.shape
    
    # --- 1. Border Alignment Strategy ---
    # Heuristic: If we are near the edge of the mask, the UV map should be near 0 or 1.
    # But backward map is (u,v) -> (x,y).
    # We want to smooth the VALUES (x,y) with respect to spatial (u,v).
    # "Jitter" means: neighbors in (u,v) point to far away (x,y) or erratic (x,y).
    
    # Simple smoothing is robust.
    if method == 'gaussian':
        # Gaussian Blur
        # Sigma=1-2 pixels is usually enough to kill jitter without destroying structure
        bx = cv2.GaussianBlur(back_map_x, (5, 5), 1.5)
        by = cv2.GaussianBlur(back_map_y, (5, 5), 1.5)
        return bx, by
        
    elif method == 'bilateral':
        # Bilateral: Preserve edges in the map (tears) while smoothing noise.
        # But documents shouldn't have tears usually.
        bx = cv2.bilateralFilter(back_map_x.astype(np.float32), 9, 75, 75)
        by = cv2.bilateralFilter(back_map_y.astype(np.float32), 9, 75, 75)
        return bx, by
        
    return back_map_x, back_map_y


def infer_high_res(model, img_input: Union[str, Path, np.ndarray], device='cpu', num_points=2500, method='griddata', refine=False):
    # NEU: Unterscheide zwischen Dateipfad und Numpy-Array
    if isinstance(img_input, (str, Path)):
        img_orig = cv2.imread(str(img_input))
        if img_orig is None:
            print(f"Could not load {img_input}")
            return None, None, None, None, None, None
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    else:
        img_orig = img_input.copy()
        
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
        pred_logits = model(input_tensor)
        pred = torch.sigmoid(pred_logits)
        
    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    uv_pred_low = pred[..., :2]
    mask_pred_low = pred[..., 2]
    # shading_pred_low = pred[..., 3:4] # 1 channel luma
    
    # Resize shading to original resolution
    # shading_high = cv2.resize(shading_pred_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    # if shading_high.ndim == 2:
    #     shading_high = shading_high[..., None]
    
    # Mask shading with predicted mask (upscaled) to remove background artifacts
    mask_pred_high = cv2.resize(mask_pred_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    mask_pred_bin = (mask_pred_high > 0.5).astype(np.float32)
    
    # Refine mask: Keep largest component only
    mask_u8 = (mask_pred_bin * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_u8 = np.zeros_like(mask_u8)
        cv2.drawContours(mask_u8, [largest], -1, 255, -1)
        mask_pred_bin = (mask_u8 > 127).astype(np.float32)

    # Erode mask slightly to be safe
    kernel = np.ones((9, 9), np.uint8)
    mask_pred_bin_eroded = cv2.erode(mask_pred_bin, kernel, iterations=1)
    
    # Apply mask to shading
    # shading_high = shading_high * mask_pred_bin_eroded[..., None]
    # shading_high = np.clip(shading_high, 0.05, 1.0) 

    # Prepare Albedo
    img_float = img_orig.astype(np.float32) / 255.0 # RGB 0-1
    
    # Smooth shading slightly to remove remaining model jitter
    # shading_final = cv2.GaussianBlur(shading_high, (15, 15), 0)
    # if shading_final.ndim == 2:
    #     shading_final = shading_final[..., None]
    
    # Calculate Albedo (Broadcasting RGB / Luma)
    albedo = img_float # / shading_final
    albedo = np.clip(albedo, 0, 1)
    
    # 5. Backward Map Generation
    target_h = int(max(h_orig, w_orig))
    target_w = int(target_h / 1.414)
    
    # Filter valid points from prediction
    valid_mask = mask_pred_low > 0.5
    
    y_idx, x_idx = np.where(valid_mask)
    src_y = y_idx / (h - 1) # 0-1
    src_x = x_idx / (w - 1) # 0-1
    
    dst_u = uv_pred_low[valid_mask, 0]
    dst_v = uv_pred_low[valid_mask, 1]
    
    # Subsample Control Points
    total_valid = len(src_x)
    if total_valid > num_points:
        step = total_valid // num_points
        # Uniform sampling via step (as requested)
        src_x = src_x[::step]
        src_y = src_y[::step]
        dst_u = dst_u[::step]
        dst_v = dst_v[::step]
        
    points = np.stack([dst_u, dst_v], axis=1) # Input: Target UV
    values_x = src_x # Output: Source X
    values_y = src_y # Output: Source Y
    
    # Output Grid sizes
    # For speed, we compute on smaller grid then resize
    interp_h, interp_w = 512, int(512 / 1.414)
    
    if method == 'tps':
        print("Solving TPS...")
        tps = TPSSolver(device=device)
        # points: Target(UV), values: Source(XY) -> Fit Target->Source
        # Note: values are 0-1. TPS works fine.
        grid_res = tps.fit_transform(points, np.stack([values_x, values_y], axis=1), (interp_h, interp_w))
        back_map_x_low = grid_res[..., 0]
        back_map_y_low = grid_res[..., 1]
        
    else: # griddata
        grid_u, grid_v = np.meshgrid(
            np.linspace(0, 1, interp_w),
            np.linspace(0, 1, interp_h)
        )
        try:
            back_map_x_low = griddata(points, values_x, (grid_u, grid_v), method='linear', fill_value=-1)
            back_map_y_low = griddata(points, values_y, (grid_u, grid_v), method='linear', fill_value=-1)
        except Exception as e:
            print(f"Interpolation failed: {e}")
            return img_orig, None, None, None, None, None

    # Refinement
    if refine:
        back_map_x_low, back_map_y_low = refine_lattice(back_map_x_low, back_map_y_low, mask_pred_bin, method='gaussian')
    
    # Resize map to full resolution
    back_map_x = cv2.resize(back_map_x_low, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    back_map_y = cv2.resize(back_map_y_low, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert normalized back_map to pixel coordinates
    map_x = (back_map_x * (w_orig - 1)).astype(np.float32)
    map_y = (back_map_y * (h_orig - 1)).astype(np.float32)
    
    # Mask invalid regions (outside hull)
    # For TPS, it extrapolates, so we need to be careful.
    # We use the grid boundary: any u,v outside 0-1 is invalid?
    # No, we are iterating over u,v in 0-1. So all points are "in page".
    # BUT, they might map to outside the original image.
    mask_valid = (back_map_x > -0.05) & (back_map_y > -0.05) & (back_map_x < 1.05) & (back_map_y < 1.05)
    
    # Also clip to valid UV range if griddata produced -1
    if method == 'griddata':
         mask_valid = mask_valid & (back_map_x != -1)
    
    # Remap Original
    dewarped_orig = cv2.remap(img_float, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    dewarped_orig[~mask_valid] = 0
    
    # Remap Albedo
    dewarped_albedo = cv2.remap(albedo, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    dewarped_albedo[~mask_valid] = 0
    
    # --- Visualizations ---
    # UV Visualization (Grid Map)
    uv_pred_high = cv2.resize(uv_pred_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    uv_vis = (img_orig * 0.4).astype(np.uint8)
    
    u_map = uv_pred_high[..., 0]
    v_map = uv_pred_high[..., 1]
    mask_bool = mask_pred_bin > 0.5
    
    num_lines = 20
    thickness_thresh = 0.08 
    
    dist_u = np.abs((u_map * num_lines) - np.round(u_map * num_lines))
    is_u_line = (dist_u < thickness_thresh) & mask_bool
    
    dist_v = np.abs((v_map * num_lines) - np.round(v_map * num_lines))
    is_v_line = (dist_v < thickness_thresh) & mask_bool
    
    uv_vis[is_u_line] = (255, 255, 0)
    uv_vis[is_v_line] = (255, 0, 255)
    uv_vis[is_u_line & is_v_line] = (255, 255, 255) 
    
    # Forward Warp (Raw Output)
    forward_warp = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    u_mapped = (uv_pred_high[..., 0] * (target_w - 1)).astype(np.int32)
    v_mapped = (uv_pred_high[..., 1] * (target_h - 1)).astype(np.int32)
    
    valid_fwd = (mask_pred_bin > 0.5) & \
                (u_mapped >= 0) & (u_mapped < target_w) & \
                (v_mapped >= 0) & (v_mapped < target_h)
                
    ys, xs = np.where(valid_fwd)
    dst_ys = v_mapped[valid_fwd]
    dst_xs = u_mapped[valid_fwd]
    forward_warp[dst_ys, dst_xs] = img_orig[ys, xs]
    
    return img_orig, dewarped_orig, dewarped_orig, dewarped_albedo, uv_vis, forward_warp, map_x, map_y # temporarily removing shading_high

def visualize_results(model_path, uvdoc_dir, doc3d_dir, uvdoc_raw_dir, doc3d_raw_dir, out_dir, num_samples=3, device='cpu', points=2500, method='griddata', refine=False, preprocess=True, steps=1):
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # model = MobileNetV3UNet(num_classes=3) # temporarily removing shading prediction: num classes 4 => 3
    model = FastViTUNet(num_classes=3)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    model.to(device)
    model.eval()
    
    uvdoc_val_names = [p.stem for p in Path(uvdoc_dir).joinpath("val/images").glob("*.jpg")]
    doc3d_val_names = orjson.loads(Path(doc3d_dir).joinpath("split.json").read_text())["val"]

    uvdoc_names_raw = [p.stem for p in Path(uvdoc_raw_dir).glob("*.png")]
    doc3d_names_raw = [p.stem for p in Path(doc3d_raw_dir).glob("*.png")]

    doc3d_val_names = list(set(doc3d_val_names) & set(doc3d_names_raw))
    uvdoc_val_names = list(set(uvdoc_val_names) & set(uvdoc_names_raw))

    doc3d_val_names = [f"{doc3d_raw_dir}/{name}.png" for name in doc3d_val_names]
    uvdoc_val_names = [f"{uvdoc_raw_dir}/{name}.png" for name in uvdoc_val_names]

    val_names_raw = doc3d_val_names + uvdoc_val_names
    
    if not val_names_raw or len(val_names_raw) == 0:
        print("No validation images found.")
        return
    
    selected_names = np.random.choice(val_names_raw, min(len(val_names_raw), num_samples), replace=False)
    
    for name in tqdm(selected_names, desc="Processing images"):
        path = Path(name)
        stem = path.stem
        
        print(f"Processing {path.name}...")
        orig_img = cv2.imread(str(path))
        if orig_img is None: continue
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        current_img = orig_img
        preprocessed_img = None
        
        # --- Schritt 1: Preprocessing ---
        if preprocess:
            print("  Running Axis Alignment Preprocessing...")
            current_img = axis_alignment_preprocessing(model, current_img, device)
            preprocessed_img = current_img.copy()
            
        base_img = preprocessed_img if preprocess else orig_img
        cum_map_x = None
        cum_map_y = None
        
        # --- Schritt 2: Iteratives Unwarping ---
        for step in range(steps):
            if steps > 1:
                print(f"  Running Unwarping Step {step+1}/{steps}...")
                
            _, _, _, _, uv_vis, fwd_warp, map_x, map_y = infer_high_res(
                model, current_img, device, points, method, refine
            )
            
            if map_x is None: break
            
            # Transformationen akkumulieren
            if cum_map_x is None:
                cum_map_x = map_x
                cum_map_y = map_y
            else:
                cum_map_x = cv2.remap(cum_map_x, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                cum_map_y = cv2.remap(cum_map_y, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                
            # Immer aus dem scharfen Basis-Bild samplen
            base_float = base_img.astype(np.float32) / 255.0
            res = cv2.remap(base_float, cum_map_x, cum_map_y, interpolation=cv2.INTER_LINEAR)
            
            # Pixel außerhalb des Dokuments schwärzen
            h_orig, w_orig = base_img.shape[:2]
            mask_valid = (cum_map_x >= 0) & (cum_map_x <= w_orig-1) & (cum_map_y >= 0) & (cum_map_y <= h_orig-1)
            res[~mask_valid] = 0
            
            albedo = res.copy()
            current_img = (res * 255).clip(0, 255).astype(np.uint8)
        
        # --- Visualisierung ---
        if res is not None:
            out_name = f"vis_highres_{stem}.png"
            h = orig_img.shape[0] # Wir richten alles an der Originalhöhe aus
            
            def resize_to_h(img, target_h):
                scale = target_h / img.shape[0]
                new_w = int(img.shape[1] * scale)
                return cv2.resize(img, (new_w, target_h))
            
            res_disp = resize_to_h(res, h)
            albedo_disp = resize_to_h(albedo, h)
            fwd_warp_disp = resize_to_h(fwd_warp, h)
            uv_vis_disp = resize_to_h(uv_vis, h)
            
            def add_label(img, text):
                img = img.astype(np.uint8).copy()
                cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (0, 0, 255), 3, cv2.LINE_AA)
                return img
            
            stack = [add_label(orig_img, "1. Original")]
            
            if preprocess:
                pre_disp = resize_to_h(preprocessed_img, h)
                stack.append(add_label(pre_disp, "1.b Preprocessed"))
                
            stack.extend([
                add_label(uv_vis_disp, "2. UV Grid (Final)"),
                add_label(fwd_warp_disp, "3. Fwd Warp"),
                add_label((res_disp * 255).astype(np.uint8), f"4. Dewarped ({steps} steps)"),
                add_label((albedo_disp * 255).astype(np.uint8), "5. Albedo")
            ])
            
            combined = np.hstack(stack)
            cv2.imwrite(str(out_dir / out_name), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"Saved {out_dir / out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet_best.pth")
    parser.add_argument("--uvdoc_dir", default="dataset_unet")
    parser.add_argument("--doc3d_dir", default="dataset_doc3d")
    parser.add_argument("--uvdoc_raw_dir", default="downloads/UVDoc_final/img")
    parser.add_argument("--doc3d_raw_dir", default="downloads/doc3d_raw/img")
    parser.add_argument("--out", default=".")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--points", type=int, default=2500, help="Number of control points for interpolation")
    parser.add_argument("--method", default="tps", choices=['griddata', 'tps'], help="Interpolation method")
    parser.add_argument("--refine", default=True, action='store_true', help="Enable lattice refinement")
    parser.add_argument("--device", default="cuda", choices=['cuda', 'cpu'], help="Device to run on")
    parser.add_argument("--preprocess", action='store_true', help="Enable Axis Alignment Preprocessing before unwarping")
    parser.add_argument("--steps", type=int, default=1, help="Number of sequential unwarping passes")
    args = parser.parse_args()
    
    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    visualize_results(args.model, args.uvdoc_dir, args.doc3d_dir, args.uvdoc_raw_dir, args.doc3d_raw_dir, args.out, args.n, device, args.points, args.method, args.refine, args.preprocess, args.steps)

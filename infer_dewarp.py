import argparse
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from ultralytics import YOLO


def _to_int_tuple(xy):
    return (int(xy[0]), int(xy[1]))


def _robust_mad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to be comparable to stddev."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def resample_path(path: np.ndarray, n: int) -> np.ndarray:
    """Resample a 2D path to n equidistant points."""
    if len(path) < 2:
        return np.tile(path[0], (n, 1))

    diff = path[1:] - path[:-1]
    dists = np.linalg.norm(diff, axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total = cum_dist[-1]

    if total <= 1e-6:
        return np.tile(path[0], (n, 1))

    target = np.linspace(0, total, n)
    x = np.interp(target, cum_dist, path[:, 0])
    y = np.interp(target, cum_dist, path[:, 1])

    return np.stack([x, y], axis=1).astype(np.float32)


def get_smart_segment(contour, idx_start, idx_end):
    """
    Extract the contour segment from idx_start to idx_end.
    Chooses the direction (forward/backward indices) that corresponds
    to the shorter physical path (assuming the edge is shorter than the rest of the perimeter).
    """
    L = len(contour)
    if idx_start == idx_end:
        return contour[idx_start : idx_start + 1]

    # Normalize indices
    idx_start = idx_start % L
    idx_end = idx_end % L

    # 1. Forward path (following index order)
    if idx_end >= idx_start:
        path_fwd = contour[idx_start : idx_end + 1]
    else:
        path_fwd = np.concatenate((contour[idx_start:], contour[: idx_end + 1]))

    dist_fwd = np.sum(np.linalg.norm(path_fwd[1:] - path_fwd[:-1], axis=1))

    # Calculate total perimeter to compare
    diffs = contour[1:] - contour[:-1]
    total_len = np.sum(np.linalg.norm(diffs, axis=1)) + np.linalg.norm(
        contour[0] - contour[-1]
    )

    # 2. Backward path (reverse index order)
    if idx_start >= idx_end:
        path_rev = contour[idx_end : idx_start + 1][::-1]
    else:
        # wrap around 0
        p1 = contour[0 : idx_start + 1][::-1]
        p2 = contour[idx_end:][::-1]
        path_rev = np.concatenate((p1, p2))
        
    dist_rev = np.sum(np.linalg.norm(path_rev[1:] - path_rev[:-1], axis=1))

    # Pick the shorter one
    if dist_fwd <= dist_rev:
        return path_fwd
    else:
        return path_rev


def snap_grid_to_contour(grid_abs, contour, corners_idxs, n):
    """
    Adjust the boundary of grid_abs to lie on the provided contour.
    corners_idxs: [TL_idx, TR_idx, BR_idx, BL_idx] indices into contour.
    """
    result_grid = grid_abs.copy()

    # 1. Resample segments for each side using identified corners
    # Top: TL -> TR
    seg_top = get_smart_segment(contour, corners_idxs[0], corners_idxs[1])
    pts_top = resample_path(seg_top, n)
    result_grid[0, :, :] = pts_top

    # Right: TR -> BR
    seg_right = get_smart_segment(contour, corners_idxs[1], corners_idxs[2])
    pts_right = resample_path(seg_right, n)
    result_grid[:, -1, :] = pts_right

    # Bottom: BL -> BR (Note: grid row -1 goes BL->BR)
    # So we want path from corners_idxs[3] (BL) to corners_idxs[2] (BR)
    seg_bottom = get_smart_segment(contour, corners_idxs[3], corners_idxs[2])
    pts_bottom = resample_path(seg_bottom, n)
    result_grid[-1, :, :] = pts_bottom

    # Left: TL -> BL (Note: grid col 0 goes TL->BL)
    # Path from corners_idxs[0] (TL) to corners_idxs[3] (BL)
    seg_left = get_smart_segment(contour, corners_idxs[0], corners_idxs[3])
    pts_left = resample_path(seg_left, n)
    result_grid[:, 0, :] = pts_left

    return result_grid


def order_points_quad(pts):
    """
    Sort 4 points into TL, TR, BR, BL order.
    """
    # Sort by y first (top vs bottom)
    pts = pts[np.argsort(pts[:, 1])]
    
    # Top 2 points
    top = pts[:2]
    # Bottom 2 points
    bottom = pts[2:]
    
    # Sort top by x -> TL, TR
    top = top[np.argsort(top[:, 0])]
    tl, tr = top[0], top[1]
    
    # Sort bottom by x -> BL, BR
    bottom = bottom[np.argsort(bottom[:, 0])]
    bl, br = bottom[0], bottom[1]
    
    # Return in order TL, TR, BR, BL
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_document_contour(img_bgr, bbox_xyxy):
    """
    Detect largest contour in the bounding box and identify its 4 corners.
    Returns: (contour_points, corner_indices) or None
    """
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    h, w = img_bgr.shape[:2]
    
    # Slight padding for crop
    pad = 10
    bx1 = max(0, x1 - pad)
    by1 = max(0, y1 - pad)
    bx2 = min(w, x2 + pad)
    by2 = min(h, y2 + pad)

    crop = img_bgr[by1:by2, bx1:bx2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.1 * (crop.shape[0] * crop.shape[1]):
        return None

    # We need to find the 4 corners on this contour.
    # 1. Approximate polygon
    epsilon_start = 0.01
    approx = None
    for factor in range(1, 10):
        ep = epsilon_start * factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, ep, True)
        if len(approx) == 4:
            break
    
    # If we still don't have 4 points, force it by convex hull -> simplification
    if len(approx) != 4:
        hull = cv2.convexHull(cnt)
        # Try to reduce hull to 4 points
        for factor in range(1, 20):
            ep = 0.01 * factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, ep, True)
            if len(approx) <= 4:
                break
                
    # If we have < 4 points (e.g. triangle), it's bad.
    # If we have > 4 points, we just pick the bounding box corners of the contour?
    # Or picking the 4 points with max distance from center?
    
    target_corners = None
    
    if len(approx) == 4:
        target_corners = approx.reshape(4, 2)
    else:
        # Fallback: Find 4 extreme points on the contour
        # (Min sum, Max sum, Min diff, Max diff) approach usually works for rectangles
        # Sum = x+y, Diff = y-x
        pts = cnt.reshape(-1, 2)
        s = pts.sum(axis=1)
        d = pts[:, 1] - pts[:, 0]
        
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        target_corners = np.array([tl, tr, br, bl])

    # Convert local crop coords to global
    offset = np.array([bx1, by1], dtype=np.float32)
    
    # Re-order target_corners to TL, TR, BR, BL geometric order to match grid
    target_corners = order_points_quad(target_corners)
    
    # Now find the indices of these corners in the ORIGINAL full contour (cnt)
    # We want the dense contour for resampling, not the approx one.
    cnt_global = cnt.reshape(-1, 2) + offset
    target_corners_global = target_corners + offset
    
    corner_idxs = []
    for corner in target_corners_global:
        # Find closest point in cnt_global
        dists = np.linalg.norm(cnt_global - corner, axis=1)
        corner_idxs.append(np.argmin(dists))
        
    return cnt_global.astype(np.float32), corner_idxs


def postprocess_grid(
    grid_xy: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    iters: int = 3,
    outlier_sigma: float = 4.0,
    anchor_edges: bool = True,
    enforce_monotonic: bool = True,
    border_info: tuple = None, # (contour, corner_idxs)
) -> np.ndarray:
    """
    Heuristic keypoint grid cleanup.

    Notes/limitations:
    - We cannot "remove" keypoints (YOLO-pose expects fixed K). Instead we *repair* points that look implausible
      by replacing them with a neighbor-based estimate.
    - This assumes the grid is ordered row-major in undistorted space.
    - `anchor_edges` and `enforce_monotonic` work best when the document is roughly axis-aligned in the image.
    """
    n = grid_xy.shape[0]
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Normalize into bbox-local [0,1] for stable thresholds.
    g = (grid_xy - np.array([x1, y1], dtype=np.float32)) / np.array([bw, bh], dtype=np.float32)
    g = g.astype(np.float32)

    # If border contour provided, pre-calculate the target boundary in normalized coords.
    # We do this once to avoid drifting/jitter.
    g_boundary_fixed = None
    if border_info is not None:
        contour, corner_idxs = border_info
        # Denormalize current g to get corners for matching - wait, we don't need to match anymore
        # We just snap to the identified contour corners
        
        # We need the base grid just to copy the inner points initially? 
        # Actually snap_grid_to_contour modifies the boundary of the input grid.
        g_abs = g * np.array([bw, bh], dtype=np.float32) + np.array([x1, y1], dtype=np.float32)
        g_abs_snapped = snap_grid_to_contour(g_abs, contour, corner_idxs, n)
        
        # Re-normalize
        g_snapped = (g_abs_snapped - np.array([x1, y1], dtype=np.float32)) / np.array([bw, bh], dtype=np.float32)
        g_boundary_fixed = g_snapped.copy()

    eps = 1e-5

    for _ in range(max(1, iters)):
        # Neighbor-based expectation
        exp = np.zeros_like(g)
        cnt = np.zeros((n, n, 1), dtype=np.float32)

        # up/down/left/right contributions
        exp[1:, :, :] += g[:-1, :, :]
        cnt[1:, :, :] += 1
        exp[:-1, :, :] += g[1:, :, :]
        cnt[:-1, :, :] += 1
        exp[:, 1:, :] += g[:, :-1, :]
        cnt[:, 1:, :] += 1
        exp[:, :-1, :] += g[:, 1:, :]
        cnt[:, :-1, :] += 1

        exp = exp / np.maximum(cnt, 1.0)

        dist = np.linalg.norm(g - exp, axis=2)  # (n,n)
        sigma = _robust_mad(dist)
        thr = float(np.median(dist) + outlier_sigma * max(sigma, 1e-6))

        mask = dist > thr
        # Repair: replace outliers with expected value
        g[mask] = exp[mask]

        # Optional: light smoothing everywhere (keeps lattice coherent)
        g = 0.85 * g + 0.15 * exp

        # Keep inside a slightly expanded range to avoid catastrophic collapse
        g[:, :, 0] = np.clip(g[:, :, 0], -0.25, 1.25)
        g[:, :, 1] = np.clip(g[:, :, 1], -0.25, 1.25)

        if g_boundary_fixed is not None:
            # Enforce the detected border
            g[0, :, :] = g_boundary_fixed[0, :, :]
            g[-1, :, :] = g_boundary_fixed[-1, :, :]
            g[:, 0, :] = g_boundary_fixed[:, 0, :]
            g[:, -1, :] = g_boundary_fixed[:, -1, :]
        elif anchor_edges:
            # Anchor edges to extrema in the current grid (helps missing/incorrect border points).
            xmin = float(np.min(g[:, :, 0]))
            xmax = float(np.max(g[:, :, 0]))
            ymin = float(np.min(g[:, :, 1]))
            ymax = float(np.max(g[:, :, 1]))

            g[0, :, 1] = ymin
            g[-1, :, 1] = ymax
            g[:, 0, 0] = xmin
            g[:, -1, 0] = xmax

        if enforce_monotonic:
            # Enforce non-decreasing x along rows (left->right) and non-decreasing y along cols (top->bottom).
            # This doesn't reorder points; it nudges them to be consistent.
            for i in range(n):
                row = g[i, :, 0].copy()
                row = np.maximum.accumulate(row)
                # small separation to reduce duplicates
                row = row + np.linspace(0, eps * (n - 1), n, dtype=np.float32)
                g[i, :, 0] = row

            for j in range(n):
                col = g[:, j, 1].copy()
                col = np.maximum.accumulate(col)
                col = col + np.linspace(0, eps * (n - 1), n, dtype=np.float32)
                g[:, j, 1] = col

            # Re-clip after monotonic adjustments
            g[:, :, 0] = np.clip(g[:, :, 0], -0.25, 1.25)
            g[:, :, 1] = np.clip(g[:, :, 1], -0.25, 1.25)

    # Convert back to absolute pixel coords
    g_abs = g * np.array([bw, bh], dtype=np.float32) + np.array([x1, y1], dtype=np.float32)
    return g_abs.astype(np.float32)


def pick_best_detection(result):
    """Pick the highest-confidence detection with keypoints."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.keypoints is None or result.keypoints.xy is None:
        return None

    conf = result.boxes.conf.detach().cpu().numpy()
    best_i = int(np.argmax(conf))

    xyxy = result.boxes.xyxy[best_i].detach().cpu().numpy()  # (4,)
    kpts = result.keypoints.xy[best_i].detach().cpu().numpy()  # (K,2) in pixels

    return {
        "conf": float(conf[best_i]),
        "xyxy": xyxy.astype(np.float32),
        "kpts": kpts.astype(np.float32),
    }


def infer_grid_n(kpts):
    k = int(kpts.shape[0])
    n = int(round(math.sqrt(k)))
    if n * n != k:
        raise ValueError(f"Expected square number of keypoints, got K={k}.")
    return n


def kpts_to_grid(kpts, n):
    """Row-major keypoints -> (n,n,2) grid."""
    return kpts.reshape(n, n, 2)


def identity_grid_from_bbox(xyxy, n):
    x1, y1, x2, y2 = map(float, xyxy)
    xs = np.linspace(x1, x2, n, dtype=np.float32)
    ys = np.linspace(y1, y2, n, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)  # shapes (n,n)
    return np.stack([gx, gy], axis=-1).astype(np.float32)  # (n,n,2)


def bilinear_remap_from_grid(grid_xy, out_w, out_h):
    """
    Build dense remap arrays from an (n,n,2) source-coordinate grid.

    - Output is an undistorted rectangle (out_w x out_h).
    - For each output pixel (u,v), we locate its cell in grid space and bilinearly
      interpolate 4 surrounding grid points to get source (x,y) in the input image.
    """
    n = grid_xy.shape[0]
    assert grid_xy.shape[0] == grid_xy.shape[1]

    # Output pixel coords -> normalized in [0, n-1]
    u = np.linspace(0, n - 1, out_w, dtype=np.float32)
    v = np.linspace(0, n - 1, out_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (out_h, out_w)

    i0 = np.floor(vv).astype(np.int32)
    j0 = np.floor(uu).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, n - 1)
    j1 = np.clip(j0 + 1, 0, n - 1)

    di = (vv - i0).astype(np.float32)
    dj = (uu - j0).astype(np.float32)

    # Gather 4 corners
    p00 = grid_xy[i0, j0]  # (H,W,2)
    p01 = grid_xy[i0, j1]
    p10 = grid_xy[i1, j0]
    p11 = grid_xy[i1, j1]

    # Bilinear interpolation
    w00 = (1 - di) * (1 - dj)
    w01 = (1 - di) * dj
    w10 = di * (1 - dj)
    w11 = di * dj

    map_x = (p00[..., 0] * w00 + p01[..., 0] * w01 + p10[..., 0] * w10 + p11[..., 0] * w11).astype(
        np.float32
    )
    map_y = (p00[..., 1] * w00 + p01[..., 1] * w01 + p10[..., 1] * w10 + p11[..., 1] * w11).astype(
        np.float32
    )

    return map_x, map_y


def draw_lattice(img_bgr, grid_xy, color=(0, 0, 255), thickness=1):
    """Draw lattice lines on a BGR image using grid points (in image pixel coords)."""
    n = grid_xy.shape[0]
    out = img_bgr.copy()

    # Rows
    for i in range(n):
        pts = [
            _to_int_tuple(grid_xy[i, j])
            for j in range(n)
        ]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, a, b, color, thickness, lineType=cv2.LINE_AA)

    # Cols
    for j in range(n):
        pts = [
            _to_int_tuple(grid_xy[i, j])
            for i in range(n)
        ]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, a, b, color, thickness, lineType=cv2.LINE_AA)

    return out


def main():
    ap = argparse.ArgumentParser(description="Infer UVDoc lattice and visualize dewarp animation")
    ap.add_argument("--model", type=str, default="runs/pose/uvdoc_yolo11s_midstrong/weights/best.pt", help="Path to YOLO pose weights")
    ap.add_argument("--img", type=str, required=True, help="Input image path")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference imgsz (Ultralytics handles resizing internally)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--frames", type=int, default=25, help="Number of frames in GIF")
    ap.add_argument("--out_gif", type=str, default="dewarp.gif", help="Output GIF path")
    ap.add_argument("--out_png", type=str, default="dewarp.png", help="Output final dewarped PNG path")
    ap.add_argument("--show", action="store_true", help="Show interactive windows")
    ap.add_argument("--pad", type=int, default=0, help="Extra padding around bbox (pixels) for output crop")
    ap.add_argument("--post", action="store_true", help="Enable heuristic keypoint post-processing (repairs outliers)")
    ap.add_argument("--post_iters", type=int, default=3, help="Post-processing iterations")
    ap.add_argument("--post_sigma", type=float, default=4.0, help="Outlier threshold in MAD-sigma units")
    ap.add_argument("--no_anchor", action="store_true", help="Disable edge anchoring during post-processing")
    ap.add_argument("--no_mono", action="store_true", help="Disable monotonicity constraints during post-processing")
    ap.add_argument("--no_border_detect", action="store_true", help="Disable document border detection for lattice anchoring")
    args = ap.parse_args()

    model_path = Path(args.model)
    img_path = Path(args.img)
    out_png_path = Path(args.out_png)
    out_gif_path = Path(args.out_gif)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    model = YOLO(str(model_path))

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # Predict
    pred = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
    if not pred:
        raise RuntimeError("No prediction returned.")

    det = pick_best_detection(pred[0])
    if det is None:
        raise RuntimeError("No valid detection/keypoints found.")

    x1, y1, x2, y2 = det["xyxy"]
    x1 = max(0, x1 - args.pad)
    y1 = max(0, y1 - args.pad)
    x2 = min(img_bgr.shape[1] - 1, x2 + args.pad)
    y2 = min(img_bgr.shape[0] - 1, y2 + args.pad)
    xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

    kpts = det["kpts"]
    n = infer_grid_n(kpts)
    grid_pred = kpts_to_grid(kpts, n)

    grid_used = grid_pred
    if args.post:
        # Detect border if requested
        border_info = None
        if not args.no_border_detect:
            border_info = get_document_contour(img_bgr, xyxy)
            if border_info is not None:
                print("Document border detected.")
            else:
                print("Document border not detected, falling back to grid extrema.")
        
        grid_used = postprocess_grid(
            grid_pred,
            xyxy,
            iters=args.post_iters,
            outlier_sigma=args.post_sigma,
            anchor_edges=(not args.no_anchor),
            enforce_monotonic=(not args.no_mono),
            border_info=border_info,
        )

    # "Undistorted coordinates within border": normalize keypoints into bbox-local [0,1]
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    kpts_bbox_norm = (kpts - np.array([x1, y1], dtype=np.float32)) / np.array([bw, bh], dtype=np.float32)

    # Build identity grid (no warp) that maps output lattice directly to bbox rectangle in the input
    grid_id = identity_grid_from_bbox(xyxy, n)

    # Output resolution: use bbox size (rounded)
    out_w = int(max(2, round(bw)))
    out_h = int(max(2, round(bh)))

    # Create a canvas with the *distorted* lattice drawn in input-image space.
    # We'll remap this canvas with the same warp as the image so the lattice itself dewarps over time.
    lattice_canvas = np.zeros_like(img_bgr)
    lattice_canvas = draw_lattice(lattice_canvas, grid_used, color=(0, 255, 0), thickness=2)

    # Animation: interpolate lattice from identity -> predicted
    frames = []
    for t in np.linspace(0.0, 1.0, max(2, args.frames), dtype=np.float32):
        grid_t = (1.0 - t) * grid_id + t * grid_used
        map_x, map_y = bilinear_remap_from_grid(grid_t, out_w=out_w, out_h=out_h)
        unwarped = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Warp the *distorted lattice* with the same remap so it gradually straightens as t increases.
        lattice_warped = cv2.remap(
            lattice_canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # Overlay the warped lattice on the (partially) dewarped image.
        # Use a mask to avoid darkening the image (since lattice_warped has black background).
        mask = (lattice_warped.sum(axis=2) > 0)[:, :, None]
        unwarped_vis = unwarped.copy()
        unwarped_vis[mask.squeeze()] = lattice_warped[mask.squeeze()]

        frames.append(cv2.cvtColor(unwarped_vis, cv2.COLOR_BGR2RGB))

    # Save outputs
    imageio.mimsave(str(out_gif_path), frames, duration=0.06)

    final_rgb = frames[-1]
    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_png_path), final_bgr)

    # Also save an overlay on the original image showing bbox + predicted lattice
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, _to_int_tuple((x1, y1)), _to_int_tuple((x2, y2)), (0, 255, 0), 2)
    overlay_raw = draw_lattice(overlay, grid_pred, color=(0, 0, 255), thickness=1)
    cv2.imwrite(str(out_png_path.with_suffix(".lattice.png")), overlay_raw)
    if args.post:
        overlay_pp = draw_lattice(overlay, grid_used, color=(0, 255, 255), thickness=1)
        cv2.imwrite(str(out_png_path.with_suffix(".lattice_post.png")), overlay_pp)

    print(f"Best detection conf: {det['conf']:.3f}")
    print(f"Grid N: {n} (K={n*n})")
    print(f"Saved: {out_gif_path}")
    print(f"Saved: {out_png_path}")
    print(f"Saved: {out_png_path.with_suffix('.lattice.png')} (bbox + raw lattice)")
    if args.post:
        print(f"Saved: {out_png_path.with_suffix('.lattice_post.png')} (bbox + post-processed lattice)")
    print("Example bbox-normalized keypoints (first 5):")
    print(kpts_bbox_norm[:5])

    if args.show:
        cv2.imshow("Original + Raw Lattice", overlay_raw)
        if args.post:
            cv2.imshow("Original + Post-Processed Lattice", overlay_pp)
        cv2.imshow("Final Dewarp", final_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

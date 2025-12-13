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


def postprocess_grid(
    grid_xy: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    iters: int = 3,
    outlier_sigma: float = 4.0,
    anchor_edges: bool = True,
    enforce_monotonic: bool = True,
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

        if anchor_edges:
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
    ap.add_argument("--model", type=str, default="runs/pose/uvdoc_yolo11n/weights/best.pt", help="Path to YOLO pose weights")
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
        grid_used = postprocess_grid(
            grid_pred,
            xyxy,
            iters=args.post_iters,
            outlier_sigma=args.post_sigma,
            anchor_edges=(not args.no_anchor),
            enforce_monotonic=(not args.no_mono),
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

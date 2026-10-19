"""
High-resolution dewarping and visualization with the backward-map model.

  python vis_unet.py --model unet_bm_best.pth --n 5                 # random validation samples
  python vis_unet.py --model unet_bm_best.pth --images a.jpg b.jpg  # own photos
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import orjson
import torch
from tqdm import tqdm

from backward_map import bm_outline, bm_to_remap
from models import load_fastvit_unet

INPUT_SIZE = 256
A4_RATIO = 1.414


@torch.no_grad()
def predict_bm(model, img_rgb: np.ndarray, device) -> np.ndarray:
    """RGB uint8 image of any size -> backward map (2, GH, GW)."""
    img = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return model(x)[0].float().cpu().numpy()


def axis_alignment_preprocessing(model, img_orig, device, expand_ratio=0.05):
    """Coarse crop + rotation of the document via the min-area rectangle of the predicted page outline."""
    h_orig, w_orig = img_orig.shape[:2]
    outline = bm_outline(predict_bm(model, img_orig, device), w_orig, h_orig)
    if cv2.contourArea(outline) < 100:
        return img_orig

    box = cv2.boxPoints(cv2.minAreaRect(outline))

    # Sort: top-left, top-right, bottom-right, bottom-left
    rect_pts = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    rect_pts[0] = box[np.argmin(s)]
    rect_pts[2] = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    rect_pts[1] = box[np.argmin(diff)]
    rect_pts[3] = box[np.argmax(diff)]

    (tl, tr, br, bl) = rect_pts
    max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    max_h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    # Padding so no page corners get cut off
    pad_x, pad_y = int(max_w * expand_ratio), int(max_h * expand_ratio)
    dst = np.array([
        [pad_x, pad_y],
        [max_w + pad_x - 1, pad_y],
        [max_w + pad_x - 1, max_h + pad_y - 1],
        [pad_x, max_h + pad_y - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_pts, dst)
    return cv2.warpPerspective(img_orig, M, (max_w + 2 * pad_x, max_h + 2 * pad_y))


def output_size(h: int, w: int):
    """Portrait A4 output at the resolution of the longer input side."""
    out_h = int(max(h, w))
    return int(out_h / A4_RATIO), out_h


def draw_grid(img_rgb: np.ndarray, bm: np.ndarray, every: int = 2) -> np.ndarray:
    """Overlays the predicted page grid (rows magenta, columns cyan) on the input image."""
    h, w = img_rgb.shape[:2]
    pts = np.stack([(bm[0] + 1) * 0.5 * (w - 1), (bm[1] + 1) * 0.5 * (h - 1)], -1).astype(np.int32)
    vis = (img_rgb * 0.5).astype(np.uint8)
    thickness = max(1, w // 400)
    for i in range(0, pts.shape[0], every):
        cv2.polylines(vis, [np.ascontiguousarray(pts[i])], False, (255, 0, 255), thickness, cv2.LINE_AA)
    for j in range(0, pts.shape[1], every):
        cv2.polylines(vis, [np.ascontiguousarray(pts[:, j])], False, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.polylines(vis, [bm_outline(bm, w, h).astype(np.int32)], True, (255, 255, 255), thickness * 2, cv2.LINE_AA)
    return vis


def dewarp_image(model, img_rgb: np.ndarray, device, steps: int = 1):
    """
    Iterative dewarping: each step predicts on the previous result, the remap maps are composed
    so the final image is always sampled once from the sharp input.
    Returns (dewarped RGB uint8, backward map of the first step).
    """
    h, w = img_rgb.shape[:2]
    current, cum_x, cum_y, first_bm = img_rgb, None, None, None
    for _ in range(steps):
        bm = predict_bm(model, current, device)
        first_bm = bm if first_bm is None else first_bm
        out_w, out_h = output_size(*current.shape[:2])
        map_x, map_y = bm_to_remap(bm, out_w, out_h, current.shape[1], current.shape[0])
        if cum_x is None:
            cum_x, cum_y = map_x, map_y
        else:
            cum_x = cv2.remap(cum_x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cum_y = cv2.remap(cum_y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        current = cv2.remap(img_rgb, cum_x, cum_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return current, first_bm


def resize_to_h(img, target_h):
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h))


def add_label(img, text):
    img = img.astype(np.uint8).copy()
    scale = max(0.6, img.shape[0] / 700)
    cv2.putText(img, text, (10, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 0, 0), max(1, int(2 * scale)), cv2.LINE_AA)
    return img


def validation_images(uvdoc_dir: Path, doc3d_dir: Path):
    paths = sorted((uvdoc_dir / "val" / "images").glob("*.jpg"))
    split_file = doc3d_dir / "split.json"
    if split_file.exists():
        paths += [doc3d_dir / "img" / f"{i}.png" for i in orjson.loads(split_file.read_bytes())["val"]]
    return paths


def visualize(model, paths, out_dir: Path, device, preprocess=False, steps=1):
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths, desc="Processing images"):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        base = axis_alignment_preprocessing(model, orig, device) if preprocess else orig

        dewarped, bm = dewarp_image(model, base, device, steps)

        h = orig.shape[0]
        stack = [add_label(orig, "1. Original")]
        if preprocess:
            stack.append(add_label(resize_to_h(base, h), "1b. Preprocessed"))
        stack += [
            add_label(resize_to_h(draw_grid(base, bm), h), "2. Predicted grid"),
            add_label(resize_to_h(dewarped, h), f"3. Dewarped ({steps} step{'s' if steps > 1 else ''})"),
        ]
        out_path = out_dir / f"vis_highres_{Path(path).stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(np.hstack(stack), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet_bm_best.pth")
    parser.add_argument("--images", nargs="*", help="Own images; default: random validation samples")
    parser.add_argument("--uvdoc_dir", default="dataset_unet")
    parser.add_argument("--doc3d_dir", default="dataset_doc3d")
    parser.add_argument("--out", default="vis_out")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--device", default="cuda", choices=['cuda', 'cpu'])
    parser.add_argument("--preprocess", action='store_true', help="Axis alignment preprocessing before unwarping")
    parser.add_argument("--steps", type=int, default=1, help="Number of sequential unwarping passes")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_fastvit_unet(args.model, backward_map=True, device=device)

    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        candidates = validation_images(Path(args.uvdoc_dir), Path(args.doc3d_dir))
        rng = np.random.default_rng()
        paths = list(rng.choice(candidates, min(args.n, len(candidates)), replace=False)) if candidates else []
    visualize(model, paths, Path(args.out), device, args.preprocess, args.steps)

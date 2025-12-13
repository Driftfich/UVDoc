import argparse
import random
import yaml
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_label(label_path):
    """
    Parses a YOLO-pose label file.
    Returns a list of dictionaries, one for each object.
    """
    objects = []
    if not label_path.exists():
        return objects
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls_id = int(parts[0])
        cx, cy, w, h = parts[1:5]
        keypoints = parts[5:]
        
        objects.append({
            'class': cls_id,
            'bbox': (cx, cy, w, h),
            'keypoints': keypoints
        })
    return objects

def draw_on_image(img, objects, kpt_shape):
    """
    Draws bounding boxes and keypoints on the image.
    """
    h, w = img.shape[:2]
    img_draw = img.copy()
    
    num_kpts = kpt_shape[0]
    dim = kpt_shape[1] # 2 or 3
    
    for obj in objects:
        # Draw BBox
        cx, cy, bw, bh = obj['bbox']
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Keypoints
        kpts = obj['keypoints']
        for i in range(num_kpts):
            idx = i * dim
            kx = kpts[idx]
            ky = kpts[idx+1]
            conf = 1.0
            if dim == 3:
                conf = kpts[idx+2] # visibility
            
            # If normalized coordinates are used (which is standard for YOLO)
            px = int(kx * w)
            py = int(ky * h)
            
            if conf > 0: # Only draw visible points
                # Use a color map or just red
                cv2.circle(img_draw, (px, py), 3, (0, 0, 255), -1)
                
    return img_draw

def visualize_samples(dataset_path, subset, num_samples, kpt_shape):
    dataset_root = Path(dataset_path)
    yaml_path = dataset_root / "dataset.yaml"
    
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found.")
        return []

    config = load_yaml(yaml_path)
    
    # Resolve image directory
    # 'train' in yaml is typically a relative path like 'images/train'
    img_rel_path = config.get(subset)
    if not img_rel_path:
        print(f"Subset '{subset}' not found in yaml.")
        return []
        
    img_dir = dataset_root / img_rel_path
    
    # Assuming labels are in a parallel directory structure:
    # images/train -> labels/train
    # This is standard YOLO structure but the yaml doesn't explicitly point to labels usually
    # It assumes images/.. and labels/.. relative to root or parent.
    # Based on our build script: dataset_yolo/images/train and dataset_yolo/labels/train
    
    # Construct label dir by replacing 'images' with 'labels' in the path parts
    # Robust way:
    parts = list(img_dir.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_dir = Path(*parts)
    except ValueError:
        # Fallback if structure is different
        label_dir = dataset_root / "labels" / subset
    
    if not img_dir.exists():
        print(f"Image directory {img_dir} does not exist.")
        return []
        
    all_images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    
    if not all_images:
        print(f"No images found in {img_dir}")
        return []
        
    selected = random.sample(all_images, min(num_samples, len(all_images)))
    
    results = []
    for img_path in selected:
        label_path = label_dir / (img_path.stem + ".txt")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        objects = parse_label(label_path)
        img_vis = draw_on_image(img, objects, kpt_shape)
        results.append((img_path.name, img_vis))
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Visualize UVDoc YOLO Dataset")
    parser.add_argument("--dataset", type=str, default="dataset_yolo", help="Path to dataset root")
    parser.add_argument("--n_train", type=int, default=2, help="Number of training samples")
    parser.add_argument("--m_val", type=int, default=2, help="Number of validation samples")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    yaml_path = dataset_path / "dataset.yaml"
    if not yaml_path.exists():
        print("dataset.yaml not found.")
        return

    config = load_yaml(yaml_path)
    kpt_shape = config.get('kpt_shape', [100, 3])
    
    print(f"Visualizing {args.n_train} train images and {args.m_val} val images...")
    
    train_imgs = visualize_samples(dataset_path, 'train', args.n_train, kpt_shape)
    val_imgs = visualize_samples(dataset_path, 'val', args.m_val, kpt_shape)
    
    total_imgs = len(train_imgs) + len(val_imgs)
    if total_imgs == 0:
        print("No images to visualize.")
        return

    cols = 2
    rows = (total_imgs + 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    idx = 1
    for name, img in train_imgs:
        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title(f"Train: {name}")
        plt.axis('off')
        idx += 1
        
    for name, img in val_imgs:
        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title(f"Val: {name}")
        plt.axis('off')
        idx += 1
        
    plt.tight_layout()
    output_file = "visualization_result.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()

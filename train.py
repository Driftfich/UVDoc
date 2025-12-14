from ultralytics import YOLO
import argparse
from pathlib import Path

def train(args):
    # 1. Load the model
    # "yolo11n-pose.pt" will be downloaded automatically if not found
    print(f"Loading model: {args.model}")
    model = YOLO(args.model) 

    # 2. Define dataset path
    # Ensure it's absolute or correctly relative
    dataset_yaml = Path(args.data).absolute()
    
    if not dataset_yaml.exists():
        print(f"Error: Dataset config not found at {dataset_yaml}")
        return

    print(f"Starting training on {dataset_yaml}...")
    
    # 3. Train
    # Ultralytics handles resizing (letterboxing/mosaic) automatically.
    # We explicitly control augmentation to avoid cutting off document corners.
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience, # Early stopping
        save=True,
        plots=True,
        rect=True,                 # Rectangular training (minimizes padding, optimizes resolution)
        # Augmentation settings
        mosaic=args.mosaic,        # Collage of 4 images (can crop edges)
        mixup=args.mixup,          # Mixup coefficient
        degrees=args.degrees,      # Rotation (+/- deg)
        translate=args.translate,  # Translation (+/- fraction)
        scale=args.scale,          # Scale (+/- gain)
        shear=args.shear,          # Shear (+/- degrees)
        perspective=args.perspective, # Perspective (+/- fraction)
        erasing=args.erasing,      # Erasing probability
        copy_paste=args.copy_paste,# Copy-paste probability
        hsv_h=args.hsv_h,          # HSV-Hue
        hsv_s=args.hsv_s,          # HSV-Saturation
        hsv_v=args.hsv_v,          # HSV-Value
        fliplr=0.0,                # Disable LR flip for text documents (keeps lattice orientation consistent)
        flipud=0.0,
    )
    
    print("Training complete.")
    print(f"Results saved to {results.save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11-Pose on UVDoc")
    
    parser.add_argument("--model", type=str, default="yolo11s-pose.pt", help="Pretrained model checkpoint")
    parser.add_argument("--data", type=str, default="dataset_yolo/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default 736 to cover 712px height)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (adjust based on VRAM)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", type=str, default="runs/pose", help="Project name for saving results")
    parser.add_argument("--name", type=str, default="uvdoc_yolo11s_midstrong", help="Experiment name")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--perspective", type=float, default=0.0004, help="Perspective augmentation probability")
    parser.add_argument("--erasing", type=float, default=0.3, help="Erasing augmentation probability")
    parser.add_argument("--copy_paste", type=float, default=0.01, help="Copy-paste augmentation probability")
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV-H augmentation (hue)")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV-S augmentation (saturation)")
    parser.add_argument("--hsv_v", type=float, default=0.2, help="HSV-V augmentation (brightness)")
    parser.add_argument("--shear", type=float, default=3.0, help="Shear augmentation (degrees)")
    
    # Augmentation defaults
    # Mosaic: 0.5 (50% chance) - balanced mix
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic augmentation probability")
    # Mixup: Default 0.0.
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation probability")
    # Degrees: 25.0
    parser.add_argument("--degrees", type=float, default=40.0, help="Rotation (+/- degrees)")
    # Translate: 0.15
    parser.add_argument("--translate", type=float, default=0.15, help="Translation (+/- fraction)")
    # Scale: 0.4
    parser.add_argument("--scale", type=float, default=0.4, help="Scale gain (+/- gain)")

    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()

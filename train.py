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
        plots=True
    )
    
    print("Training complete.")
    print(f"Results saved to {results.save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11-Pose on UVDoc")
    
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="Pretrained model checkpoint")
    parser.add_argument("--data", type=str, default="dataset_yolo/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (adjust based on VRAM)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", type=str, default="runs/pose", help="Project name for saving results")
    parser.add_argument("--name", type=str, default="uvdoc_yolo11n", help="Experiment name")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()

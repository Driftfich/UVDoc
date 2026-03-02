import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
import torch
import tensorflow as tf
from train_unet import MobileNetV3UNet, FastViTUNet
from mobileone import reparameterize_model

def check_dependencies():
    """Checks if required tools are installed."""
    missing = []
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    
    # Check for onnx2tf command
    if shutil.which("onnx2tf") is None:
        missing.append("onnx2tf")
        
    if missing:
        print("Missing dependencies for export:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease install them using:")
        print("pip install onnx onnx2tf tensorflow")
        sys.exit(1)

def export_to_onnx(model, output_path, input_shape):
    """Exports PyTorch model to ONNX."""
    print(f"Exporting PyTorch model to ONNX: {output_path}...")
    
    # Dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13, # 13 is usually safe for TFLite
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None # Use fixed batch size for better TFLite compatibility
    )
    print("ONNX export complete.")

def convert_onnx_to_tf(onnx_path, tf_output_dir):
    """Converts ONNX to TensorFlow SavedModel using onnx2tf."""
    print(f"Converting ONNX to TensorFlow SavedModel: {tf_output_dir}...")
    
    # onnx2tf command
    # -i: Input ONNX
    # -o: Output SavedModel directory
    # -v: Verbose (optional)
    cmd = [
        "onnx2tf",
        "-i", onnx_path,
        "-o", tf_output_dir,
        "--copy_onnx_input_output_names_to_tflite",
        "--non_verbose" 
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("TensorFlow SavedModel conversion complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error converting to TensorFlow: {e}")
        sys.exit(1)

def convert_tf_to_tflite(tf_model_dir, output_path, quantization):
    """Converts TensorFlow SavedModel to TFLite with specified quantization."""
    print(f"Converting SavedModel to TFLite (Quantization: {quantization})...")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    
    # Enable Flex ops (TF Select) as some complex ops or dynamic shapes
    # might not be supported natively by TFLite.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TF ops.
    ]
    
    if quantization == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8_dynamic':
        # Dynamic range quantization (8-bit weights, float32 activations during inference)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == 'fp32':
        # No quantization
        pass
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to: {output_path}")
    print(f"Size: {len(tflite_model) / (1024*1024):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Export UVDoc UNet to TFLite")
    parser.add_argument("--model_path", default="unet_best.pth", help="Path to PyTorch .pth model")
    parser.add_argument("--output_dir", default="export_output", help="Directory for exported files")
    parser.add_argument("--quant", default="fp16", choices=['fp16', 'fp32', 'int8_dynamic'], 
                        help="Quantization type: fp16 (default), fp32 (none), int8_dynamic (smaller but slower on some HW)")
    
    args = parser.parse_args()
    
    check_dependencies()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    tf_saved_model_dir = os.path.join(args.output_dir, "saved_model")
    tflite_path = os.path.join(args.output_dir, f"model_{args.quant}.tflite")
    
    # 1. Load PyTorch Model
    print("Loading PyTorch model...")
    device = torch.device('cpu')
    model = FastViTUNet(num_classes=3)
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model path {args.model_path} not found. Using random weights for testing.")
    
    model.eval()

    # FastViT Reparameterization
    print("Reparameterizing FastViT...")
    model = reparameterize_model(model)
    print("FastViT reparameterized.")
    
    # Input shape: (Batch, Channels, Height, Width)
    # 5 channels: R, G, B, X_coord, Y_coord
    input_shape = (1, 5, 256, 256)
    
    # 2. Export to ONNX
    export_to_onnx(model, onnx_path, input_shape)
    
    # 3. Convert to TF SavedModel (using onnx2tf for best NHWC support)
    convert_onnx_to_tf(onnx_path, tf_saved_model_dir)
    
    # 4. Convert to TFLite
    convert_tf_to_tflite(tf_saved_model_dir, tflite_path, args.quant)
    
    print("\nExport process finished successfully.")
    print(f"Files located in: {args.output_dir}")

if __name__ == "__main__":
    main()


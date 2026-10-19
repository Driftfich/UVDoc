"""
Exports the backward-map FastViTUNet to TFLite.

Input : (1, 256, 256, 5) float32 NHWC after onnx2tf: R, G, B in [0, 1], x, y CoordConv in [-1, 1]
Output: backward map, 2 x 32 x 32 (layout printed after conversion, the app expects NHWC (1, 32, 32, 2))
"""
import argparse
import copy
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import tensorflow as tf

from mobileone import reparameterize_model
from models import coord_grid, load_fastvit_unet


def check_dependencies():
    """Checks if required tools are installed."""
    missing = []
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")
    if shutil.which("onnx2tf") is None:
        missing.append("onnx2tf")
    if missing:
        print("Missing dependencies for export: " + ", ".join(missing))
        print("Install them with: pip install -r requirements_export.txt")
        sys.exit(1)


def use_tanh_gelu(model: torch.nn.Module) -> torch.nn.Module:
    """Exact GELU exports as Erf, which TFLite has no builtin for; the tanh approximation is builtin-only."""
    for module in model.modules():
        if isinstance(module, torch.nn.GELU):
            module.approximate = 'tanh'
    return model


def export_to_onnx(model, output_path, input_shape):
    print(f"Exporting PyTorch model to ONNX: {output_path}...")
    torch.onnx.export(
        model,
        torch.randn(input_shape),
        output_path,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # fixed shapes for TFLite / GPU delegate
    )


def convert_onnx_to_tf(onnx_path, tf_output_dir):
    """Converts ONNX to a TensorFlow SavedModel using onnx2tf (NCHW -> NHWC)."""
    print(f"Converting ONNX to TensorFlow SavedModel: {tf_output_dir}...")
    cmd = ["onnx2tf", "-i", onnx_path, "-o", tf_output_dir, "--copy_onnx_input_output_names_to_tflite", "--non_verbose"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting to TensorFlow: {e}")
        sys.exit(1)


def convert_tf_to_tflite(tf_model_dir, output_path, quantization, allow_flex=False):
    """
    Builtin ops only by default: Flex (SELECT_TF_OPS) ops run on the CPU only, block the
    GPU / NNAPI delegates and need the large Flex runtime in the app.
    """
    print(f"Converting SavedModel to TFLite (quantization: {quantization}, flex: {allow_flex})...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if allow_flex:
        ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
    converter.target_spec.supported_ops = ops

    if quantization == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8_dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization != 'fp32':
        raise ValueError(f"Unknown quantization type: {quantization}")

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Conversion with builtin ops failed: {e}")
        print("Retry with --allow_flex if the model really needs TF ops (disables GPU delegate support).")
        sys.exit(1)

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {output_path} ({len(tflite_model) / 2**20:.2f} MB)")


def verify_tflite(tflite_path, model, input_shape):
    """
    Runs TFLite and the original PyTorch model on the same input and reports the output layout and error
    (includes conversion, GELU approximation and quantization effects).
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    print(f"TFLite input {inp['shape']} {inp['dtype'].__name__} | output {out['shape']} {out['dtype'].__name__}")

    b, _, h, w = input_shape
    rgb = torch.rand(b, 3, h, w)
    x = torch.cat([rgb, coord_grid(b, h, w, 'cpu')], dim=1)
    with torch.no_grad():
        ref = model(x).numpy()  # (1, 2, 32, 32)

    feed = x.numpy().transpose(0, 2, 3, 1) if list(inp['shape']) == [b, h, w, 5] else x.numpy()
    interpreter.set_tensor(inp['index'], feed.astype(np.float32))
    interpreter.invoke()
    res = interpreter.get_tensor(out['index'])
    if res.shape[-1] == 2:
        print("Output layout: NHWC (1, GH, GW, 2) - matches ImageProcessor.kt")
        res = res.transpose(0, 3, 1, 2)
    else:
        print("WARNING: output layout is NCHW, ImageProcessor.kt expects NHWC")
    err = np.abs(res - ref) * 127.5
    print(f"TFLite vs PyTorch grid difference: mean {err.mean():.3f}px | max {err.max():.3f}px (256px input)")


def main():
    parser = argparse.ArgumentParser(description="Export the backward-map UNet to TFLite")
    parser.add_argument("--model_path", default="unet_bm_best.pth", help="Path to PyTorch .pth model")
    parser.add_argument("--output_dir", default="export_output", help="Directory for exported files")
    parser.add_argument("--quant", default="fp16", choices=['fp16', 'fp32', 'int8_dynamic'],
                        help="fp16 (default, GPU delegate friendly), fp32, int8_dynamic (smaller, slower on some HW)")
    parser.add_argument("--allow_flex", action="store_true", help="Allow TF Select ops (CPU only)")
    args = parser.parse_args()

    check_dependencies()
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    tf_saved_model_dir = os.path.join(args.output_dir, "saved_model")
    tflite_path = os.path.join(args.output_dir, f"model_{args.quant}.tflite")

    print("Loading PyTorch model...")
    model = load_fastvit_unet(args.model_path, backward_map=True, device='cpu')
    model = reparameterize_model(model).eval()
    export_model = use_tanh_gelu(copy.deepcopy(model)).eval()

    # 5 channels: R, G, B, X_coord, Y_coord (the app provides the CoordConv channels)
    input_shape = (1, 5, 256, 256)
    export_to_onnx(export_model, onnx_path, input_shape)
    convert_onnx_to_tf(onnx_path, tf_saved_model_dir)
    convert_tf_to_tflite(tf_saved_model_dir, tflite_path, args.quant, args.allow_flex)
    verify_tflite(tflite_path, model, input_shape)

    print(f"\nDone. Copy {tflite_path} to testapp/app/src/main/assets/model.tflite")


if __name__ == "__main__":
    main()

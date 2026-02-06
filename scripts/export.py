#!/usr/bin/env python
"""Export trained AOI model to ONNX format.

This script exports a trained PaDiM or PatchCore model to ONNX format for
deployment. The exported model includes:
- Input: images normalized to [0, 1] range
- Output: anomaly_map (B, 1, H, W) and pred_score (B,)

Usage:
    python scripts/export.py --run_dir runs/padim/transistor/smoke_padim2
    python scripts/export.py --run_dir runs/patchcore/transistor/smoke_patchcore2 --opset 14
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from aoi.config import load_yaml
from aoi.models import OnnxExportWrapper, build_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained AOI model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the training run directory containing config.yaml and weights/",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (18+ recommended for full compatibility)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="model.onnx",
        help="Output ONNX model filename",
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        default=True,
        help="Enable dynamic batch size in exported model",
    )
    parser.add_argument(
        "--no_dynamic_batch",
        action="store_false",
        dest="dynamic_batch",
        help="Use fixed batch size of 1",
    )
    return parser.parse_args()


def export_model(
    run_dir: Path,
    opset_version: int = 18,
    output_name: str = "model.onnx",
    dynamic_batch: bool = True,
) -> Path:
    """Export model to ONNX format.

    Args:
        run_dir: Path to the training run directory.
        opset_version: ONNX opset version.
        output_name: Output filename for the ONNX model.
        dynamic_batch: Whether to enable dynamic batch size.

    Returns:
        Path to the exported ONNX model.

    Raises:
        FileNotFoundError: If required files are not found.
        RuntimeError: If export fails.
    """
    # Validate run directory structure
    config_path = run_dir / "config.yaml"
    weights_dir = run_dir / "weights"
    weights_path = weights_dir / "model.ckpt"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Load configuration
    print(f"Loading config from: {config_path}")
    cfg = load_yaml(config_path)

    # Get model info
    model_name = cfg.get("model", {}).get("name", "unknown")
    pp_cfg = cfg.get("preprocessing", {})
    image_size = pp_cfg.get("image_size", [256, 256])
    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    print(f"Model: {model_name}")
    print(f"Input size: {image_size}")

    # Build model
    print("Building model...")
    model = build_model(cfg)

    # Load checkpoint
    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Extract state dict (handle different checkpoint formats)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Wrap model for ONNX export
    print("Wrapping model for ONNX export...")
    export_model_wrapper = OnnxExportWrapper(model)
    export_model_wrapper.eval()

    # Create dummy input
    # Shape: (batch, channels, height, width)
    batch_size = 1
    dummy_input = torch.rand(batch_size, 3, image_size[0], image_size[1])

    # Verify forward pass works
    print("Verifying forward pass...")
    with torch.no_grad():
        anomaly_map, pred_score = export_model_wrapper(dummy_input)
        print(f"  anomaly_map shape: {anomaly_map.shape}")
        print(f"  pred_score shape: {pred_score.shape}")

    # Prepare export directory
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / output_name

    # Configure dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "anomaly_map": {0: "batch"},
            "pred_score": {0: "batch"},
        }

    # Export to ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        export_model_wrapper,
        (dummy_input,),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["anomaly_map", "pred_score"],
        dynamic_axes=dynamic_axes,
    )

    print(f"ONNX model saved to: {onnx_path}")

    # Verify ONNX model
    try:
        import onnx

        print("Verifying ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except ImportError:
        print("Warning: onnx package not installed, skipping model verification")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")

    # Save export metadata
    meta_path = export_dir / "export_meta.json"
    meta = {
        "model_name": model_name,
        "opset_version": opset_version,
        "input_shape": [batch_size, 3, image_size[0], image_size[1]],
        "output_names": ["anomaly_map", "pred_score"],
        "output_shapes": {
            "anomaly_map": [batch_size, 1, image_size[0], image_size[1]],
            "pred_score": [batch_size],
        },
        "dynamic_batch": dynamic_batch,
        "preprocessing": {
            "input_range": "[0, 1]",
            "channel_order": "RGB",
            "resize_interpolation": "bilinear",
            "image_size": image_size,
            "note": "ImageNet normalization is included in the model",
        },
        "source": {
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "weights_path": str(weights_path),
        },
        "created_at": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Export metadata saved to: {meta_path}")

    return onnx_path


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        onnx_path = export_model(
            run_dir=args.run_dir,
            opset_version=args.opset,
            output_name=args.output_name,
            dynamic_batch=args.dynamic_batch,
        )
        print("\nExport completed successfully!")
        print(f"ONNX model: {onnx_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

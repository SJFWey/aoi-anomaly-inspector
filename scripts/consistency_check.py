#!/usr/bin/env python
"""Consistency check between PyTorch and ONNX model outputs.

This script verifies that the exported ONNX model produces consistent outputs
compared to the original PyTorch model. It compares:
- Image-level anomaly scores (pred_score)
- Pixel-level anomaly maps
- Binary masks after post-processing

Usage:
    python scripts/consistency_check.py --run_dir runs/padim/transistor/smoke_padim2
    python scripts/consistency_check.py --run_dir runs/patchcore/transistor/smoke_patchcore2 --num_samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aoi.config import load_yaml
from aoi.models import OnnxExportWrapper, build_model
from aoi.postprocess import anomaly_map_to_mask
from aoi.thresholds import load_thresholds


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check consistency between PyTorch and ONNX model outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the training run directory containing export/ and thresholds.json",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Path to input images for testing. If not provided, uses synthetic random images",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to test (for random images or subsampling)",
    )
    parser.add_argument(
        "--onnx_name",
        type=str,
        default="model.onnx",
        help="ONNX model filename in export/ directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_onnx_session(onnx_path: Path) -> Any:
    """Load ONNX model with ONNXRuntime.

    Args:
        onnx_path: Path to the ONNX model.

    Returns:
        ONNXRuntime InferenceSession.

    Raises:
        ImportError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for consistency check. Install with: pip install onnxruntime"
        ) from e

    # Use CPU provider for consistency
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    return session


def run_pytorch_inference(
    model: OnnxExportWrapper,
    images: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with PyTorch model.

    Note: Processes images one at a time to match ONNX inference behavior.

    Args:
        model: Wrapped PyTorch model.
        images: Input tensor (B, 3, H, W).

    Returns:
        Tuple of (anomaly_map, pred_score) as numpy arrays.
    """
    model.eval()
    batch_size = images.shape[0]

    anomaly_maps = []
    pred_scores = []

    with torch.no_grad():
        for i in range(batch_size):
            single_image = images[i : i + 1]  # Keep batch dimension: (1, 3, H, W)
            anomaly_map, pred_score = model(single_image)
            anomaly_maps.append(anomaly_map.numpy())
            pred_scores.append(pred_score.numpy())

    # Stack results
    anomaly_map = np.concatenate(anomaly_maps, axis=0)
    pred_score = np.concatenate(pred_scores, axis=0)

    return anomaly_map, pred_score


def run_onnx_inference(
    session: Any,
    images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with ONNX model.

    Note: Processes images one at a time to handle models exported with fixed batch size.

    Args:
        session: ONNXRuntime InferenceSession.
        images: Input array (B, 3, H, W).

    Returns:
        Tuple of (anomaly_map, pred_score) as numpy arrays.
    """
    input_name = session.get_inputs()[0].name
    batch_size = images.shape[0]

    anomaly_maps = []
    pred_scores = []

    # Process one image at a time for compatibility
    for i in range(batch_size):
        single_image = images[i : i + 1]  # Keep batch dimension: (1, 3, H, W)
        outputs = session.run(None, {input_name: single_image})
        anomaly_maps.append(outputs[0])
        pred_scores.append(outputs[1])

    # Stack results
    anomaly_map = np.concatenate(anomaly_maps, axis=0)
    pred_score = np.concatenate(pred_scores, axis=0)

    return anomaly_map, pred_score


def compute_metrics(
    pt_anomaly_map: np.ndarray,
    pt_pred_score: np.ndarray,
    onnx_anomaly_map: np.ndarray,
    onnx_pred_score: np.ndarray,
    pixel_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute consistency metrics between PyTorch and ONNX outputs.

    Args:
        pt_anomaly_map: PyTorch anomaly map (B, 1, H, W).
        pt_pred_score: PyTorch image scores (B,).
        onnx_anomaly_map: ONNX anomaly map (B, 1, H, W).
        onnx_pred_score: ONNX image scores (B,).
        pixel_threshold: Optional threshold for mask comparison.

    Returns:
        Dictionary of metrics.
    """
    # Image score metrics
    score_diff = np.abs(pt_pred_score - onnx_pred_score)
    score_mae = float(np.mean(score_diff))
    score_max_error = float(np.max(score_diff))

    # Anomaly map metrics
    map_diff = np.abs(pt_anomaly_map - onnx_anomaly_map)
    map_mae = float(np.mean(map_diff))
    map_mse = float(np.mean(map_diff**2))
    map_max_error = float(np.max(map_diff))

    metrics = {
        "pred_score": {
            "mae": score_mae,
            "max_error": score_max_error,
        },
        "anomaly_map": {
            "mae": map_mae,
            "mse": map_mse,
            "max_error": map_max_error,
        },
    }

    # Mask comparison if threshold provided
    if pixel_threshold is not None:
        batch_size = pt_anomaly_map.shape[0]
        ious = []
        pixel_agreements = []

        for i in range(batch_size):
            # Generate masks
            pt_map = pt_anomaly_map[i, 0]  # (H, W)
            onnx_map = onnx_anomaly_map[i, 0]  # (H, W)

            pt_mask = anomaly_map_to_mask(pt_map, pixel_threshold)
            onnx_mask = anomaly_map_to_mask(onnx_map, pixel_threshold)

            # Compute IoU
            pt_binary = pt_mask > 0
            onnx_binary = onnx_mask > 0

            intersection = np.logical_and(pt_binary, onnx_binary).sum()
            union = np.logical_or(pt_binary, onnx_binary).sum()

            if union > 0:
                iou = float(intersection / union)
            else:
                # Both masks are empty
                iou = 1.0
            ious.append(iou)

            # Compute pixel agreement rate
            agreement = (pt_binary == onnx_binary).sum() / pt_binary.size
            pixel_agreements.append(float(agreement))

        metrics["mask"] = {
            "mean_iou": float(np.mean(ious)),
            "min_iou": float(np.min(ious)),
            "mean_pixel_agreement": float(np.mean(pixel_agreements)),
            "min_pixel_agreement": float(np.min(pixel_agreements)),
        }

    return metrics


def load_real_images(
    input_dir: Path,
    image_size: tuple[int, int],
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Load real images from directory.

    Args:
        input_dir: Path to image directory.
        image_size: Target size (H, W).
        num_samples: Maximum number of images to load.
        seed: Random seed for sampling.

    Returns:
        Array of images (N, 3, H, W), float32, range [0, 1].
    """
    import cv2

    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )

    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    # Sample if needed
    rng = np.random.default_rng(seed)
    if len(image_paths) > num_samples:
        indices = rng.choice(len(image_paths), num_samples, replace=False)
        image_paths = [image_paths[i] for i in sorted(indices)]

    # Load and preprocess images
    images = []
    for path in image_paths:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Warning: Failed to load {path}, skipping")
            continue

        # BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize
        img_resized = cv2.resize(
            img_rgb,
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize to [0, 1] and convert to (C, H, W)
        img_float = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        images.append(img_chw)

    return np.stack(images, axis=0)


def generate_synthetic_images(
    num_samples: int,
    image_size: tuple[int, int],
    seed: int,
) -> np.ndarray:
    """Generate synthetic random images for testing.

    Args:
        num_samples: Number of images to generate.
        image_size: Image size (H, W).
        seed: Random seed.

    Returns:
        Array of images (N, 3, H, W), float32, range [0, 1].
    """
    rng = np.random.default_rng(seed)
    images = rng.random((num_samples, 3, image_size[0], image_size[1]))
    return images.astype(np.float32)


def run_consistency_check(
    run_dir: Path,
    input_dir: Path | None = None,
    num_samples: int = 50,
    onnx_name: str = "model.onnx",
    seed: int = 42,
) -> dict[str, Any]:
    """Run consistency check between PyTorch and ONNX models.

    Args:
        run_dir: Path to training run directory.
        input_dir: Optional path to test images.
        num_samples: Number of samples to test.
        onnx_name: ONNX model filename.
        seed: Random seed.

    Returns:
        Dictionary containing check results and metrics.
    """
    # Validate paths
    config_path = run_dir / "config.yaml"
    weights_path = run_dir / "weights" / "model.ckpt"
    onnx_path = run_dir / "export" / onnx_name
    thresholds_path = run_dir / "thresholds.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Run scripts/export.py first to generate it."
        )

    # Load configuration
    print(f"Loading config from: {config_path}")
    cfg = load_yaml(config_path)

    model_name = cfg.get("model", {}).get("name", "unknown")
    pp_cfg = cfg.get("preprocessing", {})
    image_size_raw = pp_cfg.get("image_size", [256, 256])
    if isinstance(image_size_raw, int):
        image_size_raw = [image_size_raw, image_size_raw]
    image_size: tuple[int, int] = (image_size_raw[0], image_size_raw[1])

    print(f"Model: {model_name}")
    print(f"Image size: {image_size}")

    # Load thresholds if available
    pixel_threshold = None
    if thresholds_path.exists():
        thresholds = load_thresholds(thresholds_path)
        pixel_threshold = thresholds.pixel_threshold
        print(f"Loaded pixel_threshold: {pixel_threshold}")

    # Build and load PyTorch model
    print("Loading PyTorch model...")
    model = build_model(cfg)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Wrap model
    wrapped_model = OnnxExportWrapper(model)
    wrapped_model.eval()

    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    onnx_session = load_onnx_session(onnx_path)

    # Prepare test images
    if input_dir is not None:
        print(f"Loading test images from: {input_dir}")
        images = load_real_images(input_dir, image_size, num_samples, seed)
        image_source = "real"
    else:
        print(f"Generating {num_samples} synthetic test images")
        images = generate_synthetic_images(num_samples, image_size, seed)
        image_source = "synthetic"

    print(f"Test images shape: {images.shape}")

    # Run inference
    print("Running PyTorch inference...")
    images_torch = torch.from_numpy(images)
    pt_anomaly_map, pt_pred_score = run_pytorch_inference(wrapped_model, images_torch)

    print("Running ONNX inference...")
    onnx_anomaly_map, onnx_pred_score = run_onnx_inference(onnx_session, images)

    # Compute metrics
    print("Computing consistency metrics...")
    metrics = compute_metrics(
        pt_anomaly_map,
        pt_pred_score,
        onnx_anomaly_map,
        onnx_pred_score,
        pixel_threshold=pixel_threshold,
    )

    # Prepare results
    results = {
        "passed": True,  # Will be updated based on thresholds
        "model_name": model_name,
        "image_size": list(image_size),
        "num_samples": len(images),
        "image_source": image_source,
        "pixel_threshold": pixel_threshold,
        "metrics": metrics,
        "source": {
            "run_dir": str(run_dir),
            "onnx_path": str(onnx_path),
            "config_path": str(config_path),
        },
        "checked_at": datetime.now(UTC).isoformat(),
    }

    # Determine pass/fail based on tolerances
    # These thresholds are empirical - small differences are expected due to
    # floating point precision differences between PyTorch and ONNX Runtime
    score_tolerance = 1e-3  # Allow small score differences
    map_tolerance = 1e-3  # Allow small map differences
    mask_iou_threshold = 0.99

    score_ok = metrics["pred_score"]["max_error"] < score_tolerance
    map_ok = metrics["anomaly_map"]["max_error"] < map_tolerance
    mask_ok = True
    if "mask" in metrics:
        mask_ok = metrics["mask"]["min_iou"] >= mask_iou_threshold

    results["passed"] = score_ok and map_ok and mask_ok
    results["tolerances"] = {
        "score_tolerance": score_tolerance,
        "map_tolerance": map_tolerance,
        "mask_iou_threshold": mask_iou_threshold if "mask" in metrics else None,
    }
    results["check_details"] = {
        "score_passed": score_ok,
        "map_passed": map_ok,
        "mask_passed": mask_ok if "mask" in metrics else None,
    }

    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        results = run_consistency_check(
            run_dir=args.run_dir,
            input_dir=args.input_dir,
            num_samples=args.num_samples,
            onnx_name=args.onnx_name,
            seed=args.seed,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("CONSISTENCY CHECK RESULTS")
        print("=" * 60)

        metrics = results["metrics"]

        print("\nImage Score (pred_score):")
        print(f"  MAE: {metrics['pred_score']['mae']:.2e}")
        print(f"  Max Error: {metrics['pred_score']['max_error']:.2e}")

        print("\nAnomaly Map:")
        print(f"  MAE: {metrics['anomaly_map']['mae']:.2e}")
        print(f"  MSE: {metrics['anomaly_map']['mse']:.2e}")
        print(f"  Max Error: {metrics['anomaly_map']['max_error']:.2e}")

        if "mask" in metrics:
            print("\nBinary Mask:")
            print(f"  Mean IoU: {metrics['mask']['mean_iou']:.4f}")
            print(f"  Min IoU: {metrics['mask']['min_iou']:.4f}")
            print(f"  Mean Pixel Agreement: {metrics['mask']['mean_pixel_agreement']:.4f}")

        print("\n" + "-" * 60)
        if results["passed"]:
            print("✓ CONSISTENCY CHECK PASSED")
        else:
            print("✗ CONSISTENCY CHECK FAILED")
            print("  Check details:", results["check_details"])
        print("-" * 60)

        # Save results
        export_dir = args.run_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        consistency_path = export_dir / "consistency.json"

        with consistency_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\nResults saved to: {consistency_path}")

        # Exit with appropriate code
        sys.exit(0 if results["passed"] else 1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

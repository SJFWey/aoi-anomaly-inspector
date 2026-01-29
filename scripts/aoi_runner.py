#!/usr/bin/env python
"""AOI Runner: One-command inference pipeline for anomaly detection.

This script provides the main entry point for running inference on arbitrary
image folders. It loads a trained model, applies thresholds, and generates
structured outputs including:
- overlays/: Visualization with heatmap and defect contours
- masks/: Binary defect masks
- preds.jsonl: Per-image prediction records
- report.json: Summary statistics

Usage:
    python scripts/aoi_runner.py \\
        --input_dir /path/to/images \\
        --output_dir /path/to/results \\
        --model_dir runs/padim/transistor/smoke_padim2

Example:
    python scripts/aoi_runner.py \\
        --input_dir datasets/mvtech/transistor/test/damaged \\
        --output_dir outputs/runner_test \\
        --model_dir runs/padim/transistor/smoke_padim2 \\
        --device cpu \\
        --save-masks \\
        --save-overlays
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lightning.pytorch import Trainer

from aoi.callbacks import PostprocessPredictionWriter
from aoi.config import load_yaml
from aoi.device import resolve_device
from aoi.infer_data import build_infer_dataloader
from aoi.models import build_model
from aoi.report import generate_report, load_predictions, write_report
from aoi.thresholds import load_thresholds


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AOI Runner: One-command inference for anomaly detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output directory structure:
  output_dir/
    overlays/    - Visualization images with heatmap overlay
    masks/       - Binary defect masks
    preds.jsonl  - Per-image prediction records (JSON Lines)
    report.json  - Summary statistics

The model_dir should contain:
  - config.yaml: Training configuration
  - weights/model.ckpt: Model checkpoint
  - thresholds.json: Computed thresholds (from evaluate.py --compute-thresholds)
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing input images to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write outputs (overlays, masks, preds.jsonl, report.json)",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Model run directory containing config.yaml, weights/, and thresholds.json",
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: auto|cpu|cuda (default: cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        default=True,
        help="Save binary defect masks (default: True)",
    )
    parser.add_argument(
        "--no-save-masks",
        action="store_false",
        dest="save_masks",
        help="Do not save binary defect masks",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        default=True,
        help="Save overlay visualizations (default: True)",
    )
    parser.add_argument(
        "--no-save-overlays",
        action="store_false",
        dest="save_overlays",
        help="Do not save overlay visualizations",
    )
    parser.add_argument(
        "--min-defect-area",
        type=int,
        default=100,
        help="Minimum defect area in pixels for filtering (default: 100)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.4,
        help="Overlay heatmap transparency (default: 0.4)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan input_dir for images (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only scan top-level of input_dir",
    )
    # Post-processing parameters
    parser.add_argument(
        "--morph-kernel-size",
        type=int,
        default=7,
        help="Kernel size for morphological operations (default: 7)",
    )
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=7,
        help="Kernel size for Gaussian blur (must be odd, default: 7)",
    )
    parser.add_argument(
        "--no-morphology",
        action="store_true",
        help="Disable morphological operations (opening/closing)",
    )
    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="Disable Gaussian blur before thresholding",
    )

    return parser.parse_args()


def validate_model_dir(model_dir: Path) -> tuple[Path, Path, Path]:
    """Validate model directory and return paths to required files.

    Args:
        model_dir: Path to the model run directory.

    Returns:
        Tuple of (config_path, weights_path, thresholds_path).

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    config_path = model_dir / "config.yaml"
    weights_path = model_dir / "weights" / "model.ckpt"
    thresholds_path = model_dir / "thresholds.json"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Missing thresholds: {thresholds_path}\n"
            "Run 'python scripts/evaluate.py --compute-thresholds' first to generate thresholds.json"
        )

    return config_path, weights_path, thresholds_path


def main() -> int:
    """Main entry point for AOI Runner.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    # Validate model directory
    try:
        config_path, weights_path, thresholds_path = validate_model_dir(args.model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load configuration and thresholds
    cfg = load_yaml(config_path)
    thresholds = load_thresholds(thresholds_path)

    # Extract metadata from config
    model_name = cfg.get("model", {}).get("name", "unknown")
    category = cfg.get("data", {}).get("category", "unknown")
    image_size = cfg.get("preprocessing", {}).get("image_size", [256, 256])
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    image_size_tuple = (int(image_size[0]), int(image_size[1]))

    # Extract run_id from model_dir path
    run_id = args.model_dir.name

    # Print configuration
    print("=" * 60)
    print("AOI Runner - Anomaly Detection Inference")
    print("=" * 60)
    print(f"  Input directory:   {args.input_dir}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Model directory:   {args.model_dir}")
    print(f"  Model:             {model_name}")
    print(f"  Category:          {category}")
    print(f"  Run ID:            {run_id}")
    print(f"  Image size:        {image_size_tuple}")
    print(f"  Device:            {args.device}")
    print(f"  Image threshold:   {thresholds.image_threshold:.6f}")
    print(f"  Pixel threshold:   {thresholds.pixel_threshold:.6f}")
    print(f"  Save masks:        {args.save_masks}")
    print(f"  Save overlays:     {args.save_overlays}")
    print(f"  Min defect area:   {args.min_defect_area}")
    print(
        f"  Morphology:        {'enabled' if not args.no_morphology else 'disabled'} (kernel={args.morph_kernel_size})"
    )
    print(
        f"  Blur:              {'enabled' if not args.no_blur else 'disabled'} (kernel={args.blur_kernel_size})"
    )
    print(f"  Recursive scan:    {args.recursive}")
    print("=" * 60)

    # Build dataloader for input images
    try:
        dataloader = build_infer_dataloader(
            input_dir=args.input_dir,
            image_size=image_size_tuple,
            batch_size=args.batch_size,
            recursive=args.recursive,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    num_images = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Found {num_images} images to process")

    # Resolve device and build model
    accelerator, devices = resolve_device(args.device)
    precision = cfg.get("trainer", {}).get("precision", 32)
    model = build_model(cfg)

    # Create post-processing callback
    postprocess_writer = PostprocessPredictionWriter(
        thresholds=thresholds,
        output_dir=args.output_dir,
        model=model_name,
        category=category,
        image_size=image_size_tuple,
        save_masks=args.save_masks,
        save_overlays=args.save_overlays,
        min_defect_area=args.min_defect_area,
        overlay_alpha=args.overlay_alpha,
        apply_morphology=not args.no_morphology,
        morph_kernel_size=args.morph_kernel_size,
        apply_blur=not args.no_blur,
        blur_kernel_size=args.blur_kernel_size,
    )

    # Create trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
        callbacks=[postprocess_writer],
    )

    # Run inference
    print("\nRunning inference...")
    trainer.predict(model=model, dataloaders=dataloader, ckpt_path=str(weights_path))

    # Generate summary report
    print("\nGenerating report...")
    preds_path = args.output_dir / "preds.jsonl"
    if preds_path.exists():
        predictions = load_predictions(preds_path)
        report = generate_report(
            predictions,
            model=model_name,
            category=category,
            run_id=run_id,
            thresholds={
                "image_threshold": thresholds.image_threshold,
                "pixel_threshold": thresholds.pixel_threshold,
                "quantile_image": thresholds.quantile_image,
                "quantile_pixel": thresholds.quantile_pixel,
            },
        )
        report_path = args.output_dir / "report.json"
        write_report(report, report_path)
        print(f"Report saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Total images: {report['num_images']}")
        print(f"  OK:           {report['num_ok']} ({report['ok_rate'] * 100:.1f}%)")
        print(f"  NG:           {report['num_ng']} ({report['ng_rate'] * 100:.1f}%)")
        if report.get("score_stats"):
            stats = report["score_stats"]
            print(f"  Score range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Score mean:   {stats['mean']:.4f}")
        print("=" * 60)
    else:
        print(f"Warning: No predictions file generated at {preds_path}")

    print(f"\nOutputs saved to: {args.output_dir}")
    print("  - preds.jsonl")
    print("  - report.json")
    if args.save_masks:
        print("  - masks/")
    if args.save_overlays:
        print("  - overlays/")

    return 0


if __name__ == "__main__":
    sys.exit(main())

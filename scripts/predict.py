"""Run inference with post-processing on a dataset.

This script loads a trained model and thresholds, runs prediction on test data,
and outputs structured predictions with defect information, masks, and overlays.
"""

import argparse
from pathlib import Path

from lightning.pytorch import Trainer

from aoi.callbacks import PostprocessPredictionWriter
from aoi.config import load_yaml
from aoi.data import build_test_loader
from aoi.device import resolve_device
from aoi.models import build_model
from aoi.thresholds import load_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with post-processing on test data."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing config.yaml, weights/, and thresholds.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for predictions (default: <run_dir>/predictions/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device: auto|cpu|cuda",
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
        default=0,
        help="Minimum defect area in pixels for filtering (default: 0)",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.4,
        help="Overlay heatmap transparency (default: 0.4)",
    )
    args = parser.parse_args()

    # Validate paths
    config_path = args.run_dir / "config.yaml"
    weights_path = args.run_dir / "weights" / "model.ckpt"
    thresholds_path = args.run_dir / "thresholds.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Missing thresholds: {thresholds_path}\n"
            "Run evaluate.py --compute-thresholds first to generate thresholds.json"
        )

    # Load configuration and thresholds
    cfg = load_yaml(config_path)
    thresholds = load_thresholds(thresholds_path)

    if args.device:
        cfg["device"] = args.device

    # Set output directory
    output_dir = args.output_dir or (args.run_dir / "predictions")

    # Extract metadata
    model_name = cfg.get("model", {}).get("name", "unknown")
    category = cfg.get("data", {}).get("category", "unknown")
    image_size = cfg.get("preprocessing", {}).get("image_size", [256, 256])
    if isinstance(image_size, int):
        image_size = [image_size, image_size]

    # Build model and dataloader
    accelerator, devices = resolve_device(str(cfg.get("device", "auto")))
    precision = cfg.get("trainer", {}).get("precision", 32)
    model = build_model(cfg)
    test_loader = build_test_loader(cfg)

    # Create trainer with post-processing callback
    postprocess_writer = PostprocessPredictionWriter(
        thresholds=thresholds,
        output_dir=output_dir,
        model=model_name,
        category=category,
        image_size=(int(image_size[0]), int(image_size[1])),
        save_masks=args.save_masks,
        save_overlays=args.save_overlays,
        min_defect_area=args.min_defect_area,
        overlay_alpha=args.overlay_alpha,
    )

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

    print("Running inference with post-processing...")
    print(f"  Run directory: {args.run_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Image threshold: {thresholds.image_threshold:.6f}")
    print(f"  Pixel threshold: {thresholds.pixel_threshold:.6f}")
    print(f"  Save masks: {args.save_masks}")
    print(f"  Save overlays: {args.save_overlays}")
    print(f"  Min defect area: {args.min_defect_area}")

    trainer.predict(model=model, dataloaders=test_loader, ckpt_path=str(weights_path))

    print(f"\nOutputs saved to: {output_dir}")
    print("  - preds.jsonl: Structured predictions")
    if args.save_masks:
        print("  - masks/: Binary defect masks")
    if args.save_overlays:
        print("  - overlays/: Overlay visualizations")


if __name__ == "__main__":
    main()

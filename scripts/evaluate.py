import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from lightning.pytorch import Trainer

from aoi.callbacks import evaluate_and_write_metrics
from aoi.config import load_yaml
from aoi.data import build_test_loader, build_train_loader, build_validation_loader
from aoi.device import resolve_device
from aoi.models import build_model
from aoi.thresholds import (
    compute_and_save_thresholds,
    find_optimal_thresholds,
    save_thresholds,
    Thresholds,
    ValidationScoreCollector,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained run (metrics.json) and optionally compute thresholds."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory under runs/<model>/<cat>/<id>/",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device: auto|cpu|cuda.")
    # Threshold computation arguments
    parser.add_argument(
        "--compute-thresholds",
        action="store_true",
        help="Compute image/pixel thresholds from train data and save thresholds.json.",
    )
    parser.add_argument(
        "--quantile-image",
        type=float,
        default=0.995,
        help="Quantile for image threshold (default: 0.995 = 99.5%%).",
    )
    parser.add_argument(
        "--quantile-pixel",
        type=float,
        default=0.999,
        help="Quantile for pixel threshold (default: 0.999 = 99.9%%).",
    )
    parser.add_argument(
        "--pixel-sample-per-image",
        type=int,
        default=10000,
        help="Pixels to sample per image for pixel threshold (default: 10000). Use -1 for all pixels.",
    )
    parser.add_argument(
        "--pixel-sample-seed",
        type=int,
        default=42,
        help="Random seed for pixel sampling (default: 42).",
    )
    # Validation-based threshold tuning arguments
    parser.add_argument(
        "--tune-on-validation",
        action="store_true",
        help="Tune thresholds on validation (test) data using metric optimization.",
    )
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=1.0,
        help="Fraction of test set to use for validation (default: 1.0 = all). Use <1.0 for held-out evaluation.",
    )
    parser.add_argument(
        "--image-metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall", "fp_at_recall"],
        help="Metric for image threshold optimization (default: f1).",
    )
    parser.add_argument(
        "--pixel-metric",
        type=str,
        default="f1",
        choices=["f1", "iou", "recall", "recall_at_precision"],
        help="Metric for pixel threshold optimization (default: f1).",
    )
    parser.add_argument(
        "--recall-constraint",
        type=float,
        default=0.95,
        help="Recall constraint for 'fp_at_recall' metric (default: 0.95).",
    )
    parser.add_argument(
        "--precision-constraint",
        type=float,
        default=0.5,
        help="Precision constraint for 'recall_at_precision' metric (default: 0.5).",
    )
    args = parser.parse_args()

    config_path = args.run_dir / "config.yaml"
    weights_path = args.run_dir / "weights" / "model.ckpt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights_path}")

    cfg = load_yaml(config_path)
    if args.device:
        cfg["device"] = args.device

    accelerator, devices = resolve_device(str(cfg.get("device", "auto")))
    precision = cfg.get("trainer", {}).get("precision", 32)
    model = build_model(cfg)
    test_loader = build_test_loader(cfg)
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
    )
    # Extract metadata for metrics.json
    category = cfg.get("data", {}).get("category")
    image_size = cfg.get("preprocessing", {}).get("image_size")
    device_str = cfg.get("device", "cpu")

    metrics = evaluate_and_write_metrics(
        trainer=trainer,
        model=model,
        dataloader=test_loader,
        out_path=args.run_dir / "metrics.json",
        ckpt_path=weights_path,
        category=category,
        image_size=image_size,
        device=device_str,
    )
    print(f"Updated: {args.run_dir / 'metrics.json'}")
    print(f"Metrics: {metrics}")

    # Compute thresholds if requested
    if args.compute_thresholds:
        print("\nComputing thresholds from train data...")
        train_loader = build_train_loader(cfg)
        # Handle -1 meaning "use all pixels"
        pixel_sample = None if args.pixel_sample_per_image < 0 else args.pixel_sample_per_image
        # Create a fresh trainer for predict (avoid state conflicts)
        threshold_trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=True,
        )
        thresholds = compute_and_save_thresholds(
            trainer=threshold_trainer,
            model=model,
            dataloader=train_loader,
            out_path=args.run_dir / "thresholds.json",
            ckpt_path=weights_path,
            quantile_image=args.quantile_image,
            quantile_pixel=args.quantile_pixel,
            pixel_sample_per_image=pixel_sample,
            pixel_sample_seed=args.pixel_sample_seed,
        )
        print(f"Saved: {args.run_dir / 'thresholds.json'}")
        print(
            f"Thresholds: image={thresholds.image_threshold:.6f}, pixel={thresholds.pixel_threshold:.6f}"
        )

    # Tune thresholds on validation data if requested
    if args.tune_on_validation:
        print("\nTuning thresholds on validation data...")
        val_loader = build_validation_loader(cfg, split_ratio=args.val_split_ratio, seed=42)

        # Handle -1 meaning "use all pixels"
        pixel_sample = None if args.pixel_sample_per_image < 0 else args.pixel_sample_per_image

        # Collect validation scores and labels
        val_collector = ValidationScoreCollector(
            pixel_sample_per_image=pixel_sample,
            pixel_sample_seed=args.pixel_sample_seed,
        )
        val_trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=True,
            callbacks=[val_collector],
        )
        val_trainer.predict(model=model, dataloaders=val_loader, ckpt_path=str(weights_path))

        # Get collected data
        image_scores, image_labels = val_collector.get_image_data()
        pixel_scores, pixel_labels = val_collector.get_pixel_data()

        print(f"  Collected {len(image_scores)} images, {len(pixel_scores)} pixels")
        print(
            f"  Image labels distribution: {dict(zip(*np.unique(image_labels, return_counts=True)))}"
        )

        # Find optimal thresholds
        optimal = find_optimal_thresholds(
            image_scores=image_scores,
            image_labels=image_labels,
            pixel_scores=pixel_scores if len(pixel_scores) > 0 else None,
            pixel_labels=pixel_labels if len(pixel_labels) > 0 else None,
            image_metric=args.image_metric,
            pixel_metric=args.pixel_metric,
            recall_constraint=args.recall_constraint,
            precision_constraint=args.precision_constraint,
        )

        # Create updated thresholds with validation tuning info
        tuned_thresholds = Thresholds(
            image_threshold=optimal["image_threshold"],
            pixel_threshold=optimal["pixel_threshold"] or 0.0,
            quantile_image=0.0,  # Not applicable for validation-tuned
            quantile_pixel=0.0,  # Not applicable for validation-tuned
            num_train_images=0,
            pixel_sample_per_image=pixel_sample,
            pixel_sample_seed=args.pixel_sample_seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            script_version="1.1.0",
            tuned_on_validation=True,
            image_metric=args.image_metric,
            image_metric_value=optimal["image_metric_value"],
            pixel_metric=args.pixel_metric,
            pixel_metric_value=optimal["pixel_metric_value"],
            num_validation_images=val_collector.num_images,
        )

        save_thresholds(tuned_thresholds, args.run_dir / "thresholds.json")
        print(f"\nSaved validation-tuned thresholds: {args.run_dir / 'thresholds.json'}")
        print(
            f"  Image threshold: {tuned_thresholds.image_threshold:.6f} ({args.image_metric}={optimal['image_metric_value']:.4f})"
        )
        if optimal["pixel_threshold"] is not None:
            print(
                f"  Pixel threshold: {tuned_thresholds.pixel_threshold:.6f} ({args.pixel_metric}={optimal['pixel_metric_value']:.4f})"
            )
        print(f"  Search summary: {optimal['search_summary']}")


if __name__ == "__main__":
    main()

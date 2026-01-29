import argparse
from pathlib import Path

from lightning.pytorch import Trainer

from aoi.callbacks import evaluate_and_write_metrics
from aoi.config import load_yaml
from aoi.data import build_test_loader, build_train_loader
from aoi.device import resolve_device
from aoi.models import build_model
from aoi.thresholds import compute_and_save_thresholds


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
    parser.add_argument(
        "--device", type=str, default=None, help="Override device: auto|cpu|cuda."
    )
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
        pixel_sample = (
            None if args.pixel_sample_per_image < 0 else args.pixel_sample_per_image
        )
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


if __name__ == "__main__":
    main()

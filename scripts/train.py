import argparse
import platform
from datetime import UTC, datetime
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger

from aoi.callbacks import JsonlPredictionWriter, evaluate_and_write_metrics
from aoi.config import dump_json, dump_yaml, load_yaml
from aoi.data import build_dataloaders
from aoi.device import resolve_device
from aoi.models import build_model
from aoi.run import make_run_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomalib models on MVTec AD (transistor).")
    parser.add_argument("--config", type=Path, required=True, help="Path to config yaml.")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Override MVTec category (e.g. transistor).",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Override dataset root (contains categories).",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device: auto|cpu|cuda.")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Override run id (defaults to utc timestamp).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.category:
        cfg.setdefault("data", {})["category"] = args.category
    if args.data_root:
        cfg.setdefault("data", {})["root"] = str(args.data_root)
    if args.device:
        cfg["device"] = args.device
    if args.run_id:
        cfg.setdefault("run", {})["run_id"] = args.run_id

    run_paths = make_run_paths(cfg)
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed, workers=True)

    accelerator, devices = resolve_device(str(cfg.get("device", "auto")))
    trainer_cfg = cfg.get("trainer", {})
    precision = int(trainer_cfg.get("precision", 32))
    max_epochs = int(trainer_cfg.get("max_epochs", 1))
    log_every_n_steps = int(trainer_cfg.get("log_every_n_steps", 10))

    model = build_model(cfg)
    train_loader, train_pred_loader, test_loader = build_dataloaders(cfg)

    dump_yaml(run_paths.config_path, cfg)
    dump_json(
        run_paths.meta_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": seed,
            "device": str(cfg.get("device", "auto")),
            "accelerator": accelerator,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    )

    logger = CSVLogger(save_dir=str(run_paths.run_dir / "logs"), name="")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

    run_paths.weights_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(run_paths.weights_path), weights_only=True)

    if bool(cfg.get("run", {}).get("save_predictions", True)):
        pred_trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=True,
            callbacks=[
                JsonlPredictionWriter(
                    out_path=run_paths.train_preds_path,
                    split="train",
                    model=str(cfg.get("model", {}).get("name", "")),
                    category=str(cfg.get("data", {}).get("category", "")),
                ),
            ],
        )
        pred_trainer.predict(
            model=model,
            dataloaders=train_pred_loader,
            return_predictions=False,
        )

        pred_trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            deterministic=True,
            callbacks=[
                JsonlPredictionWriter(
                    out_path=run_paths.test_preds_path,
                    split="test",
                    model=str(cfg.get("model", {}).get("name", "")),
                    category=str(cfg.get("data", {}).get("category", "")),
                ),
            ],
        )
        pred_trainer.predict(
            model=model,
            dataloaders=test_loader,
            return_predictions=False,
        )

    test_trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
    )
    evaluate_and_write_metrics(
        trainer=test_trainer,
        model=model,
        dataloader=test_loader,
        out_path=run_paths.metrics_path,
    )

    print(f"Run saved to: {run_paths.run_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np

from aoi.artifacts import RunPaths, environment_snapshot, git_commit, sha256_file, update_manifest, write_json_atomic, write_yaml_atomic
from aoi.config import load_experiment_config
from aoi.data import load_mvtec_split
from aoi.modeling import fit_model, load_model, model_metadata, save_model
from aoi.reporting import now_utc


def _run_id() -> str:
    return now_utc().replace("-", "").replace(":", "").replace("+", "Z")


def train_run(
    config_path: Path,
    *,
    data_root: Path | None = None,
    category: str | None = None,
    device: str | None = None,
    run_root: Path | None = None,
    run_id: str | None = None,
) -> RunPaths:
    config = load_experiment_config(config_path, data_root=data_root, category=category, device=device)
    root = Path(run_root) if run_root is not None else config.run_root
    paths = RunPaths(root / config.category / (run_id or _run_id()))
    if paths.root.exists() and any(paths.root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=False)

    train_good = [
        sample for sample in load_mvtec_split(config.data_root, config.category, "train") if sample.label == "good"
    ]
    if not train_good:
        raise RuntimeError(f"No train/good images found under {config.data_root / config.category}")

    write_yaml_atomic(paths.config, config.to_dict())
    model = fit_model(
        model_name=config.model_name,
        image_paths=[sample.image_path for sample in train_good],
        backbone=config.backbone,
        layers=config.layers,
        image_size=config.image_size,
        pre_trained=config.pre_trained,
        n_features=config.n_features,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        max_coreset_patches=config.max_coreset_patches,
        seed=config.seed,
        device=config.device,
        batch_size=config.batch_size,
    )
    save_model(paths.model, model)
    reloaded = load_model(paths.model, device=config.device)
    validation = reloaded.predict_path(train_good[0].image_path)
    if not np.isfinite(validation.image_score) or not np.isfinite(validation.anomaly_map).all():
        raise ValueError("Reloaded model produced non-finite validation output")
    model_hash = sha256_file(paths.model)
    write_json_atomic(
        paths.meta,
        {
            "created_at": now_utc(),
            "model": config.model_name,
            "category": config.category,
            "source_git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "model_sha256": model_hash,
            "model_state": model_metadata(model),
            "environment": environment_snapshot(),
        },
    )
    update_manifest(
        paths.manifest,
        {
            "training": {
                "status": "complete",
                "created_at": now_utc(),
                "model_sha256": model_hash,
                "outputs": [paths.config.name, paths.model.name, paths.meta.name],
            }
        },
    )
    return paths

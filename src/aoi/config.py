from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data_root: Path
    category: str
    image_size: int
    model_name: str
    backbone: str
    layers: tuple[str, ...]
    pre_trained: bool
    n_features: int
    coreset_sampling_ratio: float
    max_coreset_patches: int
    device: str
    batch_size: int
    run_root: Path
    quantile_image: float
    quantile_pixel: float
    min_area: int
    morph_kernel: int
    consistency_max_abs_error: float
    consistency_mean_abs_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dataset": {
                "root": str(self.data_root),
                "category": self.category,
                "image_size": self.image_size,
            },
            "model": {
                "name": self.model_name,
                "backbone": self.backbone,
                "layers": list(self.layers),
                "pre_trained": self.pre_trained,
                "n_features": self.n_features,
                "coreset_sampling_ratio": self.coreset_sampling_ratio,
                "max_coreset_patches": self.max_coreset_patches,
            },
            "trainer": {"device": self.device, "batch_size": self.batch_size},
            "outputs": {"run_root": str(self.run_root)},
            "thresholds": {
                "image_quantile": self.quantile_image,
                "pixel_quantile": self.quantile_pixel,
            },
            "postprocess": {"min_area": self.min_area, "morph_kernel": self.morph_kernel},
            "consistency": {
                "max_abs_error": self.consistency_max_abs_error,
                "mean_abs_error": self.consistency_mean_abs_error,
            },
        }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _positive(name: str, value: int | float) -> int | float:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _quantile(name: str, value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return value


def load_experiment_config(
    path: Path,
    data_root: Path | None = None,
    category: str | None = None,
    device: str | None = None,
) -> ExperimentConfig:
    raw = load_yaml(path)
    dataset = _mapping(raw, "dataset")
    model = _mapping(raw, "model")
    trainer = _mapping(raw, "trainer")
    outputs = _mapping(raw, "outputs")
    thresholds = _mapping(raw, "thresholds")
    postprocess = _mapping(raw, "postprocess")
    consistency = _mapping(raw, "consistency")

    model_name = str(model.get("name", "")).lower()
    if model_name not in {"padim", "patchcore"}:
        raise ValueError(f"Unsupported model name: {model_name!r}")
    backbone = str(model.get("backbone", "resnet18"))
    if backbone != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone!r}")
    layers = tuple(str(layer) for layer in model.get("layers", ["layer2"]))
    if not layers or any(layer not in {"layer1", "layer2", "layer3", "layer4"} for layer in layers):
        raise ValueError("layers must contain supported ResNet feature layers")

    image_size = int(_positive("image_size", int(dataset.get("image_size", 256))))
    n_features = int(_positive("n_features", int(model.get("n_features", 64))))
    batch_size = int(_positive("batch_size", int(trainer.get("batch_size", 8))))
    max_coreset_patches = int(
        _positive("max_coreset_patches", int(model.get("max_coreset_patches", 20000)))
    )
    coreset_sampling_ratio = float(model.get("coreset_sampling_ratio", 0.1))
    if not 0.0 < coreset_sampling_ratio <= 1.0:
        raise ValueError("coreset_sampling_ratio must be in (0, 1]")
    quantile_image = _quantile("image_quantile", float(thresholds.get("image_quantile", 0.995)))
    quantile_pixel = _quantile("pixel_quantile", float(thresholds.get("pixel_quantile", 0.999)))
    max_abs_error = float(
        _positive("consistency.max_abs_error", float(consistency.get("max_abs_error", 0.0001)))
    )
    mean_abs_error = float(
        _positive("consistency.mean_abs_error", float(consistency.get("mean_abs_error", 0.00001)))
    )

    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        data_root=Path(data_root if data_root is not None else dataset.get("root", "datasets/mvtec")),
        category=str(category if category is not None else dataset.get("category", "bottle")),
        image_size=image_size,
        model_name=model_name,
        backbone=backbone,
        layers=layers,
        pre_trained=bool(model.get("pre_trained", True)),
        n_features=n_features,
        coreset_sampling_ratio=coreset_sampling_ratio,
        max_coreset_patches=max_coreset_patches,
        device=str(device if device is not None else trainer.get("device", "cpu")),
        batch_size=batch_size,
        run_root=Path(outputs.get("run_root", f"runs/{model_name}")),
        quantile_image=quantile_image,
        quantile_pixel=quantile_pixel,
        min_area=int(postprocess.get("min_area", 50)),
        morph_kernel=int(postprocess.get("morph_kernel", 3)),
        consistency_max_abs_error=max_abs_error,
        consistency_mean_abs_error=mean_abs_error,
    )


def write_resolved_config(path: Path, config: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")

"""Model factory utilities."""

from __future__ import annotations

from typing import Any

from anomalib.models.image import Padim, Patchcore

from .device import as_int_pair

__all__ = ["build_model"]


def build_model(cfg: dict[str, Any]) -> Padim | Patchcore:
    """Build Padim or Patchcore model from configuration.

    Args:
        cfg: Full configuration dictionary containing 'model' and 'preprocessing' sections.

    Returns:
        Configured Padim or Patchcore model instance.

    Raises:
        ValueError: If model.name is not 'padim' or 'patchcore'.
    """
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "")).lower().strip()
    if model_name not in {"padim", "patchcore"}:
        msg = f"Unsupported model.name: {model_name!r} (expected padim|patchcore)"
        raise ValueError(msg)

    pp_cfg = cfg.get("preprocessing", {})
    image_size = as_int_pair(
        pp_cfg.get("image_size", (256, 256)), key="preprocessing.image_size"
    )

    if model_name == "padim":
        model = Padim(
            backbone=str(model_cfg.get("backbone", "resnet18")),
            layers=list(model_cfg.get("layers", ["layer1", "layer2", "layer3"])),
            pre_trained=bool(model_cfg.get("pre_trained", True)),
            n_features=model_cfg.get("n_features", None),
            pre_processor=True,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
        model.pre_processor = Padim.configure_pre_processor(image_size=image_size)
        return model

    # patchcore
    center_crop_size = pp_cfg.get("center_crop_size")
    center_crop_size = (
        as_int_pair(center_crop_size, key="preprocessing.center_crop_size")
        if center_crop_size
        else None
    )
    model = Patchcore(
        backbone=str(model_cfg.get("backbone", "resnet18")),
        layers=list(model_cfg.get("layers", ["layer2", "layer3"])),
        pre_trained=bool(model_cfg.get("pre_trained", True)),
        coreset_sampling_ratio=float(model_cfg.get("coreset_sampling_ratio", 0.1)),
        num_neighbors=int(model_cfg.get("num_neighbors", 9)),
        pre_processor=True,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    model.pre_processor = Patchcore.configure_pre_processor(
        image_size=image_size, center_crop_size=center_crop_size
    )
    return model

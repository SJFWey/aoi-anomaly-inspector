"""Model factory utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from anomalib.models.image import Padim, Patchcore

from .device import as_int_pair

__all__ = ["build_model", "OnnxExportWrapper"]


class OnnxExportWrapper(nn.Module):
    """Wrapper module for ONNX export with standardized outputs.

    This wrapper ensures consistent output format for ONNX export:
    - anomaly_map: (B, 1, H, W) float32 - pixel-wise anomaly scores
    - pred_score: (B,) float32 - image-level anomaly scores (max of anomaly_map)

    The wrapper includes the model's pre_processor (ImageNet normalization),
    so external callers only need to provide images normalized to [0, 1] range.
    """

    def __init__(self, model: Padim | Patchcore) -> None:
        """Initialize the export wrapper.

        Args:
            model: Trained anomalib model (Padim or Patchcore).
        """
        super().__init__()
        self.model = model
        # Store pre_processor separately for explicit control
        self.pre_processor = model.pre_processor

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ONNX export.

        Args:
            image: Input tensor of shape (B, 3, H, W), float32, normalized to [0, 1].

        Returns:
            Tuple of:
            - anomaly_map: (B, 1, H, W) float32
            - pred_score: (B,) float32
        """
        # Apply pre-processor (ImageNet normalization + any transforms)
        if self.pre_processor is not None:
            image = self.pre_processor(image)

        # Get model predictions
        # anomalib models return a dict or dataclass with anomaly_map
        outputs = self.model.model(image)

        # Extract anomaly_map from model output
        # PaDiM and PatchCore both return anomaly_map in their forward
        if isinstance(outputs, dict):
            anomaly_map = outputs.get("anomaly_map")
        elif hasattr(outputs, "anomaly_map"):
            anomaly_map = outputs.anomaly_map
        else:
            anomaly_map = outputs

        if anomaly_map is None:
            raise ValueError("Model output does not contain 'anomaly_map'")

        # Ensure anomaly_map is (B, 1, H, W)
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.unsqueeze(1)
        elif anomaly_map.ndim == 4 and anomaly_map.shape[1] != 1:
            # Take max across channel dimension if not single channel
            anomaly_map = anomaly_map.max(dim=1, keepdim=True)[0]

        # Compute image-level score as max of anomaly_map
        pred_score = anomaly_map.amax(dim=(-3, -2, -1))  # (B,)

        return anomaly_map, pred_score


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
    patchcore_model = Patchcore(
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
    patchcore_model.pre_processor = Patchcore.configure_pre_processor(
        image_size=image_size, center_crop_size=center_crop_size
    )
    return patchcore_model

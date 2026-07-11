from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from aoi.data import bgr_to_rgb_unit_tensor, read_bgr


PADIM_STD_FLOOR = 5e-2


@dataclass(frozen=True)
class Prediction:
    image_score: float
    anomaly_map: np.ndarray
    latency_ms: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _resnet18(pre_trained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pre_trained else None
    try:
        return resnet18(weights=weights)
    except Exception as exc:
        if pre_trained:
            raise RuntimeError(f"Failed to load pretrained ResNet18 weights: {exc}") from exc
        raise


class _ResNetFeatureBackbone(nn.Module):
    def __init__(
        self, backbone: str, layers: tuple[str, ...], backbone_state: dict[str, torch.Tensor] | None = None
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Unsupported backbone: only resnet18 is available")
        self.layers = layers
        self.backbone = _resnet18(pre_trained=False)
        if backbone_state is not None:
            self.backbone.load_state_dict(backbone_state)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        model = self.backbone
        value = model.conv1(images)
        value = model.bn1(value)
        value = model.relu(value)
        value = model.maxpool(value)
        maps: dict[str, torch.Tensor] = {}
        value = model.layer1(value)
        maps["layer1"] = value
        value = model.layer2(value)
        maps["layer2"] = value
        value = model.layer3(value)
        maps["layer3"] = value
        value = model.layer4(value)
        maps["layer4"] = value
        return [maps[layer] for layer in self.layers]


def _combine_feature_maps(maps: list[torch.Tensor]) -> torch.Tensor:
    target_hw = maps[-1].shape[-2:]
    resized = [
        functional.interpolate(feature, size=target_hw, mode="bilinear", align_corners=False)
        if feature.shape[-2:] != target_hw
        else feature
        for feature in maps
    ]
    return torch.cat(resized, dim=1)


class AnomalyTensorModel(nn.Module):
    def __init__(self, state: dict[str, Any]) -> None:
        super().__init__()
        self.model_name = str(state["model_name"])
        self.image_size = int(state["image_size"])
        self.feature_backbone = _ResNetFeatureBackbone(
            str(state["backbone"]), tuple(state["layers"]), state["backbone_state"]
        )
        self.register_buffer("mean_rgb", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std_rgb", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("feature_indices", torch.as_tensor(state["feature_indices"], dtype=torch.long))
        if self.model_name == "padim":
            self.register_buffer("mean", torch.as_tensor(state["mean"], dtype=torch.float32))
            self.register_buffer("inv_std", torch.as_tensor(state["inv_std"], dtype=torch.float32))
        elif self.model_name == "patchcore":
            self.register_buffer("feature_bank", torch.as_tensor(state["feature_bank"], dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    @staticmethod
    def patchcore_distance(features: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        patches = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])
        patches = functional.normalize(patches, p=2, dim=-1)
        patch_sq = (patches * patches).sum(dim=-1, keepdim=True)
        bank_sq = (bank * bank).sum(dim=-1).view(1, 1, -1)
        distance_sq = (patch_sq + bank_sq - 2.0 * patches @ bank.T).clamp_min(0.0)
        # Equivalent normalized vectors can leave small positive residuals after
        # backend-specific matrix multiplication. Treat float32 roundoff as zero
        # before sqrt amplifies it into a visible native/ONNX discrepancy.
        distance_sq = torch.where(distance_sq <= 1e-6, torch.zeros_like(distance_sq), distance_sq)
        nearest = distance_sq.min(dim=-1).values.sqrt()
        return nearest.reshape(features.shape[0], features.shape[2], features.shape[3])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected RGB input with shape [B, 3, H, W]")
        if not images.is_floating_point():
            raise ValueError("Expected float RGB input in [0, 1]")
        normalized = (images - self.mean_rgb) / self.std_rgb
        features = _combine_feature_maps(self.feature_backbone(normalized))
        selected = features[:, self.feature_indices, :, :]
        if self.model_name == "padim":
            values = selected.permute(0, 2, 3, 1)
            z_score = (values - self.mean) * self.inv_std
            anomaly = (z_score * z_score).mean(dim=-1)
        else:
            anomaly = self.patchcore_distance(selected, self.feature_bank)
        anomaly = functional.interpolate(
            anomaly[:, None], size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        score = anomaly.flatten(1).max(dim=1).values
        return anomaly, score


def validate_model_state(state: dict[str, Any]) -> None:
    def check(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Model state contains non-finite tensor: {path}")
        if isinstance(value, dict):
            for key, nested in value.items():
                check(nested, f"{path}.{key}" if path else str(key))

    check(state, "")


def build_tensor_model(state: dict[str, Any], device: str = "cpu") -> AnomalyTensorModel:
    validate_model_state(state)
    return AnomalyTensorModel(state).eval().to(torch.device(device))


class AnomalyModel:
    def __init__(self, state: dict[str, Any], device: str = "cpu") -> None:
        self.state = state
        self.device = torch.device(device)
        self.tensor_model = build_tensor_model(state, device=device)

    @property
    def name(self) -> str:
        return str(self.state["model_name"])

    @torch.inference_mode()
    def predict_path(self, image_path: Path) -> Prediction:
        start = time.perf_counter()
        tensor = bgr_to_rgb_unit_tensor(read_bgr(image_path), int(self.state["image_size"])).unsqueeze(0)
        anomaly_map, score = self.tensor_model(tensor.to(self.device))
        latency_ms = (time.perf_counter() - start) * 1000.0
        return Prediction(
            image_score=float(score[0].cpu()),
            anomaly_map=anomaly_map[0, 0].cpu().numpy().astype(np.float32),
            latency_ms=latency_ms,
        )


def _batch_tensors(image_paths: list[Path], image_size: int) -> torch.Tensor:
    return torch.stack([bgr_to_rgb_unit_tensor(read_bgr(path), image_size) for path in image_paths])


@torch.inference_mode()
def _extract_all(
    feature_backbone: _ResNetFeatureBackbone,
    image_paths: list[Path],
    image_size: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for start in range(0, len(image_paths), batch_size):
        images = _batch_tensors(image_paths[start : start + batch_size], image_size).to(device)
        features.append(_combine_feature_maps(feature_backbone((images - mean) / std)).cpu())
    return torch.cat(features)


def _feature_indices(num_channels: int, n_features: int, seed: int) -> torch.Tensor:
    if n_features >= num_channels:
        return torch.arange(num_channels)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_channels, generator=generator)[:n_features].sort().values


def fit_model(
    model_name: str,
    image_paths: list[Path],
    backbone: str,
    layers: tuple[str, ...],
    image_size: int,
    pre_trained: bool,
    n_features: int,
    coreset_sampling_ratio: float,
    max_coreset_patches: int,
    seed: int,
    device: str,
    batch_size: int = 8,
) -> AnomalyModel:
    if not image_paths:
        raise ValueError("At least one training image is required")
    set_seed(seed)
    base = _resnet18(pre_trained)
    backbone_state = {key: value.detach().cpu() for key, value in base.state_dict().items()}
    feature_backbone = _ResNetFeatureBackbone(backbone, layers, backbone_state).eval().to(device)
    features = _extract_all(feature_backbone, image_paths, image_size, device, batch_size)
    indices = _feature_indices(features.shape[1], n_features, seed)
    selected = features[:, indices, :, :]
    state: dict[str, Any] = {
        "model_name": model_name,
        "backbone": backbone,
        "layers": tuple(layers),
        "image_size": image_size,
        "pre_trained": pre_trained,
        "feature_indices": indices,
        "backbone_state": backbone_state,
        "seed": seed,
    }
    if model_name == "padim":
        values = selected.permute(0, 2, 3, 1)
        state["mean"] = values.mean(dim=0)
        # Population statistics keep one-image smoke runs finite. The explicit
        # floor also prevents tiny native/ONNX backbone differences from being
        # amplified into meaningless PaDiM scores.
        std = values.std(dim=0, correction=0).clamp_min(PADIM_STD_FLOOR)
        state["inv_std"] = 1.0 / std
    elif model_name == "patchcore":
        patches = selected.permute(0, 2, 3, 1).reshape(-1, selected.shape[1])
        patches = functional.normalize(patches, p=2, dim=1)
        sample_count = min(
            max(1, int(patches.shape[0] * coreset_sampling_ratio)), max_coreset_patches, patches.shape[0]
        )
        generator = torch.Generator().manual_seed(seed)
        state["feature_bank"] = patches[torch.randperm(patches.shape[0], generator=generator)[:sample_count]].contiguous()
        state["coreset_sampling_ratio"] = coreset_sampling_ratio
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return AnomalyModel(state, device=device)


def save_model(path: Path, model: AnomalyModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state, path)


def load_model(path: Path, device: str = "cpu") -> AnomalyModel:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return AnomalyModel(state, device=device)


def model_metadata(model: AnomalyModel) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in model.state.items():
        if isinstance(value, torch.Tensor):
            metadata[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict):
            metadata[key] = {"tensor_count": len(value)}
        else:
            metadata[key] = value
    return metadata

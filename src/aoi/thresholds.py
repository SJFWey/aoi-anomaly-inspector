"""Threshold computation and management for AOI pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = [
    "Thresholds",
    "ThresholdCollector",
    "compute_and_save_thresholds",
    "load_thresholds",
]


@dataclass
class Thresholds:
    """Data class for threshold values and metadata."""

    image_threshold: float
    pixel_threshold: float
    quantile_image: float = 0.995
    quantile_pixel: float = 0.999
    num_train_images: int = 0
    pixel_sample_per_image: int | None = None
    pixel_sample_seed: int | None = None
    created_at: str = field(default_factory=lambda: "")
    script_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Thresholds":
        """Create from dictionary."""
        return cls(
            image_threshold=float(d["image_threshold"]),
            pixel_threshold=float(d["pixel_threshold"]),
            quantile_image=float(d.get("quantile_image", 0.995)),
            quantile_pixel=float(d.get("quantile_pixel", 0.999)),
            num_train_images=int(d.get("num_train_images", 0)),
            pixel_sample_per_image=d.get("pixel_sample_per_image"),
            pixel_sample_seed=d.get("pixel_sample_seed"),
            created_at=str(d.get("created_at", "")),
            script_version=str(d.get("script_version", "1.0.0")),
        )


class ThresholdCollector(Callback):
    """Callback to collect image scores and anomaly map pixels for threshold computation."""

    def __init__(
        self,
        *,
        quantile_image: float = 0.995,
        quantile_pixel: float = 0.999,
        pixel_sample_per_image: int | None = 10000,
        pixel_sample_seed: int = 42,
    ) -> None:
        """Initialize the threshold collector.

        Args:
            quantile_image: Quantile for image threshold (e.g., 0.995 = 99.5th percentile).
            quantile_pixel: Quantile for pixel threshold (e.g., 0.999 = 99.9th percentile).
            pixel_sample_per_image: Number of pixels to sample per image for pixel threshold.
                If None, use all pixels (may be memory intensive).
            pixel_sample_seed: Random seed for pixel sampling reproducibility.
        """
        super().__init__()
        self.quantile_image = quantile_image
        self.quantile_pixel = quantile_pixel
        self.pixel_sample_per_image = pixel_sample_per_image
        self.pixel_sample_seed = pixel_sample_seed

        self._image_scores: list[float] = []
        self._pixel_values: list[np.ndarray] = []
        self._num_images: int = 0
        self._rng = np.random.default_rng(pixel_sample_seed)

    def reset(self) -> None:
        """Clear all collected data."""
        self._image_scores.clear()
        self._pixel_values.clear()
        self._num_images = 0
        self._rng = np.random.default_rng(self.pixel_sample_seed)

    def on_predict_batch_end(  # noqa: ANN001
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect scores from each prediction batch."""
        del trainer, pl_module, batch_idx, dataloader_idx

        record_batch = outputs if outputs is not None else batch
        pred_score = getattr(record_batch, "pred_score", None)
        anomaly_map = getattr(record_batch, "anomaly_map", None)

        # Fallback: derive image score from anomaly map max
        if pred_score is None and anomaly_map is not None:
            pred_score = anomaly_map.amax(dim=(-2, -1))

        # Collect image scores
        if pred_score is not None:
            scores = pred_score.detach().cpu().to(torch.float32).view(-1).numpy()
            self._image_scores.extend(scores.tolist())
            self._num_images += len(scores)

        # Collect pixel values (with optional sampling)
        if anomaly_map is not None:
            amap = anomaly_map.detach().cpu().to(torch.float32).numpy()
            batch_size = amap.shape[0]
            for i in range(batch_size):
                pixels = amap[i].flatten()
                if self.pixel_sample_per_image is not None and len(pixels) > self.pixel_sample_per_image:
                    indices = self._rng.choice(
                        len(pixels), size=self.pixel_sample_per_image, replace=False
                    )
                    pixels = pixels[indices]
                self._pixel_values.append(pixels)

    def compute_thresholds(self) -> Thresholds:
        """Compute thresholds from collected data.

        Returns:
            Thresholds dataclass with computed values.

        Raises:
            ValueError: If no data was collected.
        """
        if not self._image_scores:
            raise ValueError("No image scores collected; cannot compute thresholds.")
        if not self._pixel_values:
            raise ValueError("No pixel values collected; cannot compute thresholds.")

        # Compute image threshold
        image_scores = np.array(self._image_scores, dtype=np.float32)
        image_threshold = float(np.quantile(image_scores, self.quantile_image))

        # Compute pixel threshold
        all_pixels = np.concatenate(self._pixel_values)
        pixel_threshold = float(np.quantile(all_pixels, self.quantile_pixel))

        return Thresholds(
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            quantile_image=self.quantile_image,
            quantile_pixel=self.quantile_pixel,
            num_train_images=self._num_images,
            pixel_sample_per_image=self.pixel_sample_per_image,
            pixel_sample_seed=self.pixel_sample_seed,
            created_at=datetime.now(timezone.utc).isoformat(),
            script_version="1.0.0",
        )


def save_thresholds(thresholds: Thresholds, path: Path) -> None:
    """Save thresholds to JSON file.

    Args:
        thresholds: Thresholds dataclass to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(thresholds.to_dict(), f, indent=2, ensure_ascii=False)


def load_thresholds(path: Path) -> Thresholds:
    """Load thresholds from JSON file.

    Args:
        path: Path to thresholds JSON file.

    Returns:
        Thresholds dataclass.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Thresholds.from_dict(data)


def compute_and_save_thresholds(
    *,
    trainer: Trainer,
    model,  # noqa: ANN001
    dataloader: "DataLoader",
    out_path: Path,
    ckpt_path: Path | str | None = None,
    quantile_image: float = 0.995,
    quantile_pixel: float = 0.999,
    pixel_sample_per_image: int | None = 10000,
    pixel_sample_seed: int = 42,
) -> Thresholds:
    """Run prediction on train data and compute thresholds.

    Args:
        trainer: PyTorch Lightning Trainer instance.
        model: Model to run predictions with.
        dataloader: Train DataLoader (good samples only).
        out_path: Path to write thresholds.json.
        ckpt_path: Optional checkpoint path to load before predicting.
        quantile_image: Quantile for image threshold.
        quantile_pixel: Quantile for pixel threshold.
        pixel_sample_per_image: Pixels to sample per image (None = all).
        pixel_sample_seed: Random seed for pixel sampling.

    Returns:
        Computed Thresholds dataclass.
    """
    collector = ThresholdCollector(
        quantile_image=quantile_image,
        quantile_pixel=quantile_pixel,
        pixel_sample_per_image=pixel_sample_per_image,
        pixel_sample_seed=pixel_sample_seed,
    )
    trainer.callbacks.append(collector)  # type: ignore[attr-defined]

    if ckpt_path is not None:
        trainer.predict(model=model, dataloaders=dataloader, ckpt_path=str(ckpt_path))
    else:
        trainer.predict(model=model, dataloaders=dataloader)

    trainer.callbacks.remove(collector)  # type: ignore[attr-defined]

    thresholds = collector.compute_thresholds()
    save_thresholds(thresholds, out_path)
    return thresholds

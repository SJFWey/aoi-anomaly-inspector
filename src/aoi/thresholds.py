"""Threshold computation and management for AOI pipeline.

Threshold Application Conventions
---------------------------------

1. **image_threshold**: Used for OK/NG classification at the image level.
   - Compare `pred_score` (or `anomaly_map.max()`) against `image_threshold`.
   - If `pred_score >= image_threshold` → NG (anomaly detected)
   - If `pred_score < image_threshold` → OK (normal)

2. **pixel_threshold**: Used for generating binary defect masks.
   - Compare each pixel of `anomaly_map` against `pixel_threshold`.
   - `mask[y, x] = 1` if `anomaly_map[y, x] >= pixel_threshold` else `0`

Normalization Note:
   Normalization (e.g., min-max to [0,1]) is only for visualization purposes.
   Threshold comparisons should always use raw anomaly values.

Threshold Computation Methods:
   1. **Quantile-based** (ThresholdCollector): Compute from training normal samples.
   2. **Validation-tuned** (find_optimal_thresholds): Optimize metric on labeled val set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = [
    "Thresholds",
    "ThresholdCollector",
    "ValidationScoreCollector",
    "compute_and_save_thresholds",
    "find_optimal_thresholds",
    "load_thresholds",
    "save_thresholds",
]


@dataclass
class Thresholds:
    """Data class for threshold values and metadata."""

    image_threshold: float
    pixel_threshold: float

    # Quantile-based computation metadata
    quantile_image: float = 0.0
    quantile_pixel: float = 0.0
    num_train_images: int = 0
    pixel_sample_per_image: int | None = None
    pixel_sample_seed: int | None = None

    # Validation-tuned metadata
    tuned_on_validation: bool = False
    image_metric: str | None = None
    image_metric_value: float | None = None
    pixel_metric: str | None = None
    pixel_metric_value: float | None = None
    num_validation_images: int = 0

    # General metadata
    created_at: str = field(default_factory=lambda: "")
    script_version: str = "2.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Thresholds:
        """Create from dictionary."""
        return cls(
            image_threshold=float(d["image_threshold"]),
            pixel_threshold=float(d["pixel_threshold"]),
            quantile_image=float(d.get("quantile_image", 0.0)),
            quantile_pixel=float(d.get("quantile_pixel", 0.0)),
            num_train_images=int(d.get("num_train_images", 0)),
            pixel_sample_per_image=d.get("pixel_sample_per_image"),
            pixel_sample_seed=d.get("pixel_sample_seed"),
            tuned_on_validation=bool(d.get("tuned_on_validation", False)),
            image_metric=d.get("image_metric"),
            image_metric_value=d.get("image_metric_value"),
            pixel_metric=d.get("pixel_metric"),
            pixel_metric_value=d.get("pixel_metric_value"),
            num_validation_images=int(d.get("num_validation_images", 0)),
            created_at=str(d.get("created_at", "")),
            script_version=str(d.get("script_version", "2.0.0")),
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
                if (
                    self.pixel_sample_per_image is not None
                    and len(pixels) > self.pixel_sample_per_image
                ):
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
            created_at=datetime.now(UTC).isoformat(),
            script_version="2.0.0",
        )


class ValidationScoreCollector(Callback):
    """Callback to collect scores and labels from validation set for threshold tuning."""

    def __init__(
        self,
        *,
        pixel_sample_per_image: int | None = 5000,
        pixel_sample_seed: int = 42,
    ) -> None:
        """Initialize the validation score collector.

        Args:
            pixel_sample_per_image: Pixels to sample per image (None = all).
            pixel_sample_seed: Random seed for reproducible sampling.
        """
        super().__init__()
        self.pixel_sample_per_image = pixel_sample_per_image
        self.pixel_sample_seed = pixel_sample_seed

        self._image_scores: list[float] = []
        self._image_labels: list[int] = []
        self._pixel_scores: list[np.ndarray] = []
        self._pixel_labels: list[np.ndarray] = []
        self._num_images: int = 0
        self._rng = np.random.default_rng(pixel_sample_seed)

    @property
    def num_images(self) -> int:
        """Number of images collected."""
        return self._num_images

    def reset(self) -> None:
        """Clear all collected data."""
        self._image_scores.clear()
        self._image_labels.clear()
        self._pixel_scores.clear()
        self._pixel_labels.clear()
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
        """Collect scores and labels from each prediction batch."""
        del trainer, pl_module, batch_idx, dataloader_idx

        record_batch = outputs if outputs is not None else batch
        pred_score = getattr(record_batch, "pred_score", None)
        anomaly_map = getattr(record_batch, "anomaly_map", None)
        # Labels and masks come from the input batch (anomalib uses gt_label/gt_mask)
        label = getattr(batch, "gt_label", None)
        if label is None:
            label = getattr(batch, "label", None)
        gt_mask = getattr(batch, "gt_mask", None)
        if gt_mask is None:
            gt_mask = getattr(batch, "mask", None)

        # Fallback: derive image score from anomaly map max
        if pred_score is None and anomaly_map is not None:
            pred_score = anomaly_map.amax(dim=(-2, -1))

        # Collect image scores and labels
        if pred_score is not None:
            scores = pred_score.detach().cpu().to(torch.float32).view(-1).numpy()
            self._image_scores.extend(scores.tolist())
            self._num_images += len(scores)

            if label is not None:
                labels = label.detach().cpu().to(torch.int32).view(-1).numpy()
                self._image_labels.extend(labels.tolist())

        # Collect pixel scores and labels (with sampling)
        if anomaly_map is not None and gt_mask is not None:
            amap = anomaly_map.detach().cpu().to(torch.float32).numpy()
            mask = gt_mask.detach().cpu().to(torch.int32).numpy()
            batch_size = amap.shape[0]

            for i in range(batch_size):
                pixels = amap[i].flatten()
                labels = mask[i].flatten()

                # Sample for efficiency
                if (
                    self.pixel_sample_per_image is not None
                    and len(pixels) > self.pixel_sample_per_image
                ):
                    indices = self._rng.choice(
                        len(pixels), size=self.pixel_sample_per_image, replace=False
                    )
                    pixels = pixels[indices]
                    labels = labels[indices]

                self._pixel_scores.append(pixels)
                self._pixel_labels.append(labels)

    def get_image_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get collected image scores and labels.

        Returns:
            Tuple of (scores, labels) as numpy arrays.
        """
        return np.array(self._image_scores), np.array(self._image_labels)

    def get_pixel_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get collected pixel scores and labels.

        Returns:
            Tuple of (scores, labels) as numpy arrays.
        """
        if not self._pixel_scores:
            return np.array([]), np.array([])
        return np.concatenate(self._pixel_scores), np.concatenate(self._pixel_labels)


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
    dataloader: DataLoader,
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


# Type alias for metric names
ImageMetric = Literal["f1", "precision", "recall", "fp_at_recall"]
PixelMetric = Literal["f1", "iou", "recall", "recall_at_precision"]


def _compute_image_metric(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    metric: ImageMetric,
    recall_constraint: float = 0.95,
) -> float:
    """Compute image-level metric for a given threshold.

    Args:
        scores: Image anomaly scores.
        labels: Ground truth labels (0=normal, 1=anomaly).
        threshold: Classification threshold.
        metric: Metric to compute.
        recall_constraint: For 'fp_at_recall', target recall level.

    Returns:
        Metric value (higher is better, except fp_at_recall where lower is better).
    """
    preds = (scores >= threshold).astype(int)

    if metric == "f1":
        return float(f1_score(labels, preds, zero_division=0))
    elif metric == "precision":
        return float(precision_score(labels, preds, zero_division=0))
    elif metric == "recall":
        return float(recall_score(labels, preds, zero_division=0))
    elif metric == "fp_at_recall":
        # Return negative FP rate (so higher is better) if recall >= constraint
        recall = recall_score(labels, preds, zero_division=0)
        if recall >= recall_constraint:
            fp = np.sum((preds == 1) & (labels == 0))
            n_neg = np.sum(labels == 0)
            fp_rate = fp / n_neg if n_neg > 0 else 0.0
            return -fp_rate  # Negative so maximization minimizes FP
        return -1.0  # Penalize if recall constraint not met
    else:
        raise ValueError(f"Unknown image metric: {metric}")


def _compute_pixel_metric(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    metric: PixelMetric,
    precision_constraint: float = 0.5,
) -> float:
    """Compute pixel-level metric for a given threshold.

    Args:
        scores: Pixel anomaly scores.
        labels: Ground truth labels (0=normal, 1=anomaly).
        threshold: Segmentation threshold.
        metric: Metric to compute.
        precision_constraint: For 'recall_at_precision', target precision level.

    Returns:
        Metric value (higher is better).
    """
    preds = (scores >= threshold).astype(int)

    if metric == "f1":
        return float(f1_score(labels, preds, zero_division=0))
    elif metric == "iou":
        intersection = np.sum((preds == 1) & (labels == 1))
        union = np.sum((preds == 1) | (labels == 1))
        return intersection / union if union > 0 else 0.0
    elif metric == "recall":
        return float(recall_score(labels, preds, zero_division=0))
    elif metric == "recall_at_precision":
        prec = precision_score(labels, preds, zero_division=0)
        if prec >= precision_constraint:
            return float(recall_score(labels, preds, zero_division=0))
        return 0.0  # Penalize if precision constraint not met
    else:
        raise ValueError(f"Unknown pixel metric: {metric}")


def find_optimal_thresholds(
    *,
    image_scores: np.ndarray,
    image_labels: np.ndarray,
    pixel_scores: np.ndarray | None = None,
    pixel_labels: np.ndarray | None = None,
    image_metric: ImageMetric = "f1",
    pixel_metric: PixelMetric = "f1",
    recall_constraint: float = 0.95,
    precision_constraint: float = 0.5,
    n_candidates: int = 200,
) -> dict[str, Any]:
    """Find optimal thresholds by grid search on validation data.

    Args:
        image_scores: Image-level anomaly scores.
        image_labels: Image-level ground truth (0=normal, 1=anomaly).
        pixel_scores: Pixel-level anomaly scores (optional).
        pixel_labels: Pixel-level ground truth (optional).
        image_metric: Metric to optimize for image threshold.
        pixel_metric: Metric to optimize for pixel threshold.
        recall_constraint: Recall constraint for 'fp_at_recall' metric.
        precision_constraint: Precision constraint for 'recall_at_precision' metric.
        n_candidates: Number of threshold candidates to search.

    Returns:
        Dictionary with optimal thresholds, metric values, and search summary.
    """
    result: dict[str, Any] = {
        "image_threshold": 0.0,
        "image_metric_value": 0.0,
        "pixel_threshold": None,
        "pixel_metric_value": None,
        "search_summary": {},
    }

    # Find optimal image threshold
    if len(image_scores) > 0:
        img_min, img_max = float(np.min(image_scores)), float(np.max(image_scores))
        img_candidates = np.linspace(img_min, img_max, n_candidates)

        best_img_thresh, best_img_metric = img_min, -np.inf
        for thresh in img_candidates:
            val = _compute_image_metric(
                image_scores, image_labels, thresh, image_metric, recall_constraint
            )
            if val > best_img_metric:
                best_img_metric = val
                best_img_thresh = thresh

        result["image_threshold"] = float(best_img_thresh)
        # For fp_at_recall, convert back to positive FP rate for reporting
        if image_metric == "fp_at_recall":
            result["image_metric_value"] = abs(best_img_metric)
        else:
            result["image_metric_value"] = float(best_img_metric)

        result["search_summary"]["image"] = {
            "metric": image_metric,
            "score_range": [img_min, img_max],
            "n_candidates": n_candidates,
        }

    # Find optimal pixel threshold
    if pixel_scores is not None and pixel_labels is not None and len(pixel_scores) > 0:
        pix_min, pix_max = float(np.min(pixel_scores)), float(np.max(pixel_scores))
        pix_candidates = np.linspace(pix_min, pix_max, n_candidates)

        best_pix_thresh, best_pix_metric = pix_min, -np.inf
        for thresh in pix_candidates:
            val = _compute_pixel_metric(
                pixel_scores, pixel_labels, thresh, pixel_metric, precision_constraint
            )
            if val > best_pix_metric:
                best_pix_metric = val
                best_pix_thresh = thresh

        result["pixel_threshold"] = float(best_pix_thresh)
        result["pixel_metric_value"] = float(best_pix_metric)

        result["search_summary"]["pixel"] = {
            "metric": pixel_metric,
            "score_range": [pix_min, pix_max],
            "n_candidates": n_candidates,
        }

    return result

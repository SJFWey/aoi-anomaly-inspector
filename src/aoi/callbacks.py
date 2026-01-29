"""Lightning callbacks for prediction and metric collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    from anomalib.metrics import AUPRO
except ImportError:
    try:
        from anomalib.metrics.aupro import AUPRO
    except ImportError:
        AUPRO = None
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import roc_auc_score

from .config import dump_json
from .postprocess import postprocess_anomaly_map
from .viz import save_mask, save_overlay_with_mask

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from .thresholds import Thresholds

__all__ = [
    "JsonlPredictionWriter",
    "MetricCollector",
    "PostprocessPredictionWriter",
    "evaluate_and_write_metrics",
]

MetricsValue = float | int | str | list[int] | None
MetricsDict = dict[str, MetricsValue]


class JsonlPredictionWriter(Callback):
    """Callback to write predictions to a JSONL file during predict loop."""

    def __init__(
        self, *, out_path: Path, split: str, model: str, category: str
    ) -> None:
        """Initialize the prediction writer.

        Args:
            out_path: Path to the output JSONL file.
            split: Data split name ('train' or 'test').
            model: Model name.
            category: MVTec category name.
        """
        super().__init__()
        self.out_path = out_path
        self.split = split
        self.model = model
        self.category = category
        self._f = None

    def on_predict_start(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        del trainer, pl_module
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.out_path.open("w", encoding="utf-8")

    def on_predict_batch_end(  # noqa: ANN001
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        if self._f is None:
            return

        record_batch = outputs if outputs is not None else batch
        image_paths = getattr(record_batch, "image_path", None) or []
        gt_label = getattr(record_batch, "gt_label", None)
        pred_score = getattr(record_batch, "pred_score", None)
        anomaly_map = getattr(record_batch, "anomaly_map", None)

        if pred_score is None and anomaly_map is not None:
            pred_score = anomaly_map.amax(dim=(-2, -1))

        anomaly_max = (
            anomaly_map.amax(dim=(-2, -1)) if anomaly_map is not None else None
        )
        anomaly_mean = (
            anomaly_map.float().mean(dim=(-2, -1)) if anomaly_map is not None else None
        )

        for i, image_path in enumerate(image_paths):
            row = {
                "image_path": str(image_path),
                "split": self.split,
                "model": self.model,
                "category": self.category,
                "gt_label": int(gt_label[i].item()) if gt_label is not None else None,
                "pred_score": float(pred_score[i].item())
                if pred_score is not None
                else None,
                "anomaly_max": float(anomaly_max[i].item())
                if anomaly_max is not None
                else None,
                "anomaly_mean": float(anomaly_mean[i].item())
                if anomaly_mean is not None
                else None,
            }
            self._f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_predict_end(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        del trainer, pl_module
        if self._f is not None:
            self._f.close()
            self._f = None


class MetricCollector(Callback):
    """Callback to collect predictions during test loop for metric computation."""

    def __init__(self) -> None:
        """Initialize the metric collector."""
        super().__init__()
        self.image_scores: list[np.ndarray] = []
        self.image_labels: list[np.ndarray] = []
        self.pixel_scores: list[np.ndarray] = []
        self.pixel_labels: list[np.ndarray] = []
        # Keep 2D maps (not flattened) for AUPRO calculation
        self.pixel_scores_2d: list[torch.Tensor] = []
        self.pixel_labels_2d: list[torch.Tensor] = []

    def reset(self) -> None:
        """Clear all collected data."""
        self.image_scores.clear()
        self.image_labels.clear()
        self.pixel_scores.clear()
        self.pixel_labels.clear()
        self.pixel_scores_2d.clear()
        self.pixel_labels_2d.clear()

    def on_test_batch_end(  # noqa: ANN001
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        b = outputs if outputs is not None else batch
        gt_label = getattr(b, "gt_label", None)
        pred_score = getattr(b, "pred_score", None)
        anomaly_map = getattr(b, "anomaly_map", None)
        gt_mask = getattr(b, "gt_mask", None)

        if pred_score is None and anomaly_map is not None:
            pred_score = anomaly_map.amax(dim=(-2, -1))

        if gt_label is not None and pred_score is not None:
            self.image_labels.append(
                gt_label.detach().cpu().to(torch.int32).view(-1).numpy()
            )
            self.image_scores.append(
                pred_score.detach().cpu().to(torch.float32).view(-1).numpy()
            )

        if anomaly_map is not None and gt_mask is not None:
            self.pixel_scores.append(
                anomaly_map.detach().cpu().to(torch.float32).flatten().numpy()
            )
            self.pixel_labels.append(
                gt_mask.detach().cpu().to(torch.int32).flatten().numpy()
            )
            # Keep 2D maps for AUPRO (squeeze batch dim if needed)
            amap_2d = anomaly_map.detach().cpu().to(torch.float32)
            mask_2d = gt_mask.detach().cpu().to(torch.int32)
            self.pixel_scores_2d.append(amap_2d)
            self.pixel_labels_2d.append(mask_2d)

    def compute_auroc(self) -> MetricsDict:
        """Compute image AUROC, pixel AUROC, and AUPRO from collected data.

        Returns:
            Dictionary with 'image_AUROC', 'pixel_AUROC', and 'pixel_AUPRO' keys.
        """
        y_true = (
            np.concatenate(self.image_labels)
            if self.image_labels
            else np.array([], dtype=np.int32)
        )
        y_score = (
            np.concatenate(self.image_scores)
            if self.image_scores
            else np.array([], dtype=np.float32)
        )
        image_auroc = (
            float(roc_auc_score(y_true, y_score))
            if len(np.unique(y_true)) >= 2
            else None
        )

        p_true = (
            np.concatenate(self.pixel_labels)
            if self.pixel_labels
            else np.array([], dtype=np.int32)
        )
        p_score = (
            np.concatenate(self.pixel_scores)
            if self.pixel_scores
            else np.array([], dtype=np.float32)
        )
        pixel_auroc = (
            float(roc_auc_score(p_true, p_score))
            if len(np.unique(p_true)) >= 2
            else None
        )

        # Compute AUPRO using anomalib's metric
        pixel_aupro = self._compute_aupro()

        return {
            "image_AUROC": image_auroc,
            "pixel_AUROC": pixel_auroc,
            "pixel_AUPRO": pixel_aupro,
        }

    def _compute_aupro(self) -> float | None:
        """Compute AUPRO (Area Under Per-Region Overlap) metric.

        Returns:
            AUPRO value or None if insufficient data.
        """
        if AUPRO is None or not self.pixel_scores_2d or not self.pixel_labels_2d:
            return None

        try:
            # Initialize AUPRO metric from anomalib (fpr_limit=0.3 is standard)
            aupro_metric = AUPRO(fpr_limit=0.3)

            # Update metric with all collected 2D maps
            for preds, target in zip(
                self.pixel_scores_2d, self.pixel_labels_2d, strict=True
            ):
                # AUPRO expects preds and target with shape (N, H, W)
                # Squeeze channel dimension if present (N, 1, H, W) -> (N, H, W)
                if preds.dim() == 4 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)
                if target.dim() == 4 and target.shape[1] == 1:
                    target = target.squeeze(1)
                # preds should be float, target should be long (int64)
                aupro_metric.update(preds.float(), target.long())

            result = aupro_metric.compute()
            return float(result.item()) if result is not None else None
        except Exception:  # noqa: BLE001
            # Return None if AUPRO computation fails (e.g., no anomaly regions)
            return None


def evaluate_and_write_metrics(
    *,
    trainer: Trainer,
    model,  # noqa: ANN001
    dataloader: "DataLoader",
    out_path: Path,
    ckpt_path: Path | str | None = None,
    category: str | None = None,
    image_size: int | tuple[int, int] | list[int] | None = None,
    device: str | None = None,
) -> MetricsDict:
    """Run test loop, compute AUROC metrics, and write to JSON file.

    Args:
        trainer: PyTorch Lightning Trainer instance.
        model: Model to evaluate.
        dataloader: Test DataLoader.
        out_path: Path to write metrics JSON file.
        ckpt_path: Optional checkpoint path to load before testing.
        category: MVTec category name (written to metrics.json).
        image_size: Input image size as (H, W) (written to metrics.json).
        device: Inference device string, e.g. 'cpu' or 'cuda' (written to metrics.json).

    Returns:
        Dictionary with computed metrics and metadata.
    """
    collector = MetricCollector()
    trainer.callbacks.append(collector)  # type: ignore[attr-defined]

    if ckpt_path is not None:
        trainer.test(model=model, dataloaders=dataloader, ckpt_path=str(ckpt_path))
    else:
        trainer.test(model=model, dataloaders=dataloader)

    trainer.callbacks.remove(collector)  # type: ignore[attr-defined]

    metrics: MetricsDict = dict(collector.compute_auroc())
    # Add metadata required by milestone 3
    if category is not None:
        metrics["category"] = category
    if image_size is not None:
        # Allow square sizes like `256` in configs; normalize to [H, W].
        metrics["image_size"] = (
            [int(image_size), int(image_size)]
            if isinstance(image_size, int)
            else list(image_size)
        )
    if device is not None:
        metrics["device"] = device
    dump_json(out_path, metrics)
    return metrics


class PostprocessPredictionWriter(Callback):
    """Callback to write predictions with post-processing during predict loop.

    Generates:
    - preds.jsonl: Structured predictions with OK/NG label and defect info
    - masks/: Binary defect masks (optional)
    - overlays/: Overlay visualizations (optional)

    Uses image_threshold for OK/NG decision and pixel_threshold for mask generation.
    All coordinates use the model input image size coordinate system.
    """

    def __init__(
        self,
        *,
        thresholds: "Thresholds",
        output_dir: Path,
        model: str,
        category: str,
        image_size: tuple[int, int],
        save_masks: bool = True,
        save_overlays: bool = True,
        min_defect_area: int = 100,
        overlay_alpha: float = 0.4,
        apply_morphology: bool = True,
        morph_kernel_size: int = 7,
        apply_blur: bool = True,
        blur_kernel_size: int = 7,
    ) -> None:
        """Initialize the post-process prediction writer.

        Args:
            thresholds: Thresholds object with image_threshold and pixel_threshold.
            output_dir: Directory to write outputs (preds.jsonl, masks/, overlays/).
            model: Model name for metadata.
            category: Category name for metadata.
            image_size: Model input image size as (H, W) for metadata.
            save_masks: Whether to save binary mask images.
            save_overlays: Whether to save overlay visualizations.
            min_defect_area: Minimum area for defect filtering in connected components.
            overlay_alpha: Transparency for overlay heatmap.
            apply_morphology: Whether to apply morphological operations (opening/closing).
            morph_kernel_size: Kernel size for morphological operations.
            apply_blur: Whether to apply Gaussian blur before thresholding.
            blur_kernel_size: Kernel size for Gaussian blur.
        """
        super().__init__()
        self.thresholds = thresholds
        self.output_dir = output_dir
        self.model = model
        self.category = category
        self.image_size = image_size
        self.save_masks = save_masks
        self.save_overlays = save_overlays
        self.min_defect_area = min_defect_area
        self.overlay_alpha = overlay_alpha
        self.apply_morphology = apply_morphology
        self.morph_kernel_size = morph_kernel_size
        self.apply_blur = apply_blur
        self.blur_kernel_size = blur_kernel_size

        self._preds_path = output_dir / "preds.jsonl"
        self._masks_dir = output_dir / "masks"
        self._overlays_dir = output_dir / "overlays"
        self._f = None
        self._count = 0

    def on_predict_start(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        del trainer, pl_module
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_masks:
            self._masks_dir.mkdir(parents=True, exist_ok=True)
        if self.save_overlays:
            self._overlays_dir.mkdir(parents=True, exist_ok=True)
        self._f = self._preds_path.open("w", encoding="utf-8")
        self._count = 0

    def on_predict_batch_end(  # noqa: ANN001
        self,
        trainer: Trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        if self._f is None:
            return

        record_batch = outputs if outputs is not None else batch
        image_paths = getattr(record_batch, "image_path", None) or []
        gt_label = getattr(record_batch, "gt_label", None)
        pred_score = getattr(record_batch, "pred_score", None)
        anomaly_map = getattr(record_batch, "anomaly_map", None)

        # Try to get original image for overlay: prefer original_image, fallback to image
        images = getattr(record_batch, "original_image", None)
        if images is None:
            images = getattr(record_batch, "image", None)

        # Fallback: derive pred_score from anomaly_map max
        if pred_score is None and anomaly_map is not None:
            pred_score = anomaly_map.amax(dim=(-2, -1))

        for i, image_path in enumerate(image_paths):
            self._count += 1
            image_name = Path(image_path).stem
            out_name = f"{self._count:06d}_{image_name}"

            # Get anomaly map for this image
            amap_np = None
            if anomaly_map is not None:
                amap_tensor = anomaly_map[i]
                amap_np = amap_tensor.detach().cpu().to(torch.float32).numpy()
                # Remove batch/channel dims if present, keep (H, W)
                if amap_np.ndim == 3 and amap_np.shape[0] == 1:
                    amap_np = amap_np[0]

            # Get image score
            score = None
            if pred_score is not None:
                score = float(pred_score[i].item())

            # OK/NG decision using image_threshold
            is_anomaly = False
            if score is not None:
                is_anomaly = score >= self.thresholds.image_threshold
            label = "NG" if is_anomaly else "OK"

            # Post-process anomaly map
            postprocess_result = None
            if amap_np is not None:
                postprocess_result = postprocess_anomaly_map(
                    amap_np,
                    pixel_threshold=self.thresholds.pixel_threshold,
                    min_defect_area=self.min_defect_area,
                    apply_morphology=self.apply_morphology,
                    morph_kernel_size=self.morph_kernel_size,
                    apply_blur=self.apply_blur,
                    blur_kernel_size=self.blur_kernel_size,
                )

            # Build prediction record
            row = {
                "image_path": str(image_path),
                "model": self.model,
                "category": self.category,
                "image_size": list(self.image_size),
                "gt_label": int(gt_label[i].item()) if gt_label is not None else None,
                "pred_score": score,
                "image_threshold": self.thresholds.image_threshold,
                "pixel_threshold": self.thresholds.pixel_threshold,
                "label": label,
                "is_anomaly": is_anomaly,
            }

            if postprocess_result is not None:
                row["num_defects"] = postprocess_result.num_defects
                row["total_defect_area"] = postprocess_result.total_defect_area
                row["defects"] = [d.to_dict() for d in postprocess_result.defects]

                # Save mask
                if self.save_masks:
                    mask_path = self._masks_dir / f"{out_name}.png"
                    save_mask(postprocess_result.mask, mask_path)
                    row["mask_path"] = str(mask_path.relative_to(self.output_dir))

                # Save overlay
                if self.save_overlays and images is not None:
                    img_tensor = images[i]
                    img_np = img_tensor.detach().cpu().to(torch.float32).numpy()
                    # Convert from (C, H, W) to (H, W, C)
                    if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):
                        img_np = np.transpose(img_np, (1, 2, 0))
                    # Convert to uint8
                    img_min = float(img_np.min()) if img_np.size else 0.0
                    img_max = float(img_np.max()) if img_np.size else 0.0
                    if 0.0 <= img_min and img_max <= 1.0:
                        img_np = (img_np * 255.0).round()
                    elif 0.0 <= img_min and img_max <= 255.0:
                        img_np = np.clip(img_np, 0.0, 255.0)
                    else:
                        if img_max > img_min:
                            img_np = (img_np - img_min) / (img_max - img_min)
                        else:
                            img_np = np.zeros_like(img_np)
                        img_np = np.clip(img_np, 0.0, 1.0) * 255.0
                    img_np = img_np.astype(np.uint8)
                    # Handle single channel
                    if img_np.ndim == 2:
                        img_np = np.stack([img_np] * 3, axis=-1)
                    elif img_np.shape[-1] == 1:
                        img_np = np.concatenate([img_np] * 3, axis=-1)

                    overlay_path = self._overlays_dir / f"{out_name}.png"
                    save_overlay_with_mask(
                        img_np,
                        amap_np,
                        postprocess_result.mask,
                        overlay_path,
                        alpha=self.overlay_alpha,
                    )
                    row["overlay_path"] = str(overlay_path.relative_to(self.output_dir))

            self._f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_predict_end(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        del trainer, pl_module
        if self._f is not None:
            self._f.close()
            self._f = None

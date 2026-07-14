from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from aoi.artifacts import RunPaths, sha256_file, update_manifest, write_json_atomic, write_jsonl_atomic
from aoi.config import load_experiment_config
from aoi.data import ImageSample, load_mvtec_split, read_bgr, read_mask, split_normal_samples
from aoi.modeling import AnomalyModel, load_model
from aoi.postprocess import postprocess_map
from aoi.reporting import now_utc, summarize_records
from aoi.thresholds import fit_thresholds
from aoi.viz import save_diagnostic_composite, save_overlay


def safe_auroc(targets: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float | None:
    labels = np.asarray(targets)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, np.asarray(scores)))


def binary_metrics(targets: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> dict[str, float | int]:
    target = np.asarray(targets, dtype=np.uint8)
    prediction = np.asarray(predictions, dtype=np.uint8)
    true_positive = int(np.sum((target == 1) & (prediction == 1)))
    true_negative = int(np.sum((target == 0) & (prediction == 0)))
    false_positive = int(np.sum((target == 0) & (prediction == 1)))
    false_negative = int(np.sum((target == 1) & (prediction == 0)))
    total = len(target)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _record_for(
    sample: ImageSample,
    model: AnomalyModel,
    config: Any,
    paths: RunPaths,
    image_threshold: float,
    pixel_threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    prediction = model.predict_path(sample.image_path)
    image = read_bgr(sample.image_path)
    post = postprocess_map(
        prediction.anomaly_map,
        pixel_threshold=pixel_threshold,
        output_shape=image.shape[:2],
        min_area=config.min_area,
        morph_kernel=config.morph_kernel,
    )
    defects = [{"bbox": defect.bbox, "area": defect.area, "centroid": defect.centroid} for defect in post.defects]
    output_name = f"{sample.label}__{sample.image_path.name}"
    mask_path = paths.masks / output_name
    overlay_path = paths.overlays / output_name
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), post.mask):
        raise OSError(f"Failed to write prediction mask: {mask_path}")
    save_overlay(overlay_path, image, prediction.anomaly_map, defects)
    return (
        {
            "file": str(sample.image_path),
            "file_name": sample.image_path.name,
            "model": config.model_name,
            "category": config.category,
            "run_id": paths.root.name,
            "label": sample.label,
            "label_index": sample.label_index,
            "gt_mask_path": str(sample.mask_path) if sample.mask_path else None,
            "overlay_path": str(overlay_path),
            "mask_path": str(mask_path),
            "image_score": prediction.image_score,
            "decision": "NG" if prediction.image_score > image_threshold else "OK",
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
            "defect_count": len(defects),
            "max_defect_area": max((defect["area"] for defect in defects), default=0),
            "defects": defects,
            "latency_ms": prediction.latency_ms,
        },
        prediction.anomaly_map,
        post.mask,
        read_mask(sample.mask_path, image.shape[:2]),
    )


def _write_samples(
    records: list[dict[str, Any]], maps: list[np.ndarray], masks: list[np.ndarray], ground_truth: list[np.ndarray], paths: RunPaths
) -> None:
    for index, (record, anomaly_map, mask, gt_mask) in enumerate(zip(records[:3], maps[:3], masks[:3], ground_truth[:3], strict=True), start=1):
        image = read_bgr(Path(record["file"]))
        save_diagnostic_composite(
            paths.samples / f"{index:02d}_{record['label']}_{record['decision']}.png",
            image,
            anomaly_map,
            mask,
            gt_mask,
            f"{record['model']} {record['category']} label={record['label']} decision={record['decision']}",
            float(record["pixel_threshold"]),
        )


def evaluate_run(run_dir: Path, *, device: str = "cpu") -> RunPaths:
    paths = RunPaths(run_dir)
    if not paths.model.exists():
        raise FileNotFoundError(paths.model)
    if not paths.config.exists():
        raise FileNotFoundError(paths.config)
    config = load_experiment_config(paths.config, device=device)
    model = load_model(paths.model, device=device)
    train_good = [
        sample for sample in load_mvtec_split(config.data_root, config.category, "train") if sample.label == "good"
    ]
    if not train_good:
        raise RuntimeError("No train/good images found for threshold calibration")
    normal_split = split_normal_samples(train_good, config.calibration_fraction, config.seed)
    train_predictions = [model.predict_path(sample.image_path) for sample in normal_split.calibration]
    thresholds = fit_thresholds(
        [prediction.image_score for prediction in train_predictions],
        [prediction.anomaly_map for prediction in train_predictions],
        quantile_image=config.quantile_image,
        quantile_pixel=config.quantile_pixel,
    )
    thresholds_data = {
        "image_threshold": thresholds.image_threshold,
        "pixel_threshold": thresholds.pixel_threshold,
        "image_quantile": thresholds.quantile_image,
        "pixel_quantile": thresholds.quantile_pixel,
        "calibration_fraction": config.calibration_fraction,
        "calibration_images": len(normal_split.calibration),
        "calibration_source": (
            "shared_train_good_fallback" if normal_split.shares_samples else "held_out_train_good"
        ),
        "created_at": now_utc(),
    }
    write_json_atomic(paths.thresholds, thresholds_data)

    records: list[dict[str, Any]] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    for sample in load_mvtec_split(config.data_root, config.category, "test"):
        record, anomaly_map, mask, gt_mask = _record_for(
            sample, model, config, paths, thresholds.image_threshold, thresholds.pixel_threshold
        )
        records.append(record)
        maps.append(anomaly_map)
        masks.append(mask)
        ground_truth.append(gt_mask)

    image_targets = [record["label_index"] for record in records]
    image_scores = [record["image_score"] for record in records]
    image_predictions = [int(record["decision"] == "NG") for record in records]
    pixel_targets = np.concatenate([(mask > 0).astype(np.uint8).reshape(-1) for mask in ground_truth])
    pixel_predictions = np.concatenate([(mask > 0).astype(np.uint8).reshape(-1) for mask in masks])
    pixel_scores = np.concatenate(
        [
            cv2.resize(anomaly_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(-1)
            for anomaly_map, mask in zip(maps, masks, strict=True)
        ]
    )
    pixel_metrics = binary_metrics(pixel_targets, pixel_predictions)
    intersection = int(np.sum((pixel_targets == 1) & (pixel_predictions == 1)))
    union = int(np.sum((pixel_targets == 1) | (pixel_predictions == 1)))
    pixel_metrics["iou"] = intersection / union if union else 0.0
    warnings: list[str] = []
    image_auroc = safe_auroc(image_targets, image_scores)
    pixel_auroc = safe_auroc(pixel_targets, pixel_scores)
    if image_auroc is None:
        warnings.append("image AUROC is undefined because only one class is present")
    if pixel_auroc is None:
        warnings.append("pixel AUROC is undefined because only one class is present")
    metrics = {
        "model": config.model_name,
        "category": config.category,
        "run_id": paths.root.name,
        "created_at": now_utc(),
        "image": {"auroc": image_auroc, **binary_metrics(image_targets, image_predictions)},
        "pixel": {"auroc": pixel_auroc, **pixel_metrics},
        "warnings": warnings,
    }
    write_json_atomic(paths.metrics, metrics)
    write_jsonl_atomic(paths.preds, records)
    report = summarize_records(records, config.model_name, config.category, paths.root.name)
    report.update({"thresholds": thresholds_data, "metrics": metrics, "warnings": warnings})
    write_json_atomic(paths.report, report)
    _write_samples(records, maps, masks, ground_truth, paths)
    update_manifest(
        paths.manifest,
        {
            "evaluation": {
                "status": "complete",
                "created_at": now_utc(),
                "thresholds_sha256": sha256_file(paths.thresholds),
                "metrics_sha256": sha256_file(paths.metrics),
                "predictions_sha256": sha256_file(paths.preds),
            }
        },
    )
    return paths

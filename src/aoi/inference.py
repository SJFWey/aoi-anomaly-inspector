from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from aoi.artifacts import RunPaths, write_json_atomic, write_jsonl_atomic
from aoi.config import load_experiment_config
from aoi.data import collect_image_paths, read_bgr
from aoi.modeling import load_model
from aoi.postprocess import postprocess_map
from aoi.reporting import summarize_records
from aoi.viz import save_overlay as write_overlay


def predict_directory(
    run_dir: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    save_mask: bool = True,
    save_overlay: bool = True,
) -> list[dict[str, Any]]:
    run = RunPaths(run_dir)
    for required in (run.config, run.model, run.thresholds):
        if not required.exists():
            raise FileNotFoundError(required)
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {input_dir}")
    config = load_experiment_config(run.config, device=device)
    thresholds = json.loads(run.thresholds.read_text(encoding="utf-8"))
    model = load_model(run.model, device=device)
    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        prediction = model.predict_path(image_path)
        image = read_bgr(image_path)
        post = postprocess_map(
            prediction.anomaly_map,
            pixel_threshold=float(thresholds["pixel_threshold"]),
            output_shape=image.shape[:2],
            min_area=config.min_area,
            morph_kernel=config.morph_kernel,
        )
        relative = image_path.relative_to(input_dir)
        output_name = "__".join(relative.parts)
        mask_path = output_dir / "masks" / output_name
        overlay_path = output_dir / "overlays" / output_name
        defects = [{"bbox": defect.bbox, "area": defect.area, "centroid": defect.centroid} for defect in post.defects]
        if save_mask:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), post.mask):
                raise OSError(f"Failed to write prediction mask: {mask_path}")
        if save_overlay:
            write_overlay(overlay_path, image, prediction.anomaly_map, defects)
        label = image_path.parent.name if image_path.parent != input_dir else "unknown"
        records.append(
            {
                "file": str(image_path),
                "relative_file": str(relative),
                "model": config.model_name,
                "category": config.category,
                "run_id": run.root.name,
                "label": label,
                "image_score": prediction.image_score,
                "decision": "NG" if prediction.image_score > float(thresholds["image_threshold"]) else "OK",
                "image_threshold": float(thresholds["image_threshold"]),
                "pixel_threshold": float(thresholds["pixel_threshold"]),
                "mask_path": str(mask_path) if save_mask else None,
                "overlay_path": str(overlay_path) if save_overlay else None,
                "defect_count": len(defects),
                "max_defect_area": max((defect["area"] for defect in defects), default=0),
                "defects": defects,
                "latency_ms": prediction.latency_ms,
            }
        )
    write_jsonl_atomic(output_dir / "preds.jsonl", records)
    report = summarize_records(records, config.model_name, config.category, run.root.name)
    report["thresholds"] = thresholds
    write_json_atomic(output_dir / "report.json", report)
    return records

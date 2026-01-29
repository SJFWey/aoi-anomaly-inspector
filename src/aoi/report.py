"""Report generation utilities for AOI pipeline.

This module provides functions to generate summary reports from prediction
results (preds.jsonl).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["generate_report", "load_predictions"]


def load_predictions(preds_path: Path | str) -> list[dict[str, Any]]:
    """Load predictions from a JSONL file.

    Args:
        preds_path: Path to the preds.jsonl file.

    Returns:
        List of prediction dictionaries.

    Raises:
        FileNotFoundError: If the predictions file does not exist.
    """
    preds_path = Path(preds_path)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    predictions = []
    with preds_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    return predictions


def generate_report(
    predictions: list[dict[str, Any]],
    *,
    model: str | None = None,
    category: str | None = None,
    run_id: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate a summary report from predictions.

    Args:
        predictions: List of prediction dictionaries (from preds.jsonl).
        model: Model name (overrides value from predictions if provided).
        category: Category name (overrides value from predictions if provided).
        run_id: Run identifier.
        thresholds: Dictionary with image_threshold and pixel_threshold.

    Returns:
        Report dictionary with summary statistics.
    """
    if not predictions:
        return {
            "model": model,
            "category": category,
            "run_id": run_id,
            "thresholds": thresholds,
            "num_images": 0,
            "num_ok": 0,
            "num_ng": 0,
            "ng_rate": 0.0,
            "ok_rate": 0.0,
            "defect_stats": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # Extract model/category from first prediction if not provided
    first_pred = predictions[0]
    if model is None:
        model = first_pred.get("model", "unknown")
    if category is None:
        category = first_pred.get("category", "unknown")
    if thresholds is None:
        thresholds = {
            "image_threshold": float(first_pred.get("image_threshold", 0.0)),
            "pixel_threshold": float(first_pred.get("pixel_threshold", 0.0)),
        }

    # Count OK/NG
    num_images = len(predictions)
    num_ng = sum(1 for p in predictions if p.get("is_anomaly", False))
    num_ok = num_images - num_ng
    ng_rate = num_ng / num_images if num_images > 0 else 0.0
    ok_rate = num_ok / num_images if num_images > 0 else 0.0

    # Defect statistics (for NG images only)
    ng_predictions = [p for p in predictions if p.get("is_anomaly", False)]
    defect_stats = None
    if ng_predictions:
        total_defects = sum(p.get("num_defects", 0) for p in ng_predictions)
        total_defect_areas = [
            p.get("total_defect_area", 0)
            for p in ng_predictions
            if p.get("total_defect_area") is not None
        ]

        # Collect all defect areas for max calculation
        all_defect_areas = []
        for p in ng_predictions:
            defects = p.get("defects", [])
            for d in defects:
                area = d.get("area", 0)
                if area > 0:
                    all_defect_areas.append(area)

        defect_stats = {
            "total_defects": total_defects,
            "avg_defects_per_ng": total_defects / len(ng_predictions)
            if ng_predictions
            else 0.0,
            "max_defect_area": max(all_defect_areas) if all_defect_areas else 0,
            "avg_total_defect_area": (
                sum(total_defect_areas) / len(total_defect_areas)
                if total_defect_areas
                else 0.0
            ),
        }

    # Score statistics
    scores: list[float] = [
        float(p["pred_score"]) for p in predictions if p.get("pred_score") is not None
    ]
    score_stats = None
    if scores:
        score_stats = {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
        }

    report = {
        "model": model,
        "category": category,
        "run_id": run_id,
        "thresholds": thresholds,
        "num_images": num_images,
        "num_ok": num_ok,
        "num_ng": num_ng,
        "ng_rate": round(ng_rate, 4),
        "ok_rate": round(ok_rate, 4),
        "score_stats": score_stats,
        "defect_stats": defect_stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return report


def write_report(report: dict[str, Any], output_path: Path | str) -> None:
    """Write a report to a JSON file.

    Args:
        report: Report dictionary to write.
        output_path: Path to the output JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

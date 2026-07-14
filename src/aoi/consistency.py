from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from aoi.artifacts import RunPaths, update_manifest, write_json_atomic
from aoi.config import load_experiment_config
from aoi.data import bgr_to_rgb_unit_tensor, collect_image_paths, read_bgr
from aoi.modeling import load_model
from aoi.reporting import now_utc


def compare_outputs(native: np.ndarray, exported: np.ndarray) -> dict[str, float]:
    if native.shape != exported.shape:
        raise ValueError(f"Output shape mismatch: native={native.shape}, ONNX={exported.shape}")
    if not np.isfinite(native).all() or not np.isfinite(exported).all():
        raise ValueError("Cannot compare non-finite native or ONNX outputs")
    absolute = np.abs(native.astype(np.float64) - exported.astype(np.float64))
    return {"max_abs_error": float(absolute.max()), "mean_abs_error": float(absolute.mean())}


def enforce_tolerances(
    metrics: dict[str, float], *, max_abs_tolerance: float, mean_abs_tolerance: float
) -> None:
    values = np.asarray([metrics["max_abs_error"], metrics["mean_abs_error"]], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("consistency metrics contain non-finite values")
    if metrics["max_abs_error"] > max_abs_tolerance or metrics["mean_abs_error"] > mean_abs_tolerance:
        raise RuntimeError(
            "consistency tolerance exceeded: "
            f"max={metrics['max_abs_error']} (limit {max_abs_tolerance}), "
            f"mean={metrics['mean_abs_error']} (limit {mean_abs_tolerance})"
        )


def compare_decisions(
    native_scores: np.ndarray,
    exported_scores: np.ndarray,
    threshold: float,
    ambiguity_tolerance: float,
) -> dict[str, float | int | None]:
    native_decision = native_scores > threshold
    exported_decision = exported_scores > threshold
    agreement = native_decision == exported_decision
    ambiguous = (np.abs(native_scores - threshold) <= ambiguity_tolerance) | (
        np.abs(exported_scores - threshold) <= ambiguity_tolerance
    )
    stable = ~ambiguous
    return {
        "decision_agreement": float(np.mean(agreement)),
        "stable_decision_agreement": float(np.mean(agreement[stable])) if np.any(stable) else None,
        "ambiguous_decisions": int(np.sum(ambiguous)),
        "stable_decisions": int(np.sum(stable)),
        "decision_ambiguity_tolerance": ambiguity_tolerance,
    }


def enforce_decision_agreement(agreement: float | None) -> None:
    if agreement is None:
        return
    if not np.isfinite(agreement) or not 0.0 <= agreement <= 1.0:
        raise RuntimeError(f"invalid decision agreement: {agreement}")
    if agreement != 1.0:
        raise RuntimeError(f"native/ONNX decisions disagree: agreement={agreement}")


def check_run_consistency(
    run_dir: Path,
    input_dir: Path,
    *,
    device: str = "cpu",
    max_images: int = 8,
) -> dict[str, object]:
    paths = RunPaths(run_dir)
    for required in (paths.config, paths.model, paths.onnx):
        if not required.exists():
            raise FileNotFoundError(required)
    image_paths = collect_image_paths(input_dir)[:max_images]
    if not image_paths:
        raise RuntimeError(f"No images found under {input_dir}")
    config = load_experiment_config(paths.config, device=device)
    batch = torch.stack([bgr_to_rgb_unit_tensor(read_bgr(path), config.image_size) for path in image_paths])
    model = load_model(paths.model, device=device).tensor_model
    with torch.inference_mode():
        native_map, native_score = model(batch.to(device))
    session = ort.InferenceSession(str(paths.onnx), providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["image"]:
        raise RuntimeError("ONNX input contract does not expose only 'image'")
    if [item.name for item in session.get_outputs()] != ["anomaly_map", "pred_score"]:
        raise RuntimeError("ONNX output contract does not expose anomaly_map and pred_score")
    exported_map, exported_score = session.run(None, {"image": batch.numpy()})
    map_metrics = compare_outputs(native_map.cpu().numpy(), exported_map)
    score_metrics = compare_outputs(native_score.cpu().numpy(), exported_score)
    result: dict[str, Any] = {
        "created_at": now_utc(),
        "num_images": len(image_paths),
        "files": [str(path) for path in image_paths],
        "anomaly_map": map_metrics,
        "pred_score": score_metrics,
        "tolerances": {
            "max_abs_error": config.consistency_max_abs_error,
            "mean_abs_error": config.consistency_mean_abs_error,
        },
        "passed": False,
    }
    if paths.thresholds.exists():
        thresholds = json.loads(paths.thresholds.read_text(encoding="utf-8"))
        result.update(
            compare_decisions(
                native_score.cpu().numpy(),
                exported_score,
                float(thresholds["image_threshold"]),
                config.consistency_max_abs_error,
            )
        )
    write_json_atomic(paths.consistency, result)
    enforce_tolerances(
        map_metrics,
        max_abs_tolerance=config.consistency_max_abs_error,
        mean_abs_tolerance=config.consistency_mean_abs_error,
    )
    enforce_tolerances(
        score_metrics,
        max_abs_tolerance=config.consistency_max_abs_error,
        mean_abs_tolerance=config.consistency_mean_abs_error,
    )
    if "stable_decision_agreement" in result:
        enforce_decision_agreement(result["stable_decision_agreement"])
    result["passed"] = True
    write_json_atomic(paths.consistency, result)
    update_manifest(paths.manifest, {"consistency": {"status": "complete", "created_at": now_utc()}})
    return result

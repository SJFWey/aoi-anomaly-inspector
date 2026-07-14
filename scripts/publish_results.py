from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.artifacts import RunPaths, git_commit, sha256_file, write_json_atomic
from aoi.config import ExperimentConfig, load_experiment_config
from aoi.data import read_bgr, read_mask
from aoi.modeling import load_model
from aoi.postprocess import postprocess_map
from aoi.viz import save_diagnostic_composite


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_cases(records: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    if not records:
        raise ValueError("Cannot select cases from an empty prediction file")
    threshold = float(records[0]["image_threshold"])
    normal = [record for record in records if record["label"] == "good"]
    detected = [
        record for record in records if record["label"] != "good" and record["decision"] == "NG"
    ]
    anomalies = [record for record in records if record["label"] != "good"]
    if not normal or not detected or len(anomalies) < 2:
        raise ValueError("Publishing requires normal, detected-anomaly, and hard-anomaly candidates")

    normal_case = min(normal, key=lambda record: abs(float(record["image_score"]) - threshold))
    detected_case = max(detected, key=lambda record: float(record["image_score"]) - threshold)
    remaining = [record for record in anomalies if record["file"] != detected_case["file"]]
    hard_case = min(remaining, key=lambda record: abs(float(record["image_score"]) - threshold))
    normal_kind = "normal_ok" if normal_case["decision"] == "OK" else "normal_false_positive"
    hard_kind = "false_negative" if hard_case["decision"] == "OK" else "low_margin_true_positive"
    return [
        ("ok", normal_kind, normal_case),
        ("ng", "clear_true_positive", detected_case),
        ("hard", hard_kind, hard_case),
    ]


def _config_summary(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "backbone": config.backbone,
        "calibration_fraction": config.calibration_fraction,
        "coreset_sampling_ratio": config.coreset_sampling_ratio,
        "device": config.device,
        "image_size": config.image_size,
        "layers": list(config.layers),
        "max_coreset_patches": config.max_coreset_patches,
        "n_features": config.n_features,
        "seed": config.seed,
    }


def _publish_run(run_dir: Path, assets_dir: Path) -> tuple[str, dict[str, Any], str]:
    paths = RunPaths(run_dir)
    required = [
        paths.config,
        paths.model,
        paths.meta,
        paths.thresholds,
        paths.metrics,
        paths.preds,
        paths.onnx,
        paths.consistency,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    config = load_experiment_config(paths.config, device="cpu")
    meta = _read_json(paths.meta)
    source_commit = str(meta["source_git_commit"])
    current_commit = git_commit(REPO_ROOT)
    if source_commit != current_commit:
        raise RuntimeError(
            f"Run {run_dir} was produced by {source_commit}, but the current commit is {current_commit}"
        )
    if sha256_file(paths.model) != meta["model_sha256"]:
        raise RuntimeError(f"Model hash does not match metadata: {paths.model}")

    export_meta = _read_json(paths.export_meta)
    if sha256_file(paths.onnx) != export_meta["onnx_sha256"]:
        raise RuntimeError(f"ONNX hash does not match export metadata: {paths.onnx}")

    metrics = _read_json(paths.metrics)
    thresholds = _read_json(paths.thresholds)
    consistency = _read_json(paths.consistency)
    stable_agreement = consistency.get("stable_decision_agreement")
    if not consistency.get("passed") or stable_agreement not in {None, 1.0}:
        raise RuntimeError(f"Run has not passed numerical and decision consistency: {run_dir}")

    model = load_model(paths.model, device="cpu")
    selected: list[dict[str, Any]] = []
    for slot, case, record in _select_cases(_read_jsonl(paths.preds)):
        image_path = Path(record["file"])
        image = read_bgr(image_path)
        prediction = model.predict_path(image_path)
        post = postprocess_map(
            prediction.anomaly_map,
            pixel_threshold=float(thresholds["pixel_threshold"]),
            output_shape=image.shape[:2],
            min_area=config.min_area,
            morph_kernel=config.morph_kernel,
        )
        gt_mask = read_mask(
            Path(record["gt_mask_path"]) if record["gt_mask_path"] else None,
            image.shape[:2],
        )
        asset = assets_dir / f"{config.model_name}_{config.category}_{slot}.png"
        save_diagnostic_composite(
            asset,
            image,
            prediction.anomaly_map,
            post.mask,
            gt_mask,
            f"{config.model_name} {config.category} label={record['label']} decision={record['decision']}",
            float(thresholds["pixel_threshold"]),
        )
        selected.append(
            {
                "asset": str(asset.relative_to(REPO_ROOT)),
                "asset_sha256": sha256_file(asset),
                "case": case,
                "decision": record["decision"],
                "file": str(image_path.relative_to(config.data_root)),
                "image_score": record["image_score"],
                "image_threshold": record["image_threshold"],
                "label": record["label"],
                "slot": slot,
            }
        )

    consistency.pop("created_at", None)
    consistency.pop("files", None)
    category_result = {
        "config": _config_summary(config),
        "consistency": consistency,
        "metrics": metrics,
        "model_sha256": sha256_file(paths.model),
        "onnx_sha256": sha256_file(paths.onnx),
        "selected_cases": selected,
        "thresholds": thresholds,
    }
    return config.category, category_result, source_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the tracked verified-results JSON and diagnostic composites."
    )
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "docs" / "assets" / "results" / "verified_results.json",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "assets" / "results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    categories: dict[str, Any] = {}
    source_commits: set[str] = set()
    for run_dir in args.run_dir:
        category, result, source_commit = _publish_run(run_dir, args.assets_dir)
        if category in categories:
            raise ValueError(f"Duplicate category: {category}")
        categories[category] = result
        source_commits.add(source_commit)
    if len(source_commits) != 1:
        raise RuntimeError(f"Published runs must share one source commit: {sorted(source_commits)}")

    output = {
        "categories": dict(sorted(categories.items())),
        "dataset": "MVTec AD",
        "execution": "CPU",
        "model": "patchcore",
        "selection_policy": {
            "clear_anomaly": "highest positive margin above the image threshold",
            "hard_anomaly": "smallest absolute margin among remaining anomalous images",
            "normal": "smallest absolute margin to the image threshold",
        },
        "source_git_commit": next(iter(source_commits)),
        "workflow": "scripts/run_pipeline.py + scripts/publish_results.py",
    }
    write_json_atomic(args.output_json, output)
    print(f"Published {len(categories)} categories to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run management utilities."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["RunPaths", "make_run_paths", "now_run_id"]


@dataclass(frozen=True)
class RunPaths:
    """Data class holding all run output paths."""

    run_dir: Path
    config_path: Path
    meta_path: Path
    weights_path: Path
    train_preds_path: Path
    test_preds_path: Path
    metrics_path: Path


def now_run_id() -> str:
    """Generate a timestamp-based run ID.

    Returns:
        Run ID in the format 'YYYYMMDD_HHMMSS' (UTC).
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_paths(cfg: dict[str, Any]) -> RunPaths:
    """Create RunPaths from configuration.

    Args:
        cfg: Full configuration dictionary containing 'run', 'model', and 'data' sections.

    Returns:
        RunPaths instance with all output paths configured.
    """
    run_cfg = cfg.get("run", {})
    model_name = str(cfg.get("model", {}).get("name", "")).lower().strip()
    category = str(cfg.get("data", {}).get("category", "transistor"))

    output_root = Path(str(run_cfg.get("output_root", "runs")))
    run_id = run_cfg.get("run_id") or now_run_id()
    run_id = str(run_id)

    run_dir = output_root / model_name / category / run_id
    return RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        meta_path=run_dir / "meta.json",
        weights_path=run_dir / "weights" / "model.ckpt",
        train_preds_path=run_dir / "preds_train.jsonl",
        test_preds_path=run_dir / "preds_test.jsonl",
        metrics_path=run_dir / "metrics.json",
    )

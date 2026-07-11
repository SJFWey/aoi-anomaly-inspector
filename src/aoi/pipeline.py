from __future__ import annotations

from pathlib import Path

from aoi.artifacts import RunPaths
from aoi.consistency import check_run_consistency
from aoi.evaluation import evaluate_run
from aoi.exporting import export_run
from aoi.training import train_run


def run_pipeline(
    config_path: Path,
    *,
    consistency_input: Path,
    data_root: Path | None = None,
    category: str | None = None,
    device: str = "cpu",
    run_root: Path | None = None,
    run_id: str | None = None,
    opset: int = 18,
    max_consistency_images: int = 8,
) -> RunPaths:
    """Run training, evaluation, export, and consistency verification in order."""
    run = train_run(
        config_path,
        data_root=data_root,
        category=category,
        device=device,
        run_root=run_root,
        run_id=run_id,
    )
    evaluate_run(run.root, device=device)
    export_run(run.root, opset=opset)
    check_run_consistency(
        run.root,
        consistency_input,
        device=device,
        max_images=max_consistency_images,
    )
    return run

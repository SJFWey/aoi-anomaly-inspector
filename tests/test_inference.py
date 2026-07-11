from pathlib import Path

import pytest

from aoi.inference import predict_directory
from aoi.training import train_run
from aoi.evaluation import evaluate_run


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_predict_directory_writes_canonical_outputs(tiny_config: Path, tmp_path: Path) -> None:
    run = train_run(tiny_config, run_root=tmp_path / "runs", run_id="prediction")
    evaluate_run(run.root)
    output_dir = tmp_path / "predictions"
    records = predict_directory(
        run_dir=run.root,
        input_dir=REPO_ROOT / "examples" / "mvtec_tiny" / "bottle" / "test",
        output_dir=output_dir,
    )
    assert len(records) == 2
    assert (output_dir / "preds.jsonl").exists()
    assert {record["decision"] for record in records} <= {"OK", "NG"}


def test_predict_directory_requires_thresholds(tiny_config: Path, tmp_path: Path) -> None:
    run = train_run(tiny_config, run_root=tmp_path / "runs", run_id="missing-thresholds")
    with pytest.raises(FileNotFoundError, match="thresholds.json"):
        predict_directory(run.root, REPO_ROOT / "examples" / "mvtec_tiny" / "bottle" / "test", tmp_path / "out")

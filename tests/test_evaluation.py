from pathlib import Path

from aoi.evaluation import binary_metrics, evaluate_run, safe_auroc
from aoi.training import train_run


def test_binary_metrics_are_threshold_dependent() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics == {
        "accuracy": 0.75,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
        "true_positive": 2,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
    }


def test_safe_auroc_returns_none_for_one_class() -> None:
    assert safe_auroc([0, 0], [0.1, 0.2]) is None


def test_evaluate_run_writes_calibrated_artifacts(tiny_config: Path, tmp_path: Path) -> None:
    run = train_run(tiny_config, run_root=tmp_path / "runs", run_id="evaluation")
    evaluated = evaluate_run(run.root)
    assert evaluated.thresholds.exists()
    assert evaluated.metrics.exists()
    assert evaluated.preds.exists()
    assert evaluated.report.exists()
    assert (evaluated.predictions / "masks").exists()

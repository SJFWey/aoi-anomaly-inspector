from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from aoi.consistency import (
    check_run_consistency,
    compare_decisions,
    compare_outputs,
    enforce_decision_agreement,
    enforce_tolerances,
)
from aoi.evaluation import evaluate_run
from aoi.exporting import export_run
from aoi.training import train_run


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compare_outputs_reports_expected_errors() -> None:
    native = np.array([0.0, 1.0], dtype=np.float32)
    exported = np.array([0.0, 1.00001], dtype=np.float32)
    result = compare_outputs(native, exported)
    assert result["max_abs_error"] == pytest.approx(0.00001, abs=1e-7)
    assert result["mean_abs_error"] == pytest.approx(0.000005, abs=1e-7)


def test_enforce_tolerances_rejects_large_error() -> None:
    with pytest.raises(RuntimeError, match="consistency tolerance"):
        enforce_tolerances(
            {"max_abs_error": 0.1, "mean_abs_error": 0.01},
            max_abs_tolerance=0.001,
            mean_abs_tolerance=0.001,
        )


def test_compare_outputs_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        compare_outputs(
            np.array([float("nan")], dtype=np.float32),
            np.array([float("nan")], dtype=np.float32),
        )


def test_enforce_tolerances_rejects_non_finite_metrics() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        enforce_tolerances(
            {"max_abs_error": float("nan"), "mean_abs_error": float("nan")},
            max_abs_tolerance=0.001,
            mean_abs_tolerance=0.001,
        )


def test_enforce_decision_agreement_requires_exact_match() -> None:
    enforce_decision_agreement(None)
    enforce_decision_agreement(1.0)
    with pytest.raises(RuntimeError, match="decisions disagree"):
        enforce_decision_agreement(0.5)


def test_compare_decisions_separates_threshold_ambiguity() -> None:
    metrics = compare_decisions(
        np.array([0.4, 0.50001]),
        np.array([0.4, 0.49999]),
        threshold=0.5,
        ambiguity_tolerance=0.0001,
    )

    assert metrics["decision_agreement"] == 0.5
    assert metrics["stable_decision_agreement"] == 1.0
    assert metrics["ambiguous_decisions"] == 1


@pytest.mark.parametrize("model_name", ["padim", "patchcore"])
def test_consistency_check_writes_passing_report(
    tiny_config: Path, tmp_path: Path, model_name: str
) -> None:
    data = yaml.safe_load(tiny_config.read_text(encoding="utf-8"))
    data["model"]["name"] = model_name
    config = tmp_path / f"{model_name}.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    run = train_run(config, run_root=tmp_path / "runs", run_id=f"consistency-{model_name}")
    evaluate_run(run.root)
    export_run(run.root)
    result = check_run_consistency(
        run.root, REPO_ROOT / "examples" / "mvtec_tiny" / "bottle" / "test"
    )
    assert result["passed"] is True
    assert run.consistency.exists()


def test_single_training_image_padim_pipeline_stays_finite(tiny_config: Path, tmp_path: Path) -> None:
    source = REPO_ROOT / "examples" / "mvtec_tiny"
    dataset = tmp_path / "one_image_mvtec"
    shutil.copytree(source, dataset)
    (dataset / "bottle" / "train" / "good" / "001.png").unlink()
    data = yaml.safe_load(tiny_config.read_text(encoding="utf-8"))
    data["dataset"]["root"] = str(dataset)
    data["model"]["name"] = "padim"
    config = tmp_path / "one-image-padim.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    run = train_run(config, run_root=tmp_path / "one-image-runs", run_id="padim")
    evaluate_run(run.root)
    export_run(run.root)
    result = check_run_consistency(run.root, dataset / "bottle" / "test")

    assert result["passed"] is True
    assert np.isfinite(result["anomaly_map"]["max_abs_error"])
    assert np.isfinite(result["pred_score"]["max_abs_error"])

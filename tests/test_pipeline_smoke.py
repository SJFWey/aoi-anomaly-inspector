from pathlib import Path

from aoi.pipeline import run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_pipeline_completes_all_stages(tmp_path: Path) -> None:
    run = run_pipeline(
        config_path=REPO_ROOT / "configs" / "patchcore_tiny.yaml",
        run_root=tmp_path / "runs",
        run_id="smoke",
        consistency_input=REPO_ROOT / "examples" / "mvtec_tiny" / "bottle" / "test",
    )
    assert run.model.exists()
    assert run.metrics.exists()
    assert run.onnx.exists()
    assert run.consistency.exists()

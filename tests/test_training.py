import json
from pathlib import Path

from aoi.training import train_run


def test_train_run_writes_only_training_artifacts(tiny_config: Path, tmp_path: Path) -> None:
    run = train_run(tiny_config, run_root=tmp_path / "runs", run_id="test-run")
    assert run.model.exists()
    assert run.config.exists()
    assert run.meta.exists()
    assert run.manifest.exists()
    assert not run.thresholds.exists()
    manifest = json.loads(run.manifest.read_text(encoding="utf-8"))
    assert manifest["training"]["status"] == "complete"
    assert manifest["training"]["model_sha256"]

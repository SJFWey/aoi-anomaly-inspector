import json
import re
from pathlib import Path

from aoi.training import _run_id, train_run


def test_default_run_id_is_compact_utc() -> None:
    assert re.fullmatch(r"\d{8}T\d{6}Z", _run_id())


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

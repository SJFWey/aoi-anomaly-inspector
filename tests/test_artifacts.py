import json
from pathlib import Path

import pytest

from aoi.artifacts import RunPaths, update_manifest, write_json_atomic


def test_run_paths_expose_one_layout(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "runs" / "patchcore" / "bottle" / "run-1")
    assert paths.model == paths.root / "model.pt"
    assert paths.thresholds == paths.root / "thresholds.json"
    assert paths.onnx == paths.root / "export" / "model.onnx"
    assert paths.consistency == paths.root / "export" / "consistency.json"


def test_manifest_update_preserves_previous_sections(tmp_path: Path) -> None:
    path = tmp_path / "artifact_manifest.json"
    update_manifest(path, {"training": {"status": "complete"}})
    update_manifest(path, {"export": {"status": "complete"}})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["training"]["status"] == "complete"
    assert data["export"]["status"] == "complete"


def test_json_writer_rejects_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        write_json_atomic(tmp_path / "invalid.json", {"metric": float("nan")})

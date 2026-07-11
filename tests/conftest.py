from pathlib import Path

import yaml
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_config(tmp_path: Path) -> Path:
    data = yaml.safe_load((REPO_ROOT / "configs" / "patchcore_tiny.yaml").read_text(encoding="utf-8"))
    data["dataset"]["root"] = str(REPO_ROOT / "examples" / "mvtec_tiny")
    data["outputs"]["run_root"] = str(tmp_path / "runs")
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path

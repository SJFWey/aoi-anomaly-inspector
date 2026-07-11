from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def model(self) -> Path:
        return self.root / "model.pt"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def manifest(self) -> Path:
        return self.root / "artifact_manifest.json"

    @property
    def thresholds(self) -> Path:
        return self.root / "thresholds.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def masks(self) -> Path:
        return self.predictions / "masks"

    @property
    def overlays(self) -> Path:
        return self.predictions / "overlays"

    @property
    def preds(self) -> Path:
        return self.root / "preds.jsonl"

    @property
    def report(self) -> Path:
        return self.root / "report.json"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def onnx(self) -> Path:
        return self.export_dir / "model.onnx"

    @property
    def export_meta(self) -> Path:
        return self.export_dir / "export_meta.json"

    @property
    def consistency(self) -> Path:
        return self.export_dir / "consistency.json"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json_atomic(path: Path, data: object) -> None:
    _atomic_text(path, json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    _atomic_text(path, yaml.safe_dump(data, sort_keys=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def update_manifest(path: Path, section: dict[str, object]) -> None:
    current: dict[str, object] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Artifact manifest must be a mapping: {path}")
        current = data
    current.update(section)
    write_json_atomic(path, current)

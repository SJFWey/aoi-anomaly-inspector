"""Configuration I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_yaml", "dump_yaml", "dump_json"]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        TypeError: If the file does not contain a YAML mapping.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"Config must be a YAML mapping, got {type(data)}"
        raise TypeError(msg)
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write data to a YAML file.

    Args:
        path: Path to the output YAML file.
        data: Dictionary to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    """Write data to a JSON file with proper formatting.

    Args:
        path: Path to the output JSON file.
        data: Dictionary to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

"""Device and config parsing utilities."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["resolve_device", "as_int_pair"]


def resolve_device(device_cfg: str) -> tuple[str, int]:
    """Parse device string and return accelerator/devices tuple.

    Args:
        device_cfg: Device configuration string ("auto", "cpu", or "cuda").

    Returns:
        Tuple of (accelerator, devices) for PyTorch Lightning Trainer.

    Raises:
        ValueError: If device_cfg is not one of the supported values.
        RuntimeError: If CUDA is requested but not available.
    """
    device_cfg = device_cfg.lower().strip()
    if device_cfg not in {"auto", "cpu", "cuda"}:
        msg = f"Unsupported device: {device_cfg} (expected auto|cpu|cuda)"
        raise ValueError(msg)
    if device_cfg == "cpu":
        return ("cpu", 1)
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return ("gpu", 1)
    return ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)


def as_int_pair(value: Any, *, key: str) -> tuple[int, int]:
    """Validate and convert value to a 2-int tuple.

    Args:
        value: Value to convert (should be a list or tuple of 2 items).
        key: Config key name for error messages.

    Returns:
        Tuple of two integers.

    Raises:
        ValueError: If value is None.
        TypeError: If value is not a 2-item list/tuple.
    """
    if value is None:
        raise ValueError(f"Missing required config key: {key}")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{key} must be a 2-item list/tuple, got {value!r}")
    return (int(value[0]), int(value[1]))

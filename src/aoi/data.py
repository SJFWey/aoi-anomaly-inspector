"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from torch.utils.data import DataLoader, Subset

__all__ = [
    "build_dataloaders",
    "build_test_loader",
    "build_train_loader",
    "build_validation_loader",
]


def _make_loader_safe(
    dataset: MVTecADDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    """Create DataLoader with fallback for PermissionError on Windows."""
    loader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    if num_workers > 0:
        try:
            next(iter(loader))
        except PermissionError:
            import warnings

            warnings.warn(
                "PermissionError when spawning workers; falling back to num_workers=0",
                RuntimeWarning,
                stacklevel=3,
            )
            loader = DataLoader(
                dataset=dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=0,
                collate_fn=dataset.collate_fn,
            )
    return loader


def build_train_loader(cfg: dict[str, Any]) -> DataLoader:
    """Create train DataLoader (good samples only) for threshold computation.

    Args:
        cfg: Full configuration dictionary containing 'data' section.

    Returns:
        Train DataLoader with shuffle=False for reproducibility.

    Raises:
        FileNotFoundError: If the category directory does not exist.
    """
    data_cfg = cfg.get("data", {})
    data_root = Path(str(data_cfg.get("root", "datasets/mvtech")))
    category = str(data_cfg.get("category", "transistor"))
    category_dir = data_root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    train_dataset = MVTecADDataset(root=str(data_root), category=category, split="train")
    eval_batch_size = int(data_cfg.get("eval_batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    return _make_loader_safe(train_dataset, eval_batch_size, num_workers, shuffle=False)


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train and test DataLoaders from configuration.

    Args:
        cfg: Full configuration dictionary containing 'data' section.

    Returns:
        Tuple of (train_loader, train_pred_loader, test_loader).

    Raises:
        FileNotFoundError: If the category directory does not exist.
    """
    data_cfg = cfg.get("data", {})
    data_root = Path(str(data_cfg.get("root", "datasets/mvtech")))
    category = str(data_cfg.get("category", "transistor"))

    category_dir = data_root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    train_dataset = MVTecADDataset(root=str(data_root), category=category, split="train")
    test_dataset = MVTecADDataset(root=str(data_root), category=category, split="test")

    train_batch_size = int(data_cfg.get("train_batch_size", 32))
    eval_batch_size = int(data_cfg.get("eval_batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    def _make_loaders(workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=train_batch_size,
            num_workers=workers,
            collate_fn=train_dataset.collate_fn,
        )
        train_pred_loader = DataLoader(
            dataset=train_dataset,
            shuffle=False,
            batch_size=eval_batch_size,
            num_workers=workers,
            collate_fn=train_dataset.collate_fn,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=eval_batch_size,
            num_workers=workers,
            collate_fn=test_dataset.collate_fn,
        )
        return train_loader, train_pred_loader, test_loader

    train_loader, train_pred_loader, test_loader = _make_loaders(num_workers)
    if num_workers > 0:
        try:
            next(iter(train_loader))
        except PermissionError:
            import warnings

            warnings.warn(
                "PermissionError when spawning dataloader workers; falling back to num_workers=0",
                RuntimeWarning,
                stacklevel=2,
            )
            train_loader, train_pred_loader, test_loader = _make_loaders(0)

    return train_loader, train_pred_loader, test_loader


def build_test_loader(cfg: dict[str, Any]) -> DataLoader:
    """Create test DataLoader from configuration.

    Args:
        cfg: Full configuration dictionary containing 'data' section.

    Returns:
        Test DataLoader.

    Raises:
        FileNotFoundError: If the category directory does not exist.
    """
    data_cfg = cfg.get("data", {})
    data_root = Path(str(data_cfg.get("root", "datasets/mvtech")))
    category = str(data_cfg.get("category", "transistor"))
    category_dir = data_root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    test_dataset = MVTecADDataset(root=str(data_root), category=category, split="test")
    eval_batch_size = int(data_cfg.get("eval_batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    return _make_loader_safe(test_dataset, eval_batch_size, num_workers, shuffle=False)


def build_validation_loader(
    cfg: dict[str, Any],
    split_ratio: float = 1.0,
    seed: int = 42,
) -> DataLoader:
    """Create validation DataLoader from test split for threshold tuning.

    Uses the test set (which contains labeled anomalies) for validation-based
    threshold optimization. Optionally splits test set for held-out evaluation.

    Args:
        cfg: Full configuration dictionary containing 'data' section.
        split_ratio: Fraction of test set to use for validation (0.0 < ratio <= 1.0).
            Use 1.0 for full test set, <1.0 for held-out evaluation.
        seed: Random seed for reproducible split.

    Returns:
        Validation DataLoader with labels.

    Raises:
        FileNotFoundError: If the category directory does not exist.
        ValueError: If split_ratio is not in valid range.
    """
    if not 0.0 < split_ratio <= 1.0:
        raise ValueError(f"split_ratio must be in (0, 1], got {split_ratio}")

    data_cfg = cfg.get("data", {})
    data_root = Path(str(data_cfg.get("root", "datasets/mvtech")))
    category = str(data_cfg.get("category", "transistor"))
    category_dir = data_root / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    test_dataset = MVTecADDataset(root=str(data_root), category=category, split="test")
    eval_batch_size = int(data_cfg.get("eval_batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    # Apply subset split if needed
    if split_ratio < 1.0:
        n = len(test_dataset)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        val_size = int(n * split_ratio)
        val_indices = indices[:val_size].tolist()
        test_dataset = Subset(test_dataset, val_indices)

    return _make_loader_safe(test_dataset, eval_batch_size, num_workers, shuffle=False)

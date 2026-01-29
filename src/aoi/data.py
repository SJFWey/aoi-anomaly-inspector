"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from anomalib.data.datasets.image.mvtecad import MVTecADDataset

__all__ = ["build_dataloaders", "build_test_loader", "build_train_loader"]


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

    train_dataset = MVTecADDataset(
        root=str(data_root), category=category, split="train"
    )
    eval_batch_size = int(data_cfg.get("eval_batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    loader = DataLoader(
        dataset=train_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    if num_workers > 0:
        try:
            next(iter(loader))
        except PermissionError:
            import warnings

            warnings.warn(
                "PermissionError when spawning dataloader workers; falling back to num_workers=0",
                RuntimeWarning,
                stacklevel=2,
            )
            loader = DataLoader(
                dataset=train_dataset,
                shuffle=False,
                batch_size=eval_batch_size,
                num_workers=0,
                collate_fn=train_dataset.collate_fn,
            )
    return loader


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

    train_dataset = MVTecADDataset(
        root=str(data_root), category=category, split="train"
    )
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

    loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn,
    )
    if num_workers > 0:
        try:
            next(iter(loader))
        except PermissionError:
            import warnings

            warnings.warn(
                "PermissionError when spawning dataloader workers; falling back to num_workers=0",
                RuntimeWarning,
                stacklevel=2,
            )
            loader = DataLoader(
                dataset=test_dataset,
                shuffle=False,
                batch_size=eval_batch_size,
                num_workers=0,
                collate_fn=test_dataset.collate_fn,
            )
    return loader

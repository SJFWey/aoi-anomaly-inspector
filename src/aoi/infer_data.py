"""Data loading utilities for inference on image folders.

This module provides a simple Dataset and DataLoader for running inference on
arbitrary image folders (not MVTec format). It supports recursive directory
scanning and produces batches compatible with anomalib's model predict_step.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from anomalib.data import ImageBatch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ImageFolderDataset",
    "InferenceBatch",
    "build_infer_dataloader",
]

# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Alias for compatibility
InferenceBatch = ImageBatch


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder for inference.

    Scans a directory for images (optionally recursive), loads and preprocesses
    them to a fixed size for model inference.

    Args:
        root: Path to the image folder.
        image_size: Target image size as (H, W).
        recursive: Whether to scan subdirectories recursively.
        extensions: Set of allowed file extensions (with leading dot, lowercase).
    """

    def __init__(
        self,
        root: Path | str,
        image_size: tuple[int, int] = (256, 256),
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.extensions = extensions or IMAGE_EXTENSIONS

        if not self.root.exists():
            raise FileNotFoundError(f"Input directory not found: {self.root}")

        # Scan for images
        if recursive:
            self.image_paths = sorted(
                p
                for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in self.extensions
            )
        else:
            self.image_paths = sorted(
                p
                for p in self.root.iterdir()
                if p.is_file() and p.suffix.lower() in self.extensions
            )

        if not self.image_paths:
            raise ValueError(
                f"No images found in {self.root} "
                f"(extensions: {', '.join(sorted(self.extensions))})"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and preprocess a single image.

        Returns:
            Dictionary with:
            - image: Tensor of shape (C, H, W), float32, normalized to [0, 1]
            - image_path: Original file path as string
            - original_size: Tuple (H, W) of original image dimensions
        """
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "OpenCV (cv2) is required for image loading. "
                "Install with: pip install opencv-python"
            ) from e

        image_path = self.image_paths[idx]

        # Load image (BGR)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Failed to load image: {image_path}")

        original_h, original_w = img_bgr.shape[:2]

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to target size
        target_h, target_w = self.image_size
        img_resized = cv2.resize(
            img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # Convert to float32 and normalize to [0, 1]
        img_float = img_resized.astype(np.float32) / 255.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)

        return {
            "image": img_tensor,
            "image_path": str(image_path),
            "original_size": (original_h, original_w),
        }


def _collate_inference_batch(
    batch: "Sequence[dict[str, Any]]",
) -> ImageBatch:
    """Collate function for ImageFolderDataset.

    Args:
        batch: List of sample dictionaries from ImageFolderDataset.

    Returns:
        ImageBatch compatible with anomalib models.
    """
    images = torch.stack([sample["image"] for sample in batch])
    image_paths = [sample["image_path"] for sample in batch]

    return ImageBatch(
        image=images,
        image_path=image_paths,
    )


def build_infer_dataloader(
    input_dir: Path | str,
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 32,
    num_workers: int = 0,
    recursive: bool = True,
) -> DataLoader:
    """Build a DataLoader for inference on an image folder.

    Args:
        input_dir: Path to the input image directory.
        image_size: Target image size as (H, W).
        batch_size: Batch size for inference.
        num_workers: Number of dataloader workers.
        recursive: Whether to scan subdirectories recursively.

    Returns:
        DataLoader yielding InferenceBatch objects.
    """
    dataset = ImageFolderDataset(
        root=input_dir,
        image_size=image_size,
        recursive=recursive,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_inference_batch,
    )

    # Fallback for num_workers issues (e.g., Windows PermissionError)
    if num_workers > 0:
        try:
            next(iter(loader))
        except PermissionError:
            print(
                "PermissionError with dataloader workers; falling back to num_workers=0"
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_inference_batch,
            )

    return loader

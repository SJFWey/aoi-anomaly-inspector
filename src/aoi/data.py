from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    label: str
    label_index: int
    mask_path: Path | None


@dataclass(frozen=True)
class NormalSplit:
    fit: list[ImageSample]
    calibration: list[ImageSample]
    shares_samples: bool


def split_normal_samples(
    samples: list[ImageSample], calibration_fraction: float, seed: int
) -> NormalSplit:
    """Deterministically reserve normal images for threshold calibration."""
    ordered = sorted(samples, key=lambda sample: str(sample.image_path))
    if not ordered:
        raise ValueError("At least one normal sample is required")
    if calibration_fraction <= 0.0 or len(ordered) < 2:
        return NormalSplit(fit=ordered, calibration=ordered, shares_samples=True)

    calibration_count = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * calibration_fraction))),
    )
    generator = np.random.default_rng(seed)
    calibration_indices = set(
        int(index) for index in generator.permutation(len(ordered))[:calibration_count]
    )
    fit = [sample for index, sample in enumerate(ordered) if index not in calibration_indices]
    calibration = [sample for index, sample in enumerate(ordered) if index in calibration_indices]
    return NormalSplit(fit=fit, calibration=calibration, shares_samples=False)


def collect_image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def mask_path_for(image_path: Path, category_dir: Path, defect_type: str) -> Path:
    return category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"


def load_mvtec_split(data_root: Path, category: str, split: str) -> list[ImageSample]:
    category_dir = data_root / category
    split_dir = category_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"MVTec split directory not found: {split_dir}")

    samples: list[ImageSample] = []
    for label_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = label_dir.name
        for image_path in collect_image_paths(label_dir):
            label_index = 0 if label == "good" else 1
            mask_path = None
            if split == "test" and label_index == 1:
                mask_path = mask_path_for(image_path, category_dir, label)
            samples.append(
                ImageSample(
                    image_path=image_path,
                    label=label,
                    label_index=label_index,
                    mask_path=mask_path,
                )
            )
    if not samples:
        raise RuntimeError(f"No images found in {split_dir}")
    return samples


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def read_mask(path: Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape_hw, dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")
    if mask.shape[:2] != shape_hw:
        raise ValueError(
            f"Ground-truth mask size mismatch: image={shape_hw}, mask={mask.shape[:2]}, path={path}"
        )
    if cv2.countNonZero(mask) == 0:
        raise ValueError(f"Ground-truth mask is empty: {path}")
    return mask


def bgr_to_rgb_unit_tensor(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """Return resized RGB CHW float32 in [0, 1], without normalization."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def mask_to_array(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)

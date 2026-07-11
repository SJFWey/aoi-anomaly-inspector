from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Defect:
    bbox: list[int]
    area: int
    centroid: list[float]


@dataclass(frozen=True)
class PostprocessResult:
    mask: np.ndarray
    defects: list[Defect]


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    values = anomaly_map.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def postprocess_map(
    anomaly_map: np.ndarray,
    pixel_threshold: float,
    output_shape: tuple[int, int],
    min_area: int = 50,
    morph_kernel: int = 3,
) -> PostprocessResult:
    resized = cv2.resize(
        anomaly_map.astype(np.float32),
        (output_shape[1], output_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = (resized > pixel_threshold).astype(np.uint8) * 255
    if morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    defects: list[Defect] = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        clean[labels == idx] = 255
        defects.append(
            Defect(
                bbox=[x, y, x + w, y + h],
                area=area,
                centroid=[float(centroids[idx][0]), float(centroids[idx][1])],
            )
        )
    return PostprocessResult(mask=clean, defects=defects)


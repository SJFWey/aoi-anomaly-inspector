from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    image_threshold: float
    pixel_threshold: float
    quantile_image: float
    quantile_pixel: float


def fit_thresholds(
    image_scores: list[float],
    anomaly_maps: list[np.ndarray],
    quantile_image: float = 0.995,
    quantile_pixel: float = 0.999,
) -> Thresholds:
    if not image_scores:
        raise ValueError("At least one image score is required.")
    if not anomaly_maps:
        raise ValueError("At least one anomaly map is required.")

    pixels = np.concatenate([np.asarray(m, dtype=np.float32).reshape(-1) for m in anomaly_maps])
    return Thresholds(
        image_threshold=float(np.quantile(np.asarray(image_scores, dtype=np.float32), quantile_image)),
        pixel_threshold=float(np.quantile(pixels, quantile_pixel)),
        quantile_image=quantile_image,
        quantile_pixel=quantile_pixel,
    )


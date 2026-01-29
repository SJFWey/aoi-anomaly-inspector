"""Post-processing utilities for anomaly detection outputs.

This module converts raw anomaly maps into actionable defect information:
- Binary mask generation from pixel threshold
- Connected component analysis for defect geometry extraction
- Small area filtering to reduce noise

All coordinates use the "model input image size" coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

__all__ = [
    "DefectInfo",
    "anomaly_map_to_mask",
    "extract_components",
    "postprocess_anomaly_map",
]


@dataclass
class DefectInfo:
    """Information about a single detected defect region.

    All coordinates are in the model input image coordinate system.
    """

    component_id: int
    area: int  # Number of pixels in the defect region
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    centroid: tuple[float, float]  # (x, y) center of mass
    max_anomaly_value: float  # Maximum anomaly score within this region
    mean_anomaly_value: float  # Mean anomaly score within this region

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "component_id": self.component_id,
            "area": self.area,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "max_anomaly_value": self.max_anomaly_value,
            "mean_anomaly_value": self.mean_anomaly_value,
        }


@dataclass
class PostprocessResult:
    """Result of post-processing an anomaly map."""

    mask: np.ndarray  # Binary mask (H, W), dtype=uint8, values 0 or 255
    defects: list[DefectInfo] = field(default_factory=list)
    num_defects: int = 0
    total_defect_area: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary (excluding mask)."""
        return {
            "num_defects": self.num_defects,
            "total_defect_area": self.total_defect_area,
            "defects": [d.to_dict() for d in self.defects],
        }


def anomaly_map_to_mask(
    anomaly_map: np.ndarray,
    pixel_threshold: float,
) -> np.ndarray:
    """Convert anomaly map to binary mask using pixel threshold.

    Uses raw anomaly values (not normalized) for threshold comparison,
    consistent with how thresholds were computed during training.

    Args:
        anomaly_map: 2D array of anomaly scores, shape (H, W).
        pixel_threshold: Threshold value; pixels >= threshold are marked as defect.

    Returns:
        Binary mask as uint8 array (H, W), values 0 (normal) or 255 (defect).
    """
    # Ensure 2D
    if anomaly_map.ndim == 3:
        if anomaly_map.shape[0] == 1:
            anomaly_map = anomaly_map[0]
        elif anomaly_map.shape[-1] == 1:
            anomaly_map = anomaly_map[..., 0]
        else:
            # Take max across channel dimension
            anomaly_map = anomaly_map.max(axis=0)

    mask = (anomaly_map >= pixel_threshold).astype(np.uint8) * 255
    return mask


def extract_components(
    mask: np.ndarray,
    anomaly_map: np.ndarray | None = None,
    min_area: int = 0,
) -> list[DefectInfo]:
    """Extract connected components (defect regions) from binary mask.

    Uses 8-connectivity for connected component analysis.

    Args:
        mask: Binary mask (H, W), non-zero values indicate defects.
        anomaly_map: Optional anomaly map for extracting anomaly statistics per region.
        min_area: Minimum area (in pixels) for a component to be included.
            Components smaller than this are filtered out.

    Returns:
        List of DefectInfo objects, one per valid defect region.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for connected component analysis. "
            "Install with: pip install opencv-python"
        ) from e

    # Ensure mask is binary uint8
    binary_mask = (mask > 0).astype(np.uint8)

    # Connected component analysis with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    defects = []
    # Label 0 is background, start from 1
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        # Filter by minimum area
        if area < min_area:
            continue

        x_min = int(stats[label_id, cv2.CC_STAT_LEFT])
        y_min = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x_max = x_min + width
        y_max = y_min + height

        centroid_x = float(centroids[label_id, 0])
        centroid_y = float(centroids[label_id, 1])

        # Compute anomaly statistics if anomaly_map provided
        max_val = 0.0
        mean_val = 0.0
        if anomaly_map is not None:
            # Ensure anomaly_map is 2D
            amap = anomaly_map
            if amap.ndim == 3:
                if amap.shape[0] == 1:
                    amap = amap[0]
                elif amap.shape[-1] == 1:
                    amap = amap[..., 0]
                else:
                    amap = amap.max(axis=0)

            component_mask = labels == label_id
            component_values = amap[component_mask]
            if len(component_values) > 0:
                max_val = float(np.max(component_values))
                mean_val = float(np.mean(component_values))

        defects.append(
            DefectInfo(
                component_id=len(defects) + 1,
                area=area,
                bbox=(x_min, y_min, x_max, y_max),
                centroid=(centroid_x, centroid_y),
                max_anomaly_value=max_val,
                mean_anomaly_value=mean_val,
            )
        )

    return defects


def postprocess_anomaly_map(
    anomaly_map: np.ndarray,
    pixel_threshold: float,
    min_defect_area: int = 0,
) -> PostprocessResult:
    """Full post-processing pipeline for an anomaly map.

    Converts anomaly map to mask, extracts connected components, and computes
    defect statistics.

    Args:
        anomaly_map: 2D or 3D array of anomaly scores.
        pixel_threshold: Threshold for binary mask generation.
        min_defect_area: Minimum area for defect filtering.

    Returns:
        PostprocessResult containing mask and defect information.
    """
    mask = anomaly_map_to_mask(anomaly_map, pixel_threshold)
    defects = extract_components(mask, anomaly_map, min_area=min_defect_area)

    total_area = sum(d.area for d in defects)

    return PostprocessResult(
        mask=mask,
        defects=defects,
        num_defects=len(defects),
        total_defect_area=total_area,
    )

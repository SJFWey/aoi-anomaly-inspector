"""Visualization utilities for anomaly detection outputs.

This module provides functions to save:
- Binary defect masks as grayscale images
- Overlay visualizations (anomaly heatmap on original image)

Note: Normalization is used only for visualization; it does not affect threshold
comparisons which use raw anomaly values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "save_mask",
    "save_overlay",
    "save_defect_overlay",
    "normalize_anomaly_map",
    "create_heatmap",
]


def normalize_anomaly_map(
    anomaly_map: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Normalize anomaly map to [0, 1] range for visualization.

    This normalization is for display purposes only and should NOT be used
    for threshold comparisons.

    Args:
        anomaly_map: 2D array of anomaly scores.
        vmin: Minimum value for normalization. If None, uses anomaly_map.min().
        vmax: Maximum value for normalization. If None, uses anomaly_map.max().

    Returns:
        Normalized array in [0, 1] range.
    """
    # Ensure 2D
    if anomaly_map.ndim == 3:
        if anomaly_map.shape[0] == 1:
            anomaly_map = anomaly_map[0]
        elif anomaly_map.shape[-1] == 1:
            anomaly_map = anomaly_map[..., 0]
        else:
            anomaly_map = anomaly_map.max(axis=0)

    if vmin is None:
        vmin = float(anomaly_map.min())
    if vmax is None:
        vmax = float(anomaly_map.max())

    # Avoid division by zero
    if vmax - vmin < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)

    normalized = (anomaly_map - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def create_heatmap(
    normalized_map: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """Create a colored heatmap from normalized anomaly map.

    Args:
        normalized_map: 2D array normalized to [0, 1].
        colormap: Matplotlib colormap name (default: 'jet').

    Returns:
        RGB heatmap as uint8 array (H, W, 3).
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for heatmap generation. "
            "Install with: pip install opencv-python"
        ) from e

    # Convert to uint8 for colormap application
    gray = (normalized_map * 255).astype(np.uint8)

    # OpenCV colormap mapping
    colormap_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    cv_colormap = colormap_dict.get(colormap.lower(), cv2.COLORMAP_JET)

    # Apply colormap (returns BGR)
    heatmap_bgr = cv2.applyColorMap(gray, cv_colormap)
    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


def save_mask(
    mask: np.ndarray,
    path: Path | str,
) -> None:
    """Save binary mask as grayscale PNG image.

    Args:
        mask: Binary mask (H, W), values 0 or 255.
        path: Output file path.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for saving images. "
            "Install with: pip install opencv-python"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    cv2.imwrite(str(path), mask)


def save_overlay(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: np.ndarray,
    path: Path | str,
    alpha: float = 0.4,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: str = "turbo",
    contour_color: tuple[int, int, int] = (0, 255, 0),
    contour_thickness: int = 2,
    mask_only_heatmap: bool = False,
) -> None:
    """Save overlay with anomaly heatmap and defect contours.

    Args:
        image: Original image (H, W, 3) RGB or (H, W) grayscale.
        anomaly_map: 2D array of anomaly scores.
        mask: Binary mask (H, W) for defect contours.
        path: Output file path.
        alpha: Heatmap transparency.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        colormap: Colormap for heatmap (default: 'turbo').
        contour_color: RGB color for contours.
        contour_thickness: Thickness of contour lines.
        mask_only_heatmap: If True, only show heatmap in masked regions.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for saving images. "
            "Install with: pip install opencv-python"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure image is RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    h, w = image.shape[:2]

    # Handle anomaly map dimensions
    amap = anomaly_map
    if amap.ndim == 3:
        if amap.shape[0] == 1:
            amap = amap[0]
        elif amap.shape[-1] == 1:
            amap = amap[..., 0]
        else:
            amap = amap.max(axis=0)

    if amap.shape[:2] != (h, w):
        amap = cv2.resize(
            amap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )

    # Create heatmap overlay
    normalized = normalize_anomaly_map(amap, vmin=vmin, vmax=vmax)
    heatmap = create_heatmap(normalized, colormap=colormap)

    # Apply mask-only heatmap if requested
    if mask_only_heatmap:
        mask_resized_for_blend = mask
        if mask.shape[:2] != (h, w):
            mask_resized_for_blend = cv2.resize(
                mask, (w, h), interpolation=cv2.INTER_NEAREST
            )
        mask_bool = (mask_resized_for_blend > 0)[..., np.newaxis]
        overlay = np.where(
            mask_bool, cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0), image
        )
    else:
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    # Resize mask if needed and draw contours
    mask_resized = mask
    if mask.shape[:2] != (h, w):
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_uint8 = (mask_resized > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours (convert RGB to BGR for cv2.drawContours)
    contour_color_bgr = (contour_color[2], contour_color[1], contour_color[0])
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, contour_color_bgr, contour_thickness)

    cv2.imwrite(str(path), overlay_bgr)


def save_defect_overlay(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: np.ndarray | None,
    path: Path | str,
    pixel_threshold: float,
    alpha_defect: float = 0.6,
    alpha_normal: float = 0.0,
    colormap: str = "turbo",
    contour_color: tuple[int, int, int] = (0, 255, 0),
    contour_thickness: int = 2,
) -> None:
    """Save defect-focused overlay with clean visualization.

    This function produces cleaner visualizations by:
    - Only showing heatmap in defect regions (or above threshold)
    - Using a perceptually uniform colormap (turbo)
    - Clean contours around detected defects

    Args:
        image: Original image (H, W, 3) RGB or (H, W) grayscale.
        anomaly_map: 2D array of anomaly scores.
        mask: Binary mask (H, W) for defect regions. If None, uses threshold.
        path: Output file path.
        pixel_threshold: Threshold value for visualization cutoff.
        alpha_defect: Heatmap opacity in defect regions (0-1).
        alpha_normal: Heatmap opacity in normal regions (0-1, typically 0).
        colormap: Colormap name ('turbo', 'jet', 'inferno', 'hot', 'viridis').
        contour_color: RGB color for contours.
        contour_thickness: Thickness of contour lines.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for saving images. "
            "Install with: pip install opencv-python"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare image
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    h, w = image.shape[:2]

    # Prepare anomaly map
    amap = anomaly_map
    if amap.ndim == 3:
        if amap.shape[0] == 1:
            amap = amap[0]
        elif amap.shape[-1] == 1:
            amap = amap[..., 0]
        else:
            amap = amap.max(axis=0)
    if amap.shape[:2] != (h, w):
        amap = cv2.resize(
            amap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )

    # Create normalized map for visualization (threshold to max)
    vmin = pixel_threshold * 0.5  # Show some gradient below threshold
    vmax = max(amap.max(), pixel_threshold * 2)
    normalized = normalize_anomaly_map(amap, vmin=vmin, vmax=vmax)
    heatmap = create_heatmap(normalized, colormap=colormap)

    # Create alpha mask: high alpha in defect regions, low in normal
    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        defect_mask = (mask > 0).astype(np.float32)
    else:
        defect_mask = (amap >= pixel_threshold).astype(np.float32)

    # Smooth the mask edges for better blending
    defect_mask_smooth = cv2.GaussianBlur(defect_mask, (5, 5), 0)

    # Create per-pixel alpha: alpha_normal in normal, alpha_defect in defect
    alpha_map = alpha_normal + (alpha_defect - alpha_normal) * defect_mask_smooth
    alpha_map = alpha_map[..., np.newaxis]  # (H, W, 1)

    # Blend with variable alpha
    overlay = (image * (1 - alpha_map) + heatmap * alpha_map).astype(np.uint8)

    # Draw contours
    if mask is not None:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_uint8 = (amap >= pixel_threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_color_bgr = (contour_color[2], contour_color[1], contour_color[0])
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, contour_color_bgr, contour_thickness)

    cv2.imwrite(str(path), overlay_bgr)

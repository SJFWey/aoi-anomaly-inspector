from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aoi.postprocess import normalize_map


PANEL_SIZE = 320


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def overlay_anomaly_map(image_bgr: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = normalize_map(
        cv2.resize(
            anomaly_map.astype(np.float32),
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    )
    heatmap = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0)


def draw_defects(image_bgr: np.ndarray, defects: list[dict]) -> np.ndarray:
    output = image_bgr.copy()
    for defect in defects:
        x1, y1, x2, y2 = defect["bbox"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return output


def save_overlay(path: Path, image_bgr: np.ndarray, anomaly_map: np.ndarray, defects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_defects(overlay_anomaly_map(image_bgr, anomaly_map), defects)
    _write_image(path, overlay)


def _resize_panel(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)


def _title_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = _resize_panel(image)
    header = np.full((34, PANEL_SIZE, 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        title[:36],
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, panel], axis=0)


def _contour_overlay(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    output = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, thickness)
    return output


def _threshold_heatmap(
    anomaly_map: np.ndarray,
    pixel_threshold: float,
    output_shape: tuple[int, int],
) -> np.ndarray:
    resized = cv2.resize(
        anomaly_map.astype(np.float32),
        (output_shape[1], output_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    if pixel_threshold <= 0:
        scaled = normalize_map(resized)
    else:
        # 0.5 represents the threshold. White contours mark predicted pixels.
        scaled = np.clip(resized / (pixel_threshold * 2.0), 0.0, 1.0)
    heatmap = cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_TURBO)
    threshold_mask = (resized > pixel_threshold).astype(np.uint8) * 255
    return _contour_overlay(heatmap, threshold_mask, (255, 255, 255), thickness=2)


def _mask_panel(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[mask > 0] = color
    return panel


def _error_panel(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    gt = gt_mask > 0
    pred = pred_mask > 0
    panel = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    panel[gt & pred] = (0, 180, 0)
    panel[~gt & pred] = (0, 0, 255)
    panel[gt & ~pred] = (255, 0, 0)
    return panel


def save_diagnostic_composite(
    path: Path,
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    title: str,
    pixel_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gt_mask.shape[:2] != image_bgr.shape[:2]:
        gt_mask = cv2.resize(gt_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape[:2] != image_bgr.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    original = _contour_overlay(image_bgr, gt_mask, (0, 220, 0), thickness=3)
    heatmap = _threshold_heatmap(anomaly_map, pixel_threshold, image_bgr.shape[:2])
    overlay = overlay_anomaly_map(image_bgr, anomaly_map, alpha=0.5)
    overlay = _contour_overlay(overlay, pred_mask, (0, 255, 255), thickness=3)
    overlay = _contour_overlay(overlay, gt_mask, (0, 220, 0), thickness=2)

    panels = [
        _title_panel(original, "Original + GT contour"),
        _title_panel(heatmap, "Heatmap / pixel threshold"),
        _title_panel(overlay, "Overlay: GT green, pred yellow"),
        _title_panel(_mask_panel(gt_mask, (0, 220, 0)), "Ground truth mask"),
        _title_panel(_mask_panel(pred_mask, (0, 255, 255)), "Predicted mask"),
        _title_panel(_error_panel(gt_mask, pred_mask), "Error: TP green FP red FN blue"),
    ]
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(panels[3:], axis=1)
    body = np.concatenate([top, bottom], axis=0)
    header = np.full((46, body.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        title[:115],
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    _write_image(path, np.concatenate([header, body], axis=0))


def save_composite(
    path: Path,
    image_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    mask: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_anomaly_map(image_bgr, anomaly_map)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    composite = np.concatenate([image_bgr, overlay, mask_bgr], axis=1)
    header = np.full((44, composite.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        title[:160],
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    _write_image(path, np.concatenate([header, composite], axis=0))

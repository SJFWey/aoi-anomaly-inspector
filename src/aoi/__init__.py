"""AOI Anomaly Inspector - utilities for anomaly detection training and evaluation."""

from .callbacks import (
    JsonlPredictionWriter,
    MetricCollector,
    PostprocessPredictionWriter,
    evaluate_and_write_metrics,
)
from .config import dump_json, dump_yaml, load_yaml
from .data import build_dataloaders, build_test_loader, build_train_loader
from .device import as_int_pair, resolve_device
from .infer_data import ImageFolderDataset, InferenceBatch, build_infer_dataloader
from .models import build_model
from .postprocess import (
    DefectInfo,
    PostprocessResult,
    anomaly_map_to_mask,
    extract_components,
    postprocess_anomaly_map,
)
from .report import generate_report, load_predictions, write_report
from .run import RunPaths, make_run_paths, now_run_id
from .thresholds import (
    Thresholds,
    ThresholdCollector,
    compute_and_save_thresholds,
    load_thresholds,
    save_thresholds,
)
from .viz import (
    create_heatmap,
    normalize_anomaly_map,
    save_mask,
    save_overlay,
    save_overlay_with_mask,
)

__all__ = [
    # config
    "load_yaml",
    "dump_yaml",
    "dump_json",
    # device
    "resolve_device",
    "as_int_pair",
    # models
    "build_model",
    # data
    "build_dataloaders",
    "build_test_loader",
    "build_train_loader",
    # infer_data
    "ImageFolderDataset",
    "InferenceBatch",
    "build_infer_dataloader",
    # run
    "RunPaths",
    "make_run_paths",
    "now_run_id",
    # callbacks
    "JsonlPredictionWriter",
    "MetricCollector",
    "PostprocessPredictionWriter",
    "evaluate_and_write_metrics",
    # thresholds
    "Thresholds",
    "ThresholdCollector",
    "compute_and_save_thresholds",
    "load_thresholds",
    "save_thresholds",
    # postprocess
    "DefectInfo",
    "PostprocessResult",
    "anomaly_map_to_mask",
    "extract_components",
    "postprocess_anomaly_map",
    # report
    "generate_report",
    "load_predictions",
    "write_report",
    # viz
    "normalize_anomaly_map",
    "create_heatmap",
    "save_mask",
    "save_overlay",
    "save_overlay_with_mask",
]

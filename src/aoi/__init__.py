"""AOI Anomaly Inspector - utilities for anomaly detection training and evaluation."""

from .callbacks import (
    JsonlPredictionWriter,
    MetricCollector,
    evaluate_and_write_metrics,
)
from .config import dump_json, dump_yaml, load_yaml
from .data import build_dataloaders, build_test_loader, build_train_loader
from .device import as_int_pair, resolve_device
from .models import build_model
from .run import RunPaths, make_run_paths, now_run_id
from .thresholds import (
    Thresholds,
    ThresholdCollector,
    compute_and_save_thresholds,
    load_thresholds,
    save_thresholds,
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
    # run
    "RunPaths",
    "make_run_paths",
    "now_run_id",
    # callbacks
    "JsonlPredictionWriter",
    "MetricCollector",
    "evaluate_and_write_metrics",
    # thresholds
    "Thresholds",
    "ThresholdCollector",
    "compute_and_save_thresholds",
    "load_thresholds",
    "save_thresholds",
]

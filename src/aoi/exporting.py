from __future__ import annotations

import platform
from pathlib import Path

import onnx
import torch

from aoi.artifacts import RunPaths, sha256_file, update_manifest, write_json_atomic
from aoi.config import load_experiment_config
from aoi.modeling import load_model
from aoi.reporting import now_utc


def export_run(run_dir: Path, *, opset: int = 18) -> RunPaths:
    paths = RunPaths(run_dir)
    for required in (paths.config, paths.model):
        if not required.exists():
            raise FileNotFoundError(required)
    config = load_experiment_config(paths.config)
    model = load_model(paths.model, device="cpu").tensor_model
    example = torch.zeros(1, 3, config.image_size, config.image_size, dtype=torch.float32)
    paths.export_dir.mkdir(parents=True, exist_ok=True)
    temporary = paths.onnx.with_suffix(".tmp.onnx")
    torch.onnx.export(
        model,
        (example,),
        temporary,
        input_names=["image"],
        output_names=["anomaly_map", "pred_score"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=opset,
        dynamo=True,
    )
    exported = onnx.load(temporary)
    onnx.checker.check_model(exported)
    temporary.replace(paths.onnx)
    onnx_hash = sha256_file(paths.onnx)
    metadata = {
        "created_at": now_utc(),
        "model": config.model_name,
        "opset": opset,
        "input": {"name": "image", "shape": ["B", 3, config.image_size, config.image_size], "dtype": "float32"},
        "outputs": [
            {"name": "anomaly_map", "shape": ["B", 1, config.image_size, config.image_size], "dtype": "float32"},
            {"name": "pred_score", "shape": ["B"], "dtype": "float32"},
        ],
        "preprocessing": "RGB CHW float32 in [0, 1]; ImageNet normalization is embedded in the graph",
        "source_model_sha256": sha256_file(paths.model),
        "onnx_sha256": onnx_hash,
        "dependencies": {"torch": torch.__version__, "onnx": onnx.__version__, "python": platform.python_version()},
    }
    write_json_atomic(paths.export_meta, metadata)
    update_manifest(
        paths.manifest,
        {
            "export": {
                "status": "complete",
                "created_at": now_utc(),
                "onnx_sha256": onnx_hash,
                "export_meta_sha256": sha256_file(paths.export_meta),
            }
        },
    )
    return paths

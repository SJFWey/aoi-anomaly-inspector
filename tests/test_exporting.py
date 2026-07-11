from pathlib import Path

import onnx
import onnxruntime as ort
import numpy as np
import pytest
import yaml

from aoi.exporting import export_run
from aoi.training import train_run


@pytest.mark.parametrize("model_name", ["padim", "patchcore"])
def test_exported_model_has_documented_onnx_contract(tiny_config: Path, tmp_path: Path, model_name: str) -> None:
    data = yaml.safe_load(tiny_config.read_text(encoding="utf-8"))
    data["model"]["name"] = model_name
    config = tmp_path / f"{model_name}.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    paths = train_run(config, run_root=tmp_path / "runs", run_id=model_name)
    export_run(paths.root)
    model = onnx.load(paths.onnx)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(paths.onnx), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["image"]
    assert [item.name for item in session.get_outputs()] == ["anomaly_map", "pred_score"]
    outputs = session.run(None, {"image": np.random.rand(2, 3, 64, 64).astype("float32")})
    assert outputs[0].shape == (2, 1, 64, 64)
    assert outputs[1].shape == (2,)

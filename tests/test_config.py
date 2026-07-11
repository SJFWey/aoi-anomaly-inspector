from pathlib import Path

import pytest

from aoi.config import load_experiment_config, write_resolved_config


def test_config_overrides_and_consistency_defaults(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        """
seed: 7
dataset:
  root: ignored
  category: bottle
  image_size: 64
model:
  name: patchcore
  backbone: resnet18
  layers: [layer2]
  pre_trained: false
  n_features: 8
  coreset_sampling_ratio: 0.5
  max_coreset_patches: 32
trainer:
  device: cpu
  batch_size: 2
outputs:
  run_root: runs/patchcore
thresholds:
  image_quantile: 0.99
  pixel_quantile: 0.995
postprocess:
  min_area: 4
  morph_kernel: 1
consistency:
  max_abs_error: 0.0001
  mean_abs_error: 0.00001
""",
        encoding="utf-8",
    )
    config = load_experiment_config(source, data_root=tmp_path / "data", category="cable")
    assert config.data_root == tmp_path / "data"
    assert config.category == "cable"
    assert config.consistency_max_abs_error == 0.0001
    resolved = tmp_path / "resolved.yaml"
    write_resolved_config(resolved, config)
    assert load_experiment_config(resolved) == config


def test_config_rejects_unknown_model(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model: {name: gan}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported model"):
        load_experiment_config(path)

# AOI Anomaly Inspector

工业外观异常检测（AOI）项目：基于 **MVTec AD** 数据集在单类目上先跑通 **PaDiM** 与 **PatchCore** 的训练/评估闭环，并把训练产物结构化落盘到 `runs/`，为后续阈值、后处理、Runner 做准备。

> 当前进度：里程碑 2（训练管线）已可用：`scripts/train.py` + `scripts/evaluate.py`

## 环境

使用 `uv` 管理依赖（见 `pyproject.toml` / `uv.lock`）：

```bash
uv sync
```

## 数据集

默认按 anomalib 的 MVTec AD 目录结构组织，并在仓库内通过 `datasets/mvtech/` 指向数据集根目录（你已经下载并验证过）。

目录应类似：

```text
datasets/mvtech/
  transistor/
    train/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*.png
```

快速检查（会生成 `preview.png`）：

```bash
python scripts/check_data.py
```

## 训练（里程碑 2）

### PaDiM

```bash
python scripts/train.py --config configs/padim_mvtec.yaml --run_id padim_transistor_001 --device cpu
```

### PatchCore

```bash
python scripts/train.py --config configs/patchcore_mvtec.yaml --run_id patchcore_transistor_001 --device cpu
```

常用覆盖参数（两模型通用）：

```bash
python scripts/train.py --config <cfg.yaml> --category transistor --data_root datasets/mvtech --device cpu --run_id <id>
```

## 评估（复跑 metrics）

对某个 run 目录重新计算并更新 `metrics.json`：

```bash
python scripts/evaluate.py --run_dir runs/padim/transistor/padim_transistor_001 --device cpu
python scripts/evaluate.py --run_dir runs/patchcore/transistor/patchcore_transistor_001 --device cpu
```

## 产物结构

每次训练会输出到：

```text
runs/<model>/<category>/<run_id>/
  config.yaml
  meta.json
  weights/model.ckpt
  preds_train.jsonl
  preds_test.jsonl
  metrics.json
  logs/...
```

字段说明（当前阶段）
- `preds_*.jsonl`: 每张图一行，包含 `image_path / gt_label / pred_score / anomaly_max / anomaly_mean ...`
- `metrics.json`: 目前提供 `image_AUROC` 与 `pixel_AUROC`（后续可扩展 AUPRO 等）

## 备注

- 本仓库的默认 `data.num_workers=0`：当前环境下 `torch` 多进程 DataLoader 可能触发 `PermissionError`；如果你的环境支持多进程，可以在 `configs/*.yaml` 里调大。
- `PatchCore` 的 `coreset_sampling_ratio` 默认设为 `0.01` 以保证 CPU 可在合理时间内跑通；若使用 GPU 或想追求更高性能，可提高到 `0.1` 再试。

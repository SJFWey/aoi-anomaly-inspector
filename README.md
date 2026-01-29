# AOI Anomaly Inspector

工业外观异常检测（AOI）pipeline，支持 **PaDiM** 与 **PatchCore** 两种模型，输出 **OK/NG 决策** 与 **缺陷定位** 结果。

基于 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 数据集开发，可一键推理任意图片文件夹，输出 overlay 可视化、defect mask、结构化 JSON 报告。

## 两模型对比

### Metrics Comparison (transistor)

| Metric | PaDiM | PatchCore |
|--------|-------|-----------|
| **Image AUROC** | 0.9108 | **0.9954** |
| **Pixel AUROC** | **0.9685** | 0.9607 |
| **Pixel AUPRO** | 0.8377 | **0.9356** |

- **Image Size**: 256×256
- **Device**: CPU
- **Backbone**: ResNet-18 (layer2, layer3)

> **结论**: PatchCore 在图像级检测 (Image AUROC) 和定位质量 (AUPRO) 上表现更优；PaDiM 在像素级分割 (Pixel AUROC) 上略有优势。

### 样例可视化

下图展示两个模型在相同图像上的异常热力图对比（overlay），左列为 PaDiM，右列为 PatchCore。

#### Good Samples (正常品)

<table>
<tr>
<th>PaDiM</th>
<th>PatchCore</th>
</tr>
<tr>
<td><img src="comparison_samples/padim/good_01_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/good_01_overlay.png" width="200"/></td>
</tr>
<tr>
<td><img src="comparison_samples/padim/good_02_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/good_02_overlay.png" width="200"/></td>
</tr>
</table>

#### Defect Samples (缺陷品)

<table>
<tr>
<th>缺陷类型</th>
<th>PaDiM</th>
<th>PatchCore</th>
</tr>
<tr>
<td>cut_lead (轻度)</td>
<td><img src="comparison_samples/padim/defect_mild_cut_lead_01_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/defect_mild_cut_lead_01_overlay.png" width="200"/></td>
</tr>
<tr>
<td>damaged_case (轻度)</td>
<td><img src="comparison_samples/padim/defect_mild_damaged_case_02_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/defect_mild_damaged_case_02_overlay.png" width="200"/></td>
</tr>
<tr>
<td>bent_lead (中度)</td>
<td><img src="comparison_samples/padim/defect_medium_bent_lead_03_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/defect_medium_bent_lead_03_overlay.png" width="200"/></td>
</tr>
<tr>
<td>bent_lead (中度)</td>
<td><img src="comparison_samples/padim/defect_medium_bent_lead_04_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/defect_medium_bent_lead_04_overlay.png" width="200"/></td>
</tr>
<tr>
<td>bent_lead (重度)</td>
<td><img src="comparison_samples/padim/defect_severe_bent_lead_05_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/defect_severe_bent_lead_05_overlay.png" width="200"/></td>
</tr>
<tr>
<td>misplaced (重度)</td>
<td><img src="comparison_samples/padim/defect_severe_misplaced_06_overlay.png" width="200"/></td>
<td><img src="comparison_samples/patchcore/defect_severe_misplaced_06_overlay.png" width="200"/></td>
</tr>
</table>

---

## 快速开始

### 环境

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

## 训练

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

## 评估与阈值

对某个 run 目录重新计算指标并生成阈值：

```bash
# 计算 metrics.json（Image AUROC / Pixel AUROC / Pixel AUPRO）
python scripts/evaluate.py --run_dir runs/padim/transistor/smoke_padim2 --device cpu

# 同时生成 thresholds.json（用于 OK/NG 判定）
python scripts/evaluate.py --run_dir runs/padim/transistor/smoke_padim2 --device cpu --compute-thresholds
```

### 阈值策略

| 阈值 | 用途 | 计算方式 |
|-----|------|---------|
| `image_threshold` | OK/NG 判定 | 训练集 image score 的 99.5 分位数 |
| `pixel_threshold` | 缺陷 mask 生成 | 训练集 anomaly map 的 99.9 分位数 |

> 阈值比较使用原始值（非归一化），归一化仅用于可视化。

## 推理（生成 overlay / mask / JSON）

```bash
python scripts/predict.py --run_dir runs/padim/transistor/smoke_padim2 --device cpu
```

输出到 `<run_dir>/predictions/`：
- `preds.jsonl`: 每张图的结构化预测结果
- `masks/`: 二值缺陷 mask
- `overlays/`: 热力图叠加可视化

## AOI Runner（一键推理任意文件夹）

```bash
python scripts/aoi_runner.py \
  --input_dir <your_images/> \
  --output_dir outputs/demo \
  --model_dir runs/patchcore/transistor/smoke_patchcore2 \
  --device cpu
```

### 输出目录结构

```text
outputs/demo/
  overlays/       # 热力图叠加可视化
  masks/          # 二值缺陷 mask
  preds.jsonl     # 每张图的结构化预测
  report.json     # 汇总报告
```

### preds.jsonl 字段说明

```json
{
  "image_path": "path/to/image.png",
  "model": "patchcore",
  "category": "transistor",
  "image_size": [256, 256],
  "gt_label": 1,
  "pred_score": 25.67,
  "image_threshold": 12.02,
  "pixel_threshold": 10.75,
  "label": "NG",
  "is_anomaly": true,
  "num_defects": 3,
  "total_defect_area": 1250,
  "defects": [
    {
      "component_id": 1,
      "area": 800,
      "bbox": [50, 60, 120, 150],
      "centroid": [85.5, 105.2],
      "max_anomaly_value": 25.67,
      "mean_anomaly_value": 18.34
    }
  ],
  "mask_path": "masks/000001_image.png",
  "overlay_path": "overlays/000001_image.png"
}
```

### report.json 字段说明

```json
{
  "model": "patchcore",
  "category": "transistor",
  "run_id": "smoke_patchcore2",
  "thresholds": {
    "image_threshold": 12.02,
    "pixel_threshold": 10.75
  },
  "num_images": 100,
  "num_ng": 60,
  "ng_rate": 0.60,
  "created_at": "2026-01-29T15:00:00Z"
}
```

## 两模型对比脚本

生成对比表和样例可视化：

```bash
python scripts/compare_models.py \
  --run_dir_a runs/padim/transistor/smoke_padim2 \
  --run_dir_b runs/patchcore/transistor/smoke_patchcore2 \
  --output_dir comparison_samples
```

输出：
- `comparison_table.md`: Markdown 格式对比表
- `comparison_samples/padim/`: PaDiM 样例图
- `comparison_samples/patchcore/`: PatchCore 样例图
- `comparison_report.json`: 完整对比报告

## 产物结构

每次训练会输出到：

```text
runs/<model>/<category>/<run_id>/
  config.yaml           # 训练配置快照
  meta.json             # 元信息
  weights/model.ckpt    # 模型权重
  preds_train.jsonl     # 训练集推理分数
  preds_test.jsonl      # 测试集推理分数
  metrics.json          # 评估指标
  thresholds.json       # OK/NG 阈值
  predictions/          # predict.py 输出
    preds.jsonl
    masks/
    overlays/
  logs/                 # TensorBoard 日志
```

## 设计决策

### 阈值策略

- **Image threshold**: 在训练集良品的 image score 上取 99.5 分位数，将误报率控制在固定范围
- **Pixel threshold**: 在训练集良品的 anomaly map 像素值上取 99.9 分位数，过滤噪声

### 后处理流程

1. 用 `pixel_threshold` 对原始 anomaly map 做二值化
2. 连通域分析（8-连通），提取 bbox / 面积 / 质心
3. 可选：过滤小面积缺陷（`--min-defect-area`）

### 工程复现性

- 固定随机种子（seed=42）
- 训练配置快照保存到 `config.yaml`
- 依赖锁定到 `uv.lock`

## 备注

- 本仓库的默认 `data.num_workers=0`：当前环境下 `torch` 多进程 DataLoader 可能触发 `PermissionError`；如果你的环境支持多进程，可以在 `configs/*.yaml` 里调大。
- `PatchCore` 的 `coreset_sampling_ratio` 默认设为 `0.01` 以保证 CPU 可在合理时间内跑通；若使用 GPU 或想追求更高性能，可提高到 `0.1` 再试。

## 数据许可

本项目使用 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 数据集进行开发与验证。该数据集仅限于学术研究用途，商业使用请联系 MVTec 获取授权。

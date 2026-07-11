import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.data import load_mvtec_split
from aoi.postprocess import postprocess_map
from aoi.reporting import summarize_records, write_json, write_jsonl
from aoi.thresholds import fit_thresholds


class AoiPipelineTests(unittest.TestCase):
    def test_tiny_mvtec_fixture_loads_train_and_test(self) -> None:
        root = REPO_ROOT / "examples" / "mvtec_tiny"

        train = load_mvtec_split(root, "bottle", "train")
        test = load_mvtec_split(root, "bottle", "test")

        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)
        self.assertTrue(all(sample.label == "good" for sample in train))
        self.assertEqual({sample.label for sample in test}, {"good", "scratch"})

    def test_threshold_quantiles_are_deterministic(self) -> None:
        thresholds = fit_thresholds(
            image_scores=[0.1, 0.2, 0.3, 0.4],
            anomaly_maps=[np.array([[0.0, 1.0]], dtype=np.float32)],
            quantile_image=0.5,
            quantile_pixel=0.5,
        )

        self.assertAlmostEqual(thresholds.image_threshold, 0.25, places=6)
        self.assertAlmostEqual(thresholds.pixel_threshold, 0.5, places=6)

    def test_postprocess_filters_small_components_and_reports_bbox(self) -> None:
        anomaly_map = np.zeros((10, 10), dtype=np.float32)
        anomaly_map[2:7, 3:8] = 10.0
        anomaly_map[0, 0] = 10.0

        result = postprocess_map(
            anomaly_map,
            pixel_threshold=5.0,
            output_shape=(10, 10),
            min_area=4,
            morph_kernel=1,
        )

        self.assertEqual(len(result.defects), 1)
        self.assertEqual(result.defects[0].bbox, [3, 2, 8, 7])
        self.assertEqual(result.defects[0].area, 25)
        self.assertEqual(int(result.mask.sum() / 255), 25)

    def test_json_report_writers_emit_expected_files(self) -> None:
        records = [
            {
                "decision": "OK",
                "latency_ms": 10.0,
            },
            {
                "decision": "NG",
                "latency_ms": 20.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_json(tmp_path / "report.json", summarize_records(records, "padim", "bottle", "run-1"))
            write_jsonl(tmp_path / "preds.jsonl", records)

            report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
            preds = (tmp_path / "preds.jsonl").read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(report["num_images"], 2)
        self.assertEqual(report["num_ng"], 1)
        self.assertEqual(len(preds), 2)


if __name__ == "__main__":
    unittest.main()


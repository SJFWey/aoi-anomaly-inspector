from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.evaluation import evaluate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate thresholds and evaluate an AOI run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run = evaluate_run(args.run_dir, device=args.device)
    print(f"Thresholds: {run.thresholds}")
    print(f"Metrics: {run.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

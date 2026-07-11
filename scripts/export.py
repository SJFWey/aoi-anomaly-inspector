from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.exporting import export_run
from aoi.artifacts import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained AOI tensor model to ONNX.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    run = export_run(args.run_dir, opset=args.opset)
    print(f"ONNX model: {run.onnx}")
    print(f"SHA-256: {sha256_file(run.onnx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

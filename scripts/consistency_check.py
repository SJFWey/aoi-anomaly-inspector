from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.consistency import check_run_consistency


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify native PyTorch and ONNX Runtime output consistency.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=8)
    args = parser.parse_args()
    result = check_run_consistency(
        args.run_dir, args.input_dir, device=args.device, max_images=args.max_images
    )
    print(f"Consistency report: {args.run_dir / 'export' / 'consistency.json'}")
    print(f"Passed: {result['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

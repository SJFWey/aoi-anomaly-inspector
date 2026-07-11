from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.inference import predict_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict anomaly masks for an image directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    records = predict_directory(
        args.run_dir,
        args.input_dir,
        args.output_dir,
        device=args.device,
        save_mask=args.save_mask,
        save_overlay=args.save_overlay,
    )
    print(f"Wrote {len(records)} predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

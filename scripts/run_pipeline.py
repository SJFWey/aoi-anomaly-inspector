from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete AOI anomaly pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--consistency-input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--max-consistency-images", type=int, default=8)
    args = parser.parse_args()
    run = run_pipeline(
        args.config,
        consistency_input=args.consistency_input,
        data_root=args.data_root,
        category=args.category,
        device=args.device,
        run_root=args.run_root,
        run_id=args.run_id,
        opset=args.opset,
        max_consistency_images=args.max_consistency_images,
    )
    print(f"Run directory: {run.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

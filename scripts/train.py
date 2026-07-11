from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoi.training import train_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit an AOI anomaly model using train/good images only.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--category")
    parser.add_argument("--device")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = train_run(
        args.config,
        data_root=args.data_root,
        category=args.category,
        device=args.device,
        run_root=args.run_root,
        run_id=args.run_id,
    )
    print(f"Run directory: {run.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

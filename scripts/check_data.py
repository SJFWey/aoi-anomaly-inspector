#!/usr/bin/env python
"""Quick sanity check for an anomalib/MVTec-AD style dataset folder.

By default this expects the repo-relative layout described in README:

datasets/mvtech/
  <category>/
    train/good/*.png
    test/<defect>/*.png
    ground_truth/<defect>/*_mask.png

It prints basic counts and writes a `preview.png` under the selected category.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def _pick_defect_type(defect_types: list[str]) -> str | None:
    non_good = [d for d in defect_types if d != "good"]
    if non_good:
        # Prefer common "obvious defect" folders if they exist.
        for preferred in ("broken_large", "crack", "damaged_case", "misplaced"):
            if preferred in non_good:
                return preferred
        return non_good[0]
    return defect_types[0] if defect_types else None


def check_category(category_dir: Path, *, rng: random.Random) -> None:
    print(f"=== Dataset Check: {category_dir} ===")

    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    if not train_good_dir.exists():
        raise FileNotFoundError(f"Missing train/good directory: {train_good_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    train_good = sorted(train_good_dir.glob("*.png"))
    print(f"Train (good): {len(train_good)} images")

    defect_types = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    print("\nTest set distribution:")
    total_test = 0
    for defect_type in defect_types:
        count = len(list((test_dir / defect_type).glob("*.png")))
        total_test += count
        print(f"  - {defect_type}: {count} images")
    print(f"Test total: {total_test} images")

    # Pick samples
    img_good = None
    if train_good:
        good_path = rng.choice(train_good)
        img_good = cv2.imread(str(good_path), cv2.IMREAD_COLOR)
        if img_good is not None:
            print(f"\nGood sample: {good_path.name} (shape={img_good.shape})")
        else:
            print(f"\nWarning: failed to read good sample: {good_path.name}")

    defect_type = _pick_defect_type(defect_types)
    img_defect = None
    mask_path: Path | None = None
    defect_path: Path | None = None
    if defect_type is not None:
        defect_images = sorted((test_dir / defect_type).glob("*.png"))
        if defect_images:
            defect_path = rng.choice(defect_images)
            img_defect = cv2.imread(str(defect_path), cv2.IMREAD_COLOR)
            if img_defect is not None:
                print(f"Defect sample: {defect_path.name} (type={defect_type})")
            else:
                print(f"Warning: failed to read defect sample: {defect_path.name}")

            mask_name = defect_path.name.replace(".png", "_mask.png")
            mask_path = ground_truth_dir / defect_type / mask_name
        else:
            print(f"\nWarning: no images found for defect type: {defect_type}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Dataset Check: {category_dir.name}")

    # Good image
    if img_good is not None:
        axes[0].imshow(cv2.cvtColor(img_good, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Train: good")
    else:
        axes[0].text(0.5, 0.5, "No good sample", ha="center", va="center")
    axes[0].axis("off")

    # Defect image
    if img_defect is not None:
        axes[1].imshow(cv2.cvtColor(img_defect, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Test: {defect_type}")
    else:
        axes[1].text(0.5, 0.5, "No defect sample", ha="center", va="center")
    axes[1].axis("off")

    # Mask
    if mask_path is not None and mask_path.exists():
        img_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img_mask is not None:
            axes[2].imshow(img_mask, cmap="gray")
            axes[2].set_title("Ground truth mask")
            print(f"Mask found: {mask_path.name}")
        else:
            axes[2].text(0.5, 0.5, "Failed to read mask", ha="center", va="center")
            print(f"Warning: failed to read mask: {mask_path}")
    else:
        axes[2].text(0.5, 0.5, "No mask", ha="center", va="center")
        if defect_path is not None:
            print(f"Mask not found for {defect_path.name}: {mask_path}")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = category_dir / "preview.png"
    plt.savefig(save_path)
    print(f"Preview saved: {save_path}")

    # Avoid blocking in headless environments (e.g. CI).
    if plt.get_backend().lower() != "agg":
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick dataset sanity check.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/mvtech"),
        help="Dataset root containing category folders (default: datasets/mvtech)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category name to check. If omitted, checks the first category found.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    base_dir = args.data_root
    if not base_dir.exists():
        print(f"Error: directory does not exist: {base_dir}")
        return 1

    # If the root itself looks like a category folder (has train/), check it directly.
    if (base_dir / "train").exists():
        category_dir = base_dir
    else:
        categories = sorted(d for d in base_dir.iterdir() if d.is_dir())
        if not categories:
            print(f"Error: no categories found under: {base_dir}")
            return 1

        if args.category is not None:
            category_dir = base_dir / args.category
            if not category_dir.exists():
                print(f"Error: category not found: {category_dir}")
                return 1
        else:
            category_dir = categories[0]

    try:
        check_category(category_dir, rng=rng)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

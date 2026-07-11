import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
DEFAULT_DATA_ROOT = Path("datasets/mvtec")
LEGACY_DATA_ROOT = Path("datasets/mvtech")


def collect_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        images.extend(directory.glob(pattern))
    return sorted(set(images))


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)
    if image is None:
        print(f"Error: failed to read image: {path}")
    return image


def check_images(
    label: str,
    image_paths: list[Path],
) -> tuple[bool, dict[Path, tuple[int, int]]]:
    if not image_paths:
        print(f"Error: no images found for {label}.")
        return False, {}

    unreadable = 0
    sizes: dict[Path, tuple[int, int]] = {}
    for image_path in image_paths:
        image = read_image(image_path)
        if image is None:
            unreadable += 1
            continue
        sizes[image_path] = image.shape[:2]

    print(f"{label}: checked={len(image_paths)}, unreadable={unreadable}")
    return unreadable == 0, sizes


def mask_path_for(image_path: Path, ground_truth_dir: Path, defect_type: str) -> Path:
    return ground_truth_dir / defect_type / f"{image_path.stem}_mask.png"


def check_masks(
    test_dir: Path,
    ground_truth_dir: Path,
    defect_types: list[str],
    image_sizes: dict[Path, tuple[int, int]],
) -> bool:
    defect_types = [defect_type for defect_type in defect_types if defect_type != "good"]
    if not defect_types:
        print("Error: no defect categories found under test.")
        return False

    if not ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found at {ground_truth_dir}")

    expected = 0
    missing = 0
    unreadable = 0
    size_mismatch = 0
    empty = 0

    for defect_type in defect_types:
        for image_path in collect_images(test_dir / defect_type):
            expected += 1
            mask_path = mask_path_for(image_path, ground_truth_dir, defect_type)
            if not mask_path.exists():
                missing += 1
                print(f"Error: missing mask for defect image {image_path}: {mask_path}")
                continue
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                unreadable += 1
                continue

            image_size = image_sizes.get(image_path)
            if image_size is not None and mask.shape[:2] != image_size:
                size_mismatch += 1
                print(
                    "Error: mask size mismatch for defect image "
                    f"{image_path}: image={image_size}, mask={mask.shape[:2]}, "
                    f"mask_path={mask_path}"
                )
            if cv2.countNonZero(mask) == 0:
                empty += 1
                print(
                    "Error: empty/all-black mask for defect image "
                    f"{image_path}: {mask_path}"
                )

    print(
        "\nMask Check: "
        f"expected={expected}, missing={missing}, unreadable={unreadable}, "
        f"size_mismatch={size_mismatch}, empty={empty}"
    )
    return (
        expected > 0
        and missing == 0
        and unreadable == 0
        and size_mismatch == 0
        and empty == 0
    )


def check_category(
    category_dir: Path,
    rng: random.Random,
    show: bool = False,
    preview_out: Path | None = None,
) -> bool:
    print(f"=== Dataset Check: {category_dir} ===")
    ok = True

    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    if not train_good_dir.exists():
        print(f"Error: Train directory not found at {train_good_dir}")
        return False

    train_good = collect_images(train_good_dir)
    print(f"Train (Good): {len(train_good)} images")

    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}")
        return False

    defect_types = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    total_test = 0
    test_image_sizes: dict[Path, tuple[int, int]] = {}

    print("\nTest Set Distribution:")
    for dtype in defect_types:
        imgs = collect_images(test_dir / dtype)
        count = len(imgs)
        total_test += count
        print(f"  - {dtype}: {count} images")

    print(f"Test Total: {total_test} images")

    print("\n=== Full Image Readability Check ===")
    train_ok, _ = check_images("train/good", train_good)
    ok = train_ok and ok

    if "good" not in defect_types:
        print(f"Error: Test good directory not found at {test_dir / 'good'}")
        ok = False

    for dtype in defect_types:
        imgs = collect_images(test_dir / dtype)
        dtype_ok, dtype_sizes = check_images(f"test/{dtype}", imgs)
        test_image_sizes.update(dtype_sizes)
        ok = dtype_ok and ok

    ok = check_masks(test_dir, ground_truth_dir, defect_types, test_image_sizes) and ok

    print("\n=== Sampling Check ===")

    img_good = None
    if train_good:
        sample_good = rng.choice(train_good)
        img_good = read_image(sample_good)
        if img_good is not None:
            print(f"Good sample: {sample_good.name}, Size: {img_good.shape}")
        else:
            ok = False
    else:
        print("No good training images found.")
        ok = False

    img_defect = None
    defect_cat = None
    mask_path = None
    img_mask = None

    potential_defects = [dt for dt in defect_types if dt != "good"]

    if "broken_large" in potential_defects:
        defect_cat = "broken_large"
    elif potential_defects:
        defect_cat = potential_defects[0]

    if defect_cat:
        defect_imgs = collect_images(test_dir / defect_cat)
        if defect_imgs:
            sample_defect = rng.choice(defect_imgs)
            img_defect = read_image(sample_defect)
            mask_path = mask_path_for(sample_defect, ground_truth_dir, defect_cat)
            print(f"Defect sample: {sample_defect.name}, Category: {defect_cat}")
            if img_defect is None:
                ok = False
        else:
            print(f"No images found for defect category: {defect_cat}")
    else:
        print("No defect categories found.")

    if mask_path and mask_path.exists():
        img_mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_mask is not None:
            print(f"Mask found: {mask_path.name}")
        else:
            ok = False
    elif mask_path:
        print(f"Mask not found: {mask_path}")
        ok = False
    else:
        print("No mask path determined")

    if show or preview_out:
        render_preview(
            category_dir,
            img_good,
            img_defect,
            img_mask,
            defect_cat,
            show,
            preview_out,
        )

    return ok


def render_preview(
    category_dir: Path,
    img_good,
    img_defect,
    img_mask,
    defect_cat: str | None,
    show: bool,
    preview_out: Path | None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Dataset Check: {category_dir.name}")

    if img_good is not None:
        axes[0].imshow(cv2.cvtColor(img_good, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Train: Good")
    axes[0].axis("off")

    if img_defect is not None:
        axes[1].imshow(cv2.cvtColor(img_defect, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Test: {defect_cat}")
    else:
        axes[1].text(0.5, 0.5, "No Defect Sample", ha="center")
    axes[1].axis("off")

    if img_mask is not None:
        axes[2].imshow(img_mask, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
    else:
        axes[2].text(0.5, 0.5, "No Mask Found", ha="center")
    axes[2].axis("off")

    plt.tight_layout()
    if preview_out:
        preview_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_out, dpi=150)
        print(f"Preview saved: {preview_out}")
    if show:
        plt.show()
    plt.close(fig)


def resolve_data_root(data_root: Path) -> Path:
    if data_root == DEFAULT_DATA_ROOT and not data_root.exists() and LEGACY_DATA_ROOT.exists():
        print(
            f"Warning: default {DEFAULT_DATA_ROOT} not found; using legacy "
            f"path {LEGACY_DATA_ROOT}. Prefer renaming the directory to mvtec."
        )
        return LEGACY_DATA_ROOT
    return data_root


def preview_path_for(
    preview_out: Path | None,
    category_dir: Path,
    multiple: bool,
) -> Path | None:
    if preview_out is None:
        return None
    if not multiple:
        return preview_out
    if preview_out.suffix:
        return preview_out.with_name(
            f"{preview_out.stem}_{category_dir.name}{preview_out.suffix}"
        )
    return preview_out / f"{category_dir.name}.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an MVTec AD-style dataset before training."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root or a single category directory. Default: datasets/mvtec",
    )
    parser.add_argument(
        "--category",
        help="Optional category name under data-root. By default all categories are checked.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--preview-out",
        type=Path,
        help="Optional path for a saved sampling preview image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the sampling preview window.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = resolve_data_root(args.data_root)

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return 1

    if (base_dir / "train").exists():
        category_dirs = [base_dir]
    else:
        categories = sorted(d for d in base_dir.iterdir() if d.is_dir())
        if args.category:
            category_dir = base_dir / args.category
            if not category_dir.exists():
                print(f"Error: Category {args.category} not found under {base_dir}.")
                return 1
            category_dirs = [category_dir]
        else:
            category_dirs = categories

    if not category_dirs:
        print(f"No categories found in {base_dir}")
        return 1

    category_names = ", ".join(c.name for c in category_dirs)
    print(f"Found {len(category_dirs)} categories: {category_names}")

    rng = random.Random(args.seed)
    multiple = len(category_dirs) > 1
    all_ok = True
    for category_dir in category_dirs:
        preview_out = preview_path_for(args.preview_out, category_dir, multiple)
        all_ok = check_category(category_dir, rng, args.show, preview_out) and all_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

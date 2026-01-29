import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import random


def check_category(category_dir: Path):
    print(f"=== Dataset Check: {category_dir} ===")

    # 1. Statistics
    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    if not train_good_dir.exists():
        print(f"Error: Train directory not found at {train_good_dir}")
        return

    train_good = list(train_good_dir.glob("*.png"))
    print(f"Train (Good): {len(train_good)} images")

    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}")
        return

    defect_types = [d.name for d in test_dir.iterdir() if d.is_dir()]
    total_test = 0

    print("\nTest Set Distribution:")
    for dtype in defect_types:
        imgs = list((test_dir / dtype).glob("*.png"))
        count = len(imgs)
        total_test += count
        print(f"  - {dtype}: {count} images")

    print(f"Test Total: {total_test} images")

    # 2. Random sampling
    print("\n=== Random Sampling Check ===")

    # Sample a good image
    img_good = None
    if train_good:
        sample_good = random.choice(train_good)
        img_good = cv2.imread(str(sample_good))
        if img_good is not None:
            print(f"Good sample: {sample_good.name}, Size: {img_good.shape}")
        else:
            print(f"Failed to read good sample: {sample_good.name}")
    else:
        print("No good training images found.")

    # Sample a defective image
    # Prioritize obvious defects like 'broken_large' or 'crack' etc
    img_defect = None
    defect_cat = None
    mask_path = None

    potential_defects = [dt for dt in defect_types if dt != "good"]

    if "broken_large" in potential_defects:
        defect_cat = "broken_large"
    elif potential_defects:
        defect_cat = potential_defects[0]
    else:
        # If only 'good' exists in test or list is empty
        defect_cat = defect_types[0] if defect_types else None

    if defect_cat:
        defect_imgs = list((test_dir / defect_cat).glob("*.png"))
        if defect_imgs:
            sample_defect = random.choice(defect_imgs)
            img_defect = cv2.imread(str(sample_defect))

            # Find corresponding Mask
            # MVTec AD mask naming convention usually: 000.png -> 000_mask.png
            mask_name = sample_defect.name.replace(".png", "_mask.png")

            # Ground truth is typically in ground_truth/<defect_type>/<mask_name>
            mask_path = ground_truth_dir / defect_cat / mask_name

            print(f"Defect sample: {sample_defect.name}, Category: {defect_cat}")
        else:
            print(f"No images found for defect category: {defect_cat}")
    else:
        print("No defect categories found.")

    # 3. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Dataset Check: {category_dir.name}")

    # Display good sample
    if img_good is not None:
        axes[0].imshow(cv2.cvtColor(img_good, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Train: Good")
    axes[0].axis("off")

    # Display defect sample
    if img_defect is not None:
        axes[1].imshow(cv2.cvtColor(img_defect, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Test: {defect_cat}")
    else:
        axes[1].text(0.5, 0.5, "No Defect Sample", ha="center")
    axes[1].axis("off")

    # Display Mask (if exists)
    if mask_path and mask_path.exists():
        img_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        axes[2].imshow(img_mask, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
        print(f"Mask found: {mask_path.name}")
    else:
        axes[2].text(0.5, 0.5, "No Mask Found", ha="center")
        if mask_path:
            print(f"Mask not found: {mask_path}")
        else:
            print("No mask path determined")

    axes[2].axis("off")

    plt.tight_layout()
    save_path = category_dir / "preview.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    if plt.get_backend() != "Agg":
        plt.show()


def check_data():
    base_dir = Path("./datasets/mvtech")

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return

    # Check if base_dir itself is a category (has 'train' subdir)
    if (base_dir / "train").exists():
        check_category(base_dir)
        return

    # Check for subdirectories (categories)
    categories = [d for d in base_dir.iterdir() if d.is_dir()]
    if not categories:
        print(f"No categories found in {base_dir}")
        return

    print(
        f"Found {len(categories)} categories: {', '.join([c.name for c in categories])}"
    )

    # Just check the first one effectively
    cat = categories[0]
    print(f"Checking first category: {cat.name} ...")
    check_category(cat)


if __name__ == "__main__":
    check_data()

"""Generate model comparison table and sample visualizations.

This script compares metrics from two models and generates:
1. A comparison table in Markdown format
2. Sample visualizations showing good/defect images with overlays from both models
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_metrics(run_dir: Path) -> dict[str, Any]:
    """Load metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    with open(metrics_path) as f:
        return json.load(f)


def load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    """Load predictions from preds.jsonl in predictions directory."""
    preds_path = run_dir / "predictions" / "preds.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions in {run_dir}")
    predictions = []
    with open(preds_path) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def generate_comparison_table(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    model_a: str,
    model_b: str,
) -> str:
    """Generate a Markdown comparison table."""
    category = metrics_a.get("category", "unknown")

    # Format metrics to 4 decimal places
    def fmt(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    # Build table
    lines = [
        f"### Metrics Comparison ({category})",
        "",
        "| Metric | " + model_a.upper() + " | " + model_b.upper() + " |",
        "|--------|" + "-" * (len(model_a) + 2) + "|" + "-" * (len(model_b) + 2) + "|",
        f"| Image AUROC | {fmt(metrics_a.get('image_AUROC'))} | {fmt(metrics_b.get('image_AUROC'))} |",
        f"| Pixel AUROC | {fmt(metrics_a.get('pixel_AUROC'))} | {fmt(metrics_b.get('pixel_AUROC'))} |",
        f"| Pixel AUPRO | {fmt(metrics_a.get('pixel_AUPRO'))} | {fmt(metrics_b.get('pixel_AUPRO'))} |",
        "",
        f"**Image Size**: {metrics_a.get('image_size', 'N/A')}  ",
        f"**Device**: {metrics_a.get('device', 'N/A')}",
        "",
    ]
    return "\n".join(lines)


def select_sample_images(
    predictions: list[dict[str, Any]],
    num_good: int = 2,
    num_defect: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    """Select sample images for visualization.

    Selects:
    - Good images (gt_label=0, label=OK)
    - Defect images sorted by pred_score (mix of mild and severe)
    """
    good_samples = []
    defect_samples = []

    for pred in predictions:
        if pred.get("gt_label") == 0:
            good_samples.append(pred)
        else:
            defect_samples.append(pred)

    # Sort defects by pred_score to get a range of severity
    defect_samples.sort(key=lambda x: x.get("pred_score", 0))

    # Select mild, medium, and severe defects
    selected_defects = []
    if len(defect_samples) >= num_defect:
        # Pick from different severity ranges
        indices = np.linspace(0, len(defect_samples) - 1, num_defect, dtype=int)
        selected_defects = [defect_samples[i] for i in indices]
    else:
        selected_defects = defect_samples[:num_defect]

    return {
        "good": good_samples[:num_good],
        "defect": selected_defects,
    }


def copy_sample_images(
    samples: dict[str, list[dict[str, Any]]],
    run_dir: Path,
    output_dir: Path,
    model_name: str,
) -> list[dict[str, Any]]:
    """Copy sample overlay and mask images to output directory."""
    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    sample_info = []
    pred_dir = run_dir / "predictions"

    for category, items in samples.items():
        for i, pred in enumerate(items):
            overlay_src = pred_dir / pred.get("overlay_path", "")
            mask_src = pred_dir / pred.get("mask_path", "")

            # Generate descriptive filename
            defect_type = Path(pred.get("image_path", "")).parent.name
            severity = "mild" if i < 2 else ("medium" if i < 4 else "severe")
            if category == "good":
                base_name = f"good_{i + 1:02d}"
            else:
                base_name = f"defect_{severity}_{defect_type}_{i + 1:02d}"

            # Copy files
            if overlay_src.exists():
                overlay_dst = model_out / f"{base_name}_overlay.png"
                shutil.copy2(overlay_src, overlay_dst)

            if mask_src.exists():
                mask_dst = model_out / f"{base_name}_mask.png"
                shutil.copy2(mask_src, mask_dst)

            sample_info.append(
                {
                    "category": category,
                    "defect_type": defect_type if category == "defect" else "good",
                    "pred_score": pred.get("pred_score"),
                    "label": pred.get("label"),
                    "num_defects": pred.get("num_defects", 0),
                    "original_path": pred.get("image_path"),
                    "overlay_name": f"{base_name}_overlay.png",
                    "mask_name": f"{base_name}_mask.png",
                }
            )

    return sample_info


def create_comparison_grid(
    output_dir: Path,
    model_a: str,
    model_b: str,
    samples_a: list[dict[str, Any]],
    samples_b: list[dict[str, Any]],
) -> None:
    """Create side-by-side comparison grids of overlays from both models."""
    grid_dir = output_dir / "comparison_grids"
    grid_dir.mkdir(exist_ok=True)

    # Match samples by original path
    samples_b_map = {s["original_path"]: s for s in samples_b}

    for sample_a in samples_a:
        orig_path = sample_a["original_path"]
        if orig_path not in samples_b_map:
            continue
        sample_b = samples_b_map[orig_path]

        # Load overlays
        overlay_a_path = output_dir / model_a / sample_a["overlay_name"]
        overlay_b_path = output_dir / model_b / sample_b["overlay_name"]

        if not overlay_a_path.exists() or not overlay_b_path.exists():
            continue

        img_a = cv2.imread(str(overlay_a_path))
        img_b = cv2.imread(str(overlay_b_path))

        if img_a is None or img_b is None:
            continue

        # Resize if needed
        h = max(img_a.shape[0], img_b.shape[0])
        w = max(img_a.shape[1], img_b.shape[1])

        if img_a.shape[:2] != (h, w):
            img_a = cv2.resize(img_a, (w, h))
        if img_b.shape[:2] != (h, w):
            img_b = cv2.resize(img_b, (w, h))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_a, model_a.upper(), (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(img_b, model_b.upper(), (10, 30), font, 0.8, (255, 255, 255), 2)

        # Concatenate side by side
        combined = np.hstack([img_a, img_b])

        # Save
        out_name = sample_a["overlay_name"].replace("_overlay.png", "_comparison.png")
        cv2.imwrite(str(grid_dir / out_name), combined)


def generate_sample_markdown(
    samples_a: list[dict[str, Any]],
    samples_b: list[dict[str, Any]],
    model_a: str,
    model_b: str,
    output_dir: Path,
) -> str:
    """Generate markdown section showing sample comparisons."""
    lines = [
        "### Sample Visualizations",
        "",
        "Below are sample overlay comparisons between the two models.",
        "",
    ]

    # Group by category
    good_samples = [s for s in samples_a if s["category"] == "good"]
    defect_samples = [s for s in samples_a if s["category"] == "defect"]

    # Good samples
    if good_samples:
        lines.append("#### Good Samples (No Defects)")
        lines.append("")
        lines.append(f"| {model_a.upper()} | {model_b.upper()} |")
        lines.append("|:---:|:---:|")
        for s in good_samples:
            lines.append(
                f"| ![{s['overlay_name']}](comparison_samples/{model_a}/{s['overlay_name']}) "
                f"| ![{s['overlay_name']}](comparison_samples/{model_b}/{s['overlay_name']}) |"
            )
        lines.append("")

    # Defect samples
    if defect_samples:
        lines.append("#### Defect Samples")
        lines.append("")
        lines.append(f"| Type | {model_a.upper()} | {model_b.upper()} |")
        lines.append("|:---|:---:|:---:|")
        for s in defect_samples:
            defect_type = s["defect_type"]
            lines.append(
                f"| {defect_type} "
                f"| ![{s['overlay_name']}](comparison_samples/{model_a}/{s['overlay_name']}) "
                f"| ![{s['overlay_name']}](comparison_samples/{model_b}/{s['overlay_name']}) |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model comparison table and sample visualizations."
    )
    parser.add_argument(
        "--run_dir_a",
        type=Path,
        required=True,
        help="Run directory for first model (e.g., runs/padim/transistor/smoke_padim2)",
    )
    parser.add_argument(
        "--run_dir_b",
        type=Path,
        required=True,
        help="Run directory for second model (e.g., runs/patchcore/transistor/smoke_patchcore2)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("comparison_samples"),
        help="Output directory for comparison artifacts (default: comparison_samples)",
    )
    parser.add_argument(
        "--num-good",
        type=int,
        default=2,
        help="Number of good sample images (default: 2)",
    )
    parser.add_argument(
        "--num-defect",
        type=int,
        default=6,
        help="Number of defect sample images (default: 6)",
    )
    args = parser.parse_args()

    # Extract model names from paths
    model_a = args.run_dir_a.parts[-3]  # e.g., "padim"
    model_b = args.run_dir_b.parts[-3]  # e.g., "patchcore"

    print(f"Comparing {model_a.upper()} vs {model_b.upper()}")

    # Load metrics
    print("Loading metrics...")
    metrics_a = load_metrics(args.run_dir_a)
    metrics_b = load_metrics(args.run_dir_b)

    # Generate comparison table
    table_md = generate_comparison_table(metrics_a, metrics_b, model_a, model_b)
    print("\n" + "=" * 60)
    print(table_md)
    print("=" * 60)

    # Try to load predictions for sample selection
    try:
        print("\nLoading predictions for sample selection...")
        preds_a = load_predictions(args.run_dir_a)
        preds_b = load_predictions(args.run_dir_b)

        # Select samples
        samples_a = select_sample_images(preds_a, args.num_good, args.num_defect)
        samples_b = select_sample_images(preds_b, args.num_good, args.num_defect)

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy sample images
        print(f"Copying sample images to {args.output_dir}...")
        sample_info_a = copy_sample_images(
            samples_a, args.run_dir_a, args.output_dir, model_a
        )
        sample_info_b = copy_sample_images(
            samples_b, args.run_dir_b, args.output_dir, model_b
        )

        # Create comparison grids
        print("Creating comparison grids...")
        create_comparison_grid(
            args.output_dir, model_a, model_b, sample_info_a, sample_info_b
        )

        # Generate sample markdown
        sample_md = generate_sample_markdown(
            sample_info_a, sample_info_b, model_a, model_b, args.output_dir
        )

        # Save comparison report
        report = {
            "model_a": model_a,
            "model_b": model_b,
            "metrics": {
                model_a: metrics_a,
                model_b: metrics_b,
            },
            "samples": {
                model_a: sample_info_a,
                model_b: sample_info_b,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        report_path = args.output_dir / "comparison_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved comparison report to {report_path}")

        # Save markdown snippets
        table_path = args.output_dir / "comparison_table.md"
        with open(table_path, "w") as f:
            f.write(table_md)
        print(f"Saved comparison table to {table_path}")

        sample_md_path = args.output_dir / "sample_visualizations.md"
        with open(sample_md_path, "w") as f:
            f.write(sample_md)
        print(f"Saved sample visualizations markdown to {sample_md_path}")

    except FileNotFoundError as e:
        print(f"\nWarning: {e}")
        print("Sample visualizations require running predict.py on both models first.")
        print("Only metrics comparison table generated.")

        # Save just the table
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table_path = args.output_dir / "comparison_table.md"
        with open(table_path, "w") as f:
            f.write(table_md)
        print(f"Saved comparison table to {table_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

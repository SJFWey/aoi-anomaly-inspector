from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "scripts/aoi_runner.py",
    "scripts/run_experiment.py",
    "weights/model.ckpt",
    "preds_train.jsonl",
    "preds_test.jsonl",
)


def test_legacy_entry_points_are_absent() -> None:
    assert not (REPO_ROOT / "scripts" / "aoi_runner.py").exists()
    assert not (REPO_ROOT / "scripts" / "run_experiment.py").exists()


def test_tracked_text_does_not_reference_old_contracts() -> None:
    roots = [REPO_ROOT / "README.md", REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "configs"]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".toml"}
    )
    for forbidden in FORBIDDEN:
        assert forbidden not in text

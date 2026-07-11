import contextlib
import importlib
import importlib.util
import io
import struct
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class FakeImage:
    def __init__(self, width: int, height: int, channels: int, nonzero: bool) -> None:
        self.shape = (height, width) if channels == 1 else (height, width, channels)
        self.nonzero = nonzero


def parse_test_png(path: Path, channels: int) -> FakeImage | None:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None

    offset = 8
    width = 0
    height = 0
    idat = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            break

    raw = zlib.decompress(bytes(idat))
    pixels = bytearray()
    for row_start in range(0, len(raw), width + 1):
        pixels.extend(raw[row_start + 1 : row_start + 1 + width])
    return FakeImage(width, height, channels, any(pixels))


def install_fake_cv2() -> None:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.IMREAD_GRAYSCALE = 0
    fake_cv2.COLOR_BGR2RGB = 4

    def imread(path: str, flags: int = fake_cv2.IMREAD_COLOR) -> FakeImage | None:
        channels = 1 if flags == fake_cv2.IMREAD_GRAYSCALE else 3
        return parse_test_png(Path(path), channels)

    def count_non_zero(image: FakeImage) -> int:
        return 1 if image.nonzero else 0

    fake_cv2.imread = imread
    fake_cv2.countNonZero = count_non_zero
    sys.modules["cv2"] = fake_cv2


def install_fake_matplotlib() -> None:
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = None
    fake_pyplot.tight_layout = None
    fake_pyplot.show = None
    fake_pyplot.close = None
    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot


if importlib.util.find_spec("cv2") is None:
    install_fake_cv2()
if importlib.util.find_spec("matplotlib") is None:
    install_fake_matplotlib()

check_data = importlib.import_module("scripts.check_data")


def png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_png(
    path: Path,
    width: int = 4,
    height: int = 4,
    value: int = 127,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    row = b"\x00" + bytes([value]) * width
    raw = row * height
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


def make_valid_dataset(root: Path) -> tuple[Path, Path]:
    category_dir = root / "bottle"
    write_png(category_dir / "train" / "good" / "000.png")
    write_png(category_dir / "test" / "good" / "000.png")
    write_png(category_dir / "test" / "scratch" / "000.png")
    mask_path = category_dir / "ground_truth" / "scratch" / "000_mask.png"
    write_png(mask_path, value=255)
    return category_dir, mask_path


def run_check(category_dir: Path) -> tuple[int, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exit_code = check_data.main(["--data-root", str(category_dir)])
    return exit_code, output.getvalue()


class CheckDataTests(unittest.TestCase):
    def test_missing_mask_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            category_dir, mask_path = make_valid_dataset(Path(tmp))
            mask_path.unlink()

            exit_code, output = run_check(category_dir)

            self.assertEqual(exit_code, 1)
            self.assertIn("missing mask", output)

    def test_bad_image_fails_even_when_not_sampled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            category_dir, _ = make_valid_dataset(Path(tmp))
            bad_image = category_dir / "test" / "good" / "999.png"
            bad_image.write_text("not an image")

            exit_code, output = run_check(category_dir)

            self.assertEqual(exit_code, 1)
            self.assertIn("failed to read image", output)
            self.assertIn("test/good: checked=2, unreadable=1", output)

    def test_mask_size_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            category_dir, mask_path = make_valid_dataset(Path(tmp))
            write_png(mask_path, width=2, height=3, value=255)

            exit_code, output = run_check(category_dir)

            self.assertEqual(exit_code, 1)
            self.assertIn("mask size mismatch", output)

    def test_all_black_mask_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            category_dir, mask_path = make_valid_dataset(Path(tmp))
            write_png(mask_path, value=0)

            exit_code, output = run_check(category_dir)

            self.assertEqual(exit_code, 1)
            self.assertIn("empty/all-black mask", output)


if __name__ == "__main__":
    unittest.main()

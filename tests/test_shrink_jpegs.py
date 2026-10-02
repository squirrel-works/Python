import io
import random
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageFilter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shrink_jpegs as sj  # noqa: E402


def photo_like(size: tuple[int, int], seed: int = 0) -> Image.Image:
    """Blurred noise -- compresses roughly like a photo, unlike flat colour."""
    w, h = size
    small = Image.frombytes(
        "RGB", (w // 4, h // 4), random.Random(seed).randbytes((w // 4) * (h // 4) * 3)
    )
    return small.resize(size, Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))


def png_roundtrip(img: Image.Image, **save_kwargs) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG", **save_kwargs)
    buf.seek(0)
    return Image.open(buf)


def fake_icc(colour_space: bytes) -> bytes:
    # Only the header's colour space signature (bytes 16-19) matters here.
    return b"\0" * 16 + colour_space + b"\0" * 108


def run_main(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["shrink_jpegs.py", *map(str, argv)])
    return sj.main()


# --- compress_to_target -----------------------------------------------------


def test_reencode_applies_exif_orientation(tmp_path: Path) -> None:
    src = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, displayed portrait
    photo_like((400, 200)).save(src, quality=95, exif=exif.tobytes())
    out = tmp_path / "out.jpg"

    ok, _ = sj.compress_to_target(src, out, src.stat().st_size // 2)

    assert ok
    with Image.open(out) as img:
        assert img.width < img.height


def test_heic_orientation_not_applied_twice(tmp_path: Path) -> None:
    pytest.importorskip("pillow_heif")
    src = tmp_path / "rotated.heic"
    exif = Image.Exif()
    exif[0x0112] = 6
    photo_like((400, 200)).save(src, exif=exif.tobytes())
    out = tmp_path / "out.jpg"

    ok, _ = sj.compress_to_target(src, out, 10**6)

    assert ok
    with Image.open(out) as img:
        assert img.width < img.height


def test_png_named_jpg_is_reencoded_not_copied(tmp_path: Path) -> None:
    src = tmp_path / "actually_png.jpg"
    Image.new("RGB", (64, 64), "red").save(src, format="PNG")
    out = tmp_path / "out.jpg"

    ok, _ = sj.compress_to_target(src, out, 10**6)

    assert ok
    with Image.open(out) as img:
        assert img.format == "JPEG"


def test_fitting_jpeg_is_copied_byte_for_byte(tmp_path: Path) -> None:
    src = tmp_path / "small.jpg"
    photo_like((128, 128)).save(src, quality=80)
    out = tmp_path / "out.jpg"

    ok, size = sj.compress_to_target(src, out, src.stat().st_size)

    assert ok
    assert out.read_bytes() == src.read_bytes()
    assert size == src.stat().st_size


def test_heavy_downscale_still_reaches_target(tmp_path: Path) -> None:
    # Needs far more than the ~0.35x linear reduction that ten fixed 0.9
    # steps allowed -- previously the image was dropped from the zip.
    src = tmp_path / "big.jpg"
    photo_like((2000, 1500), seed=2).save(src, quality=92)
    out = tmp_path / "out.jpg"
    target = 20_000

    ok, size = sj.compress_to_target(src, out, target)

    assert ok
    assert size == out.stat().st_size <= target
    with Image.open(out) as img:
        assert min(img.size) >= sj.MIN_DIM


def test_unreachable_target_fails_without_writing(tmp_path: Path) -> None:
    src = tmp_path / "big.jpg"
    photo_like((400, 400)).save(src, quality=92)
    out = tmp_path / "out.jpg"

    ok, _ = sj.compress_to_target(src, out, 100)

    assert not ok
    assert list(tmp_path.iterdir()) == [src]


def test_unreadable_image_raises_oserror(tmp_path: Path) -> None:
    src = tmp_path / "bad.png"
    src.write_bytes(b"not an image")

    with pytest.raises(OSError, match="failed to open"):
        sj.compress_to_target(src, tmp_path / "out.jpg", 10**6)


@pytest.mark.parametrize(
    ("mode", "colour_space", "kept"),
    [("RGB", b"RGB ", True), ("CMYK", b"CMYK", False), ("L", b"GRAY", False)],
)
def test_icc_profile_carried_over_only_when_rgb(
    tmp_path: Path, mode: str, colour_space: bytes, kept: bool,
) -> None:
    src = tmp_path / "src.jpg"
    icc = fake_icc(colour_space)
    Image.new(mode, (64, 64)).save(src, quality=95, icc_profile=icc)
    out = tmp_path / "out.jpg"

    ok, _ = sj.compress_to_target(src, out, src.stat().st_size - 1)

    assert ok
    with Image.open(out) as img:
        assert (img.info.get("icc_profile") == icc) is kept


# --- load_as_rgb ------------------------------------------------------------


def _pa_transparent() -> Image.Image:
    img = Image.new("PA", (8, 8))
    img.putpalette([0, 0, 0] * 256)
    return img


@pytest.mark.parametrize(
    "make",
    [
        lambda: png_roundtrip(Image.new("RGBA", (8, 8), (0, 0, 0, 0))),
        lambda: png_roundtrip(Image.new("LA", (8, 8), (0, 0))),
        lambda: png_roundtrip(Image.new("RGB", (8, 8)), transparency=(0, 0, 0)),
        lambda: png_roundtrip(Image.new("P", (8, 8), 0), transparency=0),
        _pa_transparent,
    ],
    ids=["RGBA", "LA", "RGB-colour-key", "P-colour-key", "PA"],
)
def test_transparency_is_flattened_onto_white(make) -> None:
    assert sj.load_as_rgb(make()).getpixel((0, 0)) == (255, 255, 255)


def test_16bit_grayscale_is_scaled_not_clipped() -> None:
    img = png_roundtrip(Image.new("I;16", (8, 8), 32768))

    assert sj.load_as_rgb(img).getpixel((0, 0)) == (128, 128, 128)


# --- zip overhead -----------------------------------------------------------


def test_zip_overhead_is_an_upper_bound(tmp_path: Path) -> None:
    names = [f"dir/{'x' * 100}_{i}.jpg" for i in range(50)] + ["ünï/фото.jpg"]
    blob = tmp_path / "blob"
    blob.write_bytes(b"\xff" * 1000)
    archive = tmp_path / "t.zip"

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in names:
            zf.write(blob, arcname=name)

    data_bytes = len(names) * blob.stat().st_size
    assert archive.stat().st_size <= data_bytes + sj.zip_overhead_bytes(names)


# --- main -------------------------------------------------------------------


def test_main_packs_case_colliding_names_separately(
    tmp_path: Path, monkeypatch,
) -> None:
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    photo_like((800, 600), seed=1).save(src_dir / "photo.JPG", quality=95)
    Image.new("RGB", (800, 600), "blue").save(src_dir / "photo.png")
    out = tmp_path / "dist" / "out.zip"
    target_mb = 0.15

    rc = run_main(
        monkeypatch, src_dir, "-o", out, "--target-zip-mb", target_mb, "--workers", 2,
    )

    assert rc == 0
    assert out.stat().st_size <= target_mb * 1_000_000
    assert [p.name for p in out.parent.iterdir()] == ["out.zip"]
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["photo.JPG", "photo.png.jpg"]
        with Image.open(io.BytesIO(zf.read("photo.png.jpg"))) as from_png:
            r, g, b = from_png.getpixel((0, 0))
            assert b > 200 and r < 50 and g < 50
        with Image.open(io.BytesIO(zf.read("photo.JPG"))) as from_jpg:
            assert from_jpg.size == (800, 600)


def test_main_excludes_unreadable_image_and_reports_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    photo_like((128, 128)).save(src_dir / "good.jpg")
    (src_dir / "bad.png").write_bytes(b"not an image")
    out = tmp_path / "out.zip"

    rc = run_main(monkeypatch, src_dir, "-o", out, "--workers", 2)

    assert rc == 1
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["good.jpg"]


def test_main_refuses_to_overwrite_an_input(tmp_path: Path, monkeypatch) -> None:
    victim = tmp_path / "a.jpg"
    photo_like((64, 64)).save(victim)
    before = victim.read_bytes()

    assert run_main(monkeypatch, tmp_path, "-o", victim) == 1
    assert victim.read_bytes() == before


def test_main_fails_fast_when_target_cannot_hold_zip_metadata(
    tmp_path: Path, monkeypatch,
) -> None:
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    photo_like((64, 64)).save(src_dir / "a.jpg")
    out = tmp_path / "out.zip"

    assert run_main(monkeypatch, src_dir, "-o", out, "--target-zip-mb", 0.04) == 1
    assert not out.exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--target-zip-mb", "0"],
        ["--target-zip-mb", "nan"],
        ["--target-zip-mb", "inf"],
        ["--workers", "0"],
    ],
)
def test_main_rejects_invalid_arguments(tmp_path: Path, monkeypatch, argv) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_main(monkeypatch, tmp_path, *argv)

    assert exc_info.value.code == 2

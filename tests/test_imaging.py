from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from helpers import jpeg_bytes, save_jpeg, texture
from pickframe import imaging
from pickframe.imaging import (
    ImageLoadError,
    jpeg_dimensions,
    largest_embedded_jpeg,
    load_image,
    load_thumbnail,
)


def test_orientation_is_applied(tmp_path: Path):
    rgb = texture((200, 300))  # landscape pixels
    path = save_jpeg(tmp_path / "rot.jpg", rgb, datetime(2024, 1, 1, 12, 0, 0), orientation=6)
    loaded = load_image(path, max_side=1600)
    # Orientation 6 means "rotate 90° clockwise to display": the photo is portrait.
    assert loaded.rgb.shape[:2] == (300, 200)
    assert (loaded.width, loaded.height) == (200, 300)
    assert loaded.exif.orientation == 6
    assert loaded.exif.capture_time is not None
    assert loaded.source == "image"


def test_draft_downscale(tmp_path: Path):
    path = save_jpeg(tmp_path / "big.jpg", texture((1200, 1800)))
    loaded = load_image(path, max_side=400)
    assert max(loaded.rgb.shape[:2]) == 400
    assert (loaded.width, loaded.height) == (1800, 1200)
    assert loaded.scale == pytest.approx(400 / 1800, rel=0.01)
    assert loaded.rgb.dtype == np.uint8
    assert loaded.rgb.flags["C_CONTIGUOUS"]


def test_rgba_png_is_flattened_on_white(tmp_path: Path):
    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[:, :30, 3] = 255  # left half opaque red, right half transparent
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba, "RGBA").save(path)
    rgb = load_image(path).rgb
    assert tuple(rgb[20, 10]) == (255, 0, 0)
    assert tuple(rgb[20, 50]) == (255, 255, 255)


def test_sixteen_bit_grayscale(tmp_path: Path):
    gray = (np.linspace(0, 65535, 64 * 64).reshape(64, 64)).astype(np.uint16)
    path = tmp_path / "deep.png"
    Image.fromarray(gray).save(path)
    rgb = load_image(path).rgb
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert rgb[-1, -1, 0] > 200
    assert rgb[0, 0, 0] < 10


def test_palette_and_grayscale_images(tmp_path: Path):
    path = tmp_path / "pal.gif"
    Image.fromarray(texture((32, 32))).convert("P").save(path.with_suffix(".png"))
    rgb = load_image(path.with_suffix(".png")).rgb
    assert rgb.shape == (32, 32, 3)
    gray_path = tmp_path / "gray.png"
    Image.fromarray(texture((32, 32))[..., 0]).save(gray_path)
    assert load_image(gray_path).rgb.shape == (32, 32, 3)


def test_corrupt_file_raises(tmp_path: Path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not a jpeg")
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_thumbnail_size(tmp_path: Path):
    path = save_jpeg(tmp_path / "t.jpg", texture((600, 900)))
    thumb = load_thumbnail(path, 120)
    assert max(thumb.shape[:2]) == 120


def test_jpeg_dimensions_and_lossless_skip():
    data = jpeg_bytes(texture((48, 80)))
    assert jpeg_dimensions(data, 0) == (80, 48, 0xC0)
    # Patch the SOF marker to "lossless" (0xC3): reported, but not decodable.
    sof = data.find(b"\xff\xc0")
    lossless = data[:sof] + b"\xff\xc3" + data[sof + 2 :]
    assert jpeg_dimensions(lossless, 0)[2] == 0xC3
    assert largest_embedded_jpeg(b"junk" + lossless + b"junk") is None
    assert jpeg_dimensions(b"\xff\xd8\x00\x00", 0) is None
    assert jpeg_dimensions(b"\xff\xd8\xff\xda\x00\x04", 0) is None


def _fake_raw(small: bytes, large: bytes) -> bytes:
    rng = np.random.default_rng(1)
    junk = (
        rng.integers(0, 255, 3000, dtype=np.uint8)
        .tobytes()
        .replace(b"\xff\xd8\xff", b"\x00\x00\x00")
    )
    return b"II*\x00" + junk + small + junk + large + junk


def test_embedded_preview_scanner():
    small = jpeg_bytes(texture((80, 120), seed=1))
    large = jpeg_bytes(texture((400, 600), seed=2))
    found = largest_embedded_jpeg(_fake_raw(small, large))
    assert found is not None
    assert found.startswith(b"\xff\xd8")
    assert found.endswith(b"\xff\xd9")
    assert Image.open(__import__("io").BytesIO(found)).size == (600, 400)
    assert largest_embedded_jpeg(b"no jpeg here at all") is None


def test_raw_fallback_uses_largest_preview(tmp_path: Path):
    small = jpeg_bytes(texture((80, 120), seed=1))
    large = jpeg_bytes(texture((400, 600), seed=2))
    path = tmp_path / "DSC_0001.NEF"
    path.write_bytes(_fake_raw(small, large))
    loaded = load_image(path, max_side=300)
    assert loaded.source == "raw-preview"
    assert (loaded.width, loaded.height) == (600, 400)
    assert max(loaded.rgb.shape[:2]) == 300


def test_raw_preview_with_own_orientation(tmp_path: Path):
    e = Image.Exif()
    e[0x0112] = 6
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(texture((200, 300), seed=3)).save(buffer, format="JPEG", exif=e)
    path = tmp_path / "DSCF0001.ORF"
    path.write_bytes(b"\x00" * 100 + buffer.getvalue())
    loaded = load_image(path)
    assert loaded.rgb.shape[:2] == (300, 200)
    assert (loaded.width, loaded.height) == (200, 300)


def test_raw_without_preview_raises(tmp_path: Path):
    path = tmp_path / "empty.CR2"
    path.write_bytes(b"\x00" * 2048)
    with pytest.raises(ImageLoadError, match="no embedded preview"):
        load_image(path)


def test_raw_fallback_without_rawpy(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)  # import rawpy -> ImportError
    path = tmp_path / "x.arw"
    path.write_bytes(_fake_raw(jpeg_bytes(texture((60, 90))), jpeg_bytes(texture((300, 450)))))
    assert load_image(path).width == 450


def test_heif_needs_plugin(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(imaging, "heif_available", lambda: False)
    path = tmp_path / "x.heic"
    path.write_bytes(b"\x00" * 10)
    with pytest.raises(ImageLoadError, match="pickframe\\[heif\\]"):
        load_image(path)


def test_rotate_for_orientation_all_values():
    im = Image.fromarray(texture((20, 30)))
    sizes = {o: imaging._rotate_for_orientation(im, o).size for o in range(1, 9)}
    assert sizes[1] == (30, 20)
    assert sizes[3] == (30, 20)
    assert all(sizes[o] == (20, 30) for o in (5, 6, 7, 8))


def test_heif_roundtrip(tmp_path: Path, monkeypatch):
    pytest.importorskip("pillow_heif")
    monkeypatch.setattr(imaging, "_heif_state", None)
    assert imaging.heif_available()
    path = tmp_path / "phone.heic"
    try:
        Image.fromarray(texture((120, 160))).save(path, format="HEIF")
    except (KeyError, OSError) as exc:  # wheel built without an encoder
        pytest.skip(f"HEIF encoding unavailable: {exc}")
    loaded = load_image(path)
    assert loaded.rgb.shape[:2] == (120, 160)


def test_heif_plugin_missing(monkeypatch):
    monkeypatch.setattr(imaging, "_heif_state", None)
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    assert imaging.heif_available() is False
    monkeypatch.setattr(imaging, "_heif_state", None)  # let later tests detect it again

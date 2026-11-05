"""Decode photos into small, upright working images.

* JPEGs use Pillow's draft mode, which lets libjpeg decode at 1/2, 1/4 or
  1/8 scale. A 24 MP file decodes several times faster this way.
* EXIF orientation is applied, so faces are upright for the face models.
* Camera raw files are analysed through their embedded full-size JPEG
  preview (what the camera showed on its screen). ``rawpy`` is used when
  installed; otherwise a built-in scanner extracts the largest preview.
"""

from __future__ import annotations

import mmap
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps

from . import exif as exif_mod
from .formats import HEIF_EXTENSIONS, is_raw

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 400_000_000  # large panoramas are fine; decompression bombs are not

_SOF_DECODABLE = {0xC0, 0xC1, 0xC2}  # baseline, extended, progressive Huffman
_SOF_ALL = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_MIN_PREVIEW = 900  # below this long side, prefer demosaicing the raw data (needs rawpy)

_heif_lock = threading.Lock()
_heif_state: bool | None = None


class ImageLoadError(Exception):
    """Raised when a file cannot be decoded."""


@dataclass
class LoadedImage:
    """A decoded, upright working image (RGB uint8) plus metadata."""

    rgb: np.ndarray
    width: int  # full-resolution size of the upright photo
    height: int
    exif: exif_mod.ExifInfo
    source: str = "image"

    @property
    def scale(self) -> float:
        """Working-image pixels per original pixel."""
        return self.rgb.shape[1] / max(1, self.width)


def heif_available() -> bool:
    global _heif_state
    with _heif_lock:
        if _heif_state is None:
            try:
                from pillow_heif import register_heif_opener

                register_heif_opener()
                try:  # AVIF support lives in a separate registration in newer releases
                    from pillow_heif import register_avif_opener

                    register_avif_opener()
                except ImportError:
                    pass
                _heif_state = True
            except ImportError:
                _heif_state = False
        return _heif_state


def _shrink(im: Image.Image, max_side: int) -> Image.Image:
    if max(im.size) > max_side:
        im = im.copy() if im.readonly else im
        im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC, reducing_gap=2.0)
    return im


def _to_rgb_array(im: Image.Image) -> np.ndarray:
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        top = float(arr.max()) or 1.0
        arr = (arr / top * 255.0).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(arr[..., None], 3, axis=2))
    if im.mode != "RGB":
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im.convert("RGBA"), mask=im.convert("RGBA").getchannel("A"))
            im = background
        else:
            im = im.convert("RGB")
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))


def _open_pillow(im: Image.Image, max_side: int, info: exif_mod.ExifInfo, source: str) -> LoadedImage:
    width, height = im.size
    if info.orientation in (5, 6, 7, 8):
        width, height = height, width
    if im.format == "JPEG":
        im.draft("RGB", (max_side, max_side))
    try:
        im = ImageOps.exif_transpose(im) or im
    except Exception:  # broken EXIF must never stop analysis
        pass
    im = _shrink(im, max_side)
    return LoadedImage(rgb=_to_rgb_array(im), width=width, height=height, exif=info, source=source)


def _rotate_for_orientation(im: Image.Image, orientation: int) -> Image.Image:
    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    return im.transpose(method) if method is not None else im


def jpeg_dimensions(data: bytes | mmap.mmap, start: int) -> tuple[int, int, int] | None:
    """Return ``(width, height, sof_marker)`` of the JPEG starting at ``start``."""
    n = len(data)
    pos = start + 2
    for _ in range(64):
        if pos + 4 > n or data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        while marker == 0xFF and pos + 2 < n:  # fill bytes
            pos += 1
            marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        length = (data[pos + 2] << 8) | data[pos + 3]
        if length < 2:
            return None
        if marker in _SOF_ALL:
            if pos + 9 > n:
                return None
            height = (data[pos + 5] << 8) | data[pos + 6]
            width = (data[pos + 7] << 8) | data[pos + 8]
            return width, height, marker
        pos += 2 + length
    return None


def _jpeg_end(data: bytes | mmap.mmap, start: int) -> int:
    """Offset just past the EOI marker of the JPEG at ``start`` (best effort)."""
    pos = start + 2
    n = len(data)
    while pos + 4 <= n and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xDA:
            break
        length = (data[pos + 2] << 8) | data[pos + 3]
        pos += 2 + length
    end = data.find(b"\xff\xd9", pos)
    return n if end < 0 else end + 2


def largest_embedded_jpeg(data: bytes | mmap.mmap) -> bytes | None:
    """Find the biggest decodable JPEG stream inside a raw file."""
    best: tuple[int, int] | None = None  # (area, offset)
    pos = 0
    while True:
        idx = data.find(b"\xff\xd8\xff", pos)
        if idx < 0:
            break
        dims = jpeg_dimensions(data, idx)
        if dims and dims[2] in _SOF_DECODABLE and min(dims[0], dims[1]) >= 64:
            area = dims[0] * dims[1]
            if best is None or area > best[0]:
                best = (area, idx)
        pos = idx + 3
    if best is None:
        return None
    start = best[1]
    return bytes(data[start : _jpeg_end(data, start)])


def _load_raw(path: Path, max_side: int) -> LoadedImage:
    info = exif_mod.read_raw_head(path) or exif_mod.ExifInfo()
    preview: Image.Image | None = None
    orientation = info.orientation
    use_preview_exif = False
    source = "raw-preview"
    try:
        import rawpy
    except ImportError:
        rawpy = None
    if rawpy is not None:
        try:
            with rawpy.imread(str(path)) as raw:
                orientation = {0: 1, 3: 3, 5: 8, 6: 6}.get(int(raw.sizes.flip), orientation)
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        preview = Image.open(BytesIO(thumb.data))
                    else:
                        preview = Image.fromarray(np.asarray(thumb.data, dtype=np.uint8))
                except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                    preview = None
                if preview is None or max(preview.size) < _MIN_PREVIEW:
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                    preview = Image.fromarray(rgb)  # postprocess already applies the flip
                    orientation = 1
                    source = "raw-demosaic"
        except Exception:  # unsupported by LibRaw: fall back to the built-in scanner
            preview = None
            orientation = info.orientation
    if preview is None:
        try:
            with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = largest_embedded_jpeg(mm)
        except (OSError, ValueError) as exc:
            raise ImageLoadError(f"cannot read {path.name}: {exc}") from exc
        if data is None:
            raise ImageLoadError(
                f"{path.name}: no embedded preview found (install 'pickframe[raw]' for full raw support)"
            )
        preview = Image.open(BytesIO(data))
        embedded = exif_mod.read_pillow(preview)
        if embedded.capture_time and not info.capture_time:
            info.capture_time = embedded.capture_time
        if embedded.orientation != 1:
            use_preview_exif = True  # the preview carries its own (authoritative) tag
            orientation = 1
    width, height = preview.size  # full size, before draft-mode downscaling
    if preview.format == "JPEG":
        preview.draft("RGB", (max_side, max_side))
    if use_preview_exif:
        preview = ImageOps.exif_transpose(preview) or preview
    preview = _rotate_for_orientation(preview, orientation)
    if (preview.width > preview.height) != (width > height):
        width, height = height, width
    preview = _shrink(preview, max_side)
    info.orientation = 1
    return LoadedImage(rgb=_to_rgb_array(preview), width=width, height=height, exif=info, source=source)


def load_image(path: Path, max_side: int = 1600) -> LoadedImage:
    """Decode ``path`` to an upright RGB array whose long side is at most ``max_side``."""
    path = Path(path)
    if is_raw(path):
        return _load_raw(path, max_side)
    if path.suffix.lower() in HEIF_EXTENSIONS and not heif_available():
        raise ImageLoadError(f"{path.name}: HEIF/AVIF support needs 'pip install pickframe[heif]'")
    try:
        with Image.open(path) as im:
            info = exif_mod.read_pillow(im)
            return _open_pillow(im, max_side, info, "image")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"cannot decode {path.name}: {exc}") from exc


def load_thumbnail(path: Path, size: int = 320) -> np.ndarray:
    """Small RGB preview for user interfaces and reports."""
    return load_image(path, max_side=size).rgb

"""Image helpers shared by the tests."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def texture(size: tuple[int, int] = (240, 320), seed: int = 0) -> np.ndarray:
    """A detailed synthetic RGB image (sharp edges everywhere)."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(60):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    noise = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    return cv2.add(img, noise)


def save_jpeg(
    path: Path,
    rgb: np.ndarray,
    when: datetime | None = None,
    orientation: int | None = None,
    subsec: str | None = None,
    quality: int = 92,
) -> Path:
    """Write ``rgb`` as JPEG with optional EXIF capture time and orientation."""
    exif = Image.Exif()
    exif[0x010F] = "Test"
    exif[0x0110] = "Camera"
    if orientation is not None:
        exif[0x0112] = orientation
    if when is not None:
        stamp = when.strftime("%Y:%m:%d %H:%M:%S")
        exif[0x0132] = stamp
        sub = exif.get_ifd(0x8769)
        sub[0x9003] = stamp
        if subsec is not None:
            sub[0x9291] = subsec
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="JPEG", quality=quality, exif=exif)
    return path


def jpeg_bytes(rgb: np.ndarray, quality: int = 90) -> bytes:
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def crop_shift(rgb: np.ndarray, dx: float = 0.0, dy: float = 0.0, zoom: float = 1.0) -> np.ndarray:
    h, w = rgb.shape[:2]
    cw, ch = int(w * 0.9 / zoom), int(h * 0.9 / zoom)
    x = max(0, min(w - cw, int((w - cw) / 2 + dx * w)))
    y = max(0, min(h - ch, int((h - ch) / 2 + dy * h)))
    return cv2.resize(
        rgb[y : y + ch, x : x + cw], (int(w * 0.9), int(h * 0.9)), interpolation=cv2.INTER_CUBIC
    )

#!/usr/bin/env python3
"""Build a small demo "shoot" for screenshots, docs and end-to-end checks.

It downloads public-domain / CC0 photos from the scikit-image repository and
turns them into bursts the way a real session looks: slightly shifted
frames, camera shake, missed focus, an exposure bracket and a blink.

=====================  ====================================================
astronaut.png          Eileen Collins, NASA. Public domain.
chelsea.png            Chelsea the cat, Stefan van der Walt. CC0.
coffee.png             Coffee cup, Rachel Michetti. CC0.
rocket.jpg             DSCOVR launch, SpaceX. Public domain.
hubble_deep_field.jpg  Hubble eXtreme Deep Field, NASA. Public domain.
=====================  ====================================================

Usage::

    python tools/make_demo.py demo-shoot
    pickframe cull demo-shoot --dry-run --list
"""

from __future__ import annotations

import argparse
import hashlib
import math
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.22.0/skimage/data/"
SOURCES = {
    "astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    "chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "rocket.jpg": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
    "hubble_deep_field.jpg": "3a19c5dd8a927a9334bb1229a6d63711b1c0c767fb27e2286e7c84a3e2c2f5f4",
}

# Upper and lower eyelid landmarks (corner to corner) of the MediaPipe face mesh.
_LIDS = (
    ([33, 246, 161, 160, 159, 158, 157, 173, 133], [33, 7, 163, 144, 145, 153, 154, 155, 133]),
    ([362, 398, 384, 385, 386, 387, 388, 466, 263], [362, 382, 381, 380, 374, 373, 390, 249, 263]),
)


def fetch(name: str, cache: Path) -> np.ndarray:
    path = cache / name
    if not path.is_file():
        with urllib.request.urlopen(BASE + name, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCES[name]:
            raise SystemExit(f"checksum mismatch for {name}")
        path.write_bytes(data)
    return np.asarray(Image.open(path).convert("RGB"))


def crop(img: np.ndarray, dx: float, dy: float, zoom: float = 1.0) -> np.ndarray:
    """Recompose: shift by a fraction of the frame and zoom in slightly."""
    h, w = img.shape[:2]
    cw, ch = int(w * 0.9 / zoom), int(h * 0.9 / zoom)
    x = int((w - cw) / 2 + dx * w)
    y = int((h - ch) / 2 + dy * h)
    x, y = max(0, min(w - cw, x)), max(0, min(h - ch, y))
    return cv2.resize(img[y : y + ch, x : x + cw], (int(w * 0.9), int(h * 0.9)), interpolation=cv2.INTER_CUBIC)


def shake(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def exposure(img: np.ndarray, stops: float) -> np.ndarray:
    linear = (img.astype(np.float32) / 255.0) ** 2.2 * (2.0**stops)
    return (np.clip(linear, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def close_eyes(img: np.ndarray) -> np.ndarray:
    """Stretch the upper eyelids down to the lower lids (a convincing blink)."""
    from pickframe.analysis.faces import FaceAnalyzer

    faces = FaceAnalyzer().landmarks(img)
    if not faces:
        raise SystemExit("no face found for the blink frame")
    pts = faces[0]
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for upper_idx, lower_idx in _LIDS:
        upper, lower = pts[upper_idx], pts[lower_idx]
        x0 = int(math.floor(min(upper[:, 0].min(), lower[:, 0].min())))
        x1 = int(math.ceil(max(upper[:, 0].max(), lower[:, 0].max())))
        eye_width = x1 - x0
        ou, ol = np.argsort(upper[:, 0]), np.argsort(lower[:, 0])
        for x in range(max(0, x0), min(w, x1 + 1)):
            top = float(np.interp(x, upper[ou, 0], upper[ou, 1]))
            bottom = float(np.interp(x, lower[ol, 0], lower[ol, 1]))
            if bottom <= top + 0.5:
                continue
            brow = top - 0.45 * eye_width
            for y in range(max(0, int(brow)), min(h, int(math.ceil(bottom)) + 1)):
                if y >= brow:
                    map_y[y, x] = brow + (y - brow) * ((top - brow) / (bottom - brow))
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def save(img: np.ndarray, path: Path, when: datetime) -> None:
    exif = Image.Exif()
    exif[0x010F] = "Pickframe"
    exif[0x0110] = "Demo Camera"
    exif[0x0132] = when.strftime("%Y:%m:%d %H:%M:%S")
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
    sub[0x9291] = f"{when.microsecond // 10000:02d}"
    Image.fromarray(img).save(path, quality=92, exif=exif)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="output folder")
    parser.add_argument("--cache", type=Path, default=None, help="where to keep the downloaded originals")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or args.out / ".sources"
    cache.mkdir(parents=True, exist_ok=True)

    astronaut = fetch("astronaut.png", cache)
    cat = fetch("chelsea.png", cache)
    coffee = fetch("coffee.png", cache)
    rocket = fetch("rocket.jpg", cache)
    hubble = fetch("hubble_deep_field.jpg", cache)

    t = datetime(2026, 6, 20, 14, 2, 10)
    frames: list[tuple[str, np.ndarray, datetime]] = []

    def burst(items: list[np.ndarray], start: datetime, step: float = 0.35) -> None:
        for offset, image in enumerate(items):
            frames.append(("", image, start + timedelta(seconds=step * offset)))

    a = [crop(astronaut, dx, dy, z) for dx, dy, z in ((0.0, 0.0, 1.0), (0.01, 0.0, 1.03), (0.0, 0.01, 1.0),
                                                       (-0.01, 0.0, 1.02), (0.012, -0.005, 1.0))]  # fmt: skip
    burst([a[1], a[0], close_eyes(a[2]), shake(a[3], 15, 8), cv2.GaussianBlur(a[4], (0, 0), 1.6)], t)
    c = [crop(cat, dx, 0.0, z) for dx, z in ((0.0, 1.0), (0.02, 1.04), (-0.015, 1.0))]
    burst([c[0], c[1], cv2.GaussianBlur(c[2], (0, 0), 3.5)], t + timedelta(minutes=3))
    k = crop(coffee, 0.0, 0.0)
    burst([exposure(k, -1.7), k, exposure(k, 1.7)], t + timedelta(minutes=6), step=0.25)
    r = [crop(rocket, dx, 0.0, z) for dx, z in ((0.0, 1.0), (0.01, 1.05))]
    burst(r, t + timedelta(minutes=9), step=0.5)
    frames.append(("", crop(hubble, 0.0, 0.0), t + timedelta(minutes=14)))

    for index, (_, image, when) in enumerate(frames, start=1):
        save(image, args.out / f"DSC_{index:04d}.jpg", when)
    print(f"wrote {len(frames)} photos to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

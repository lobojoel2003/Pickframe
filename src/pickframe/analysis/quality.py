"""Sharpness and exposure measurements.

Sharpness is the variance of the Laplacian, measured per tile on a
fixed-size grayscale copy. Only the sharpest tiles count, so a portrait with
a sharp subject and a creamy background still scores as sharp, while camera
shake or missed focus (no sharp tile anywhere) scores low.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

#: Working size for the sharpness measurement (long side, pixels).
SHARPNESS_SIZE = 1024
_TILE = 64
_TOP_FRACTION = 0.1

# log10(variance) range mapped to the 0-1 sharpness score. Calibrated on
# sharp photos vs. the same photos with Gaussian / motion blur applied.
_SHARP_LOG_LOW = -3.0
_SHARP_LOG_HIGH = -1.6


@dataclass
class Exposure:
    score: float  # 0 (unusable) .. 1 (well exposed)
    brightness: float  # mean luminance, 0..1
    highlights: float  # fraction of clipped highlight pixels
    shadows: float  # fraction of crushed shadow pixels

    @property
    def overexposed(self) -> bool:
        return self.highlights > 0.15 or self.brightness > 0.85

    @property
    def underexposed(self) -> bool:
        return self.brightness < 0.1 or self.shadows > 0.6


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb


def laplacian_energy(gray: np.ndarray, tile: int = _TILE, top: float = _TOP_FRACTION) -> float:
    """Mean Laplacian variance of the sharpest ``top`` fraction of tiles.

    ``gray`` is uint8 or float in 0..255; the result uses 0..1 intensities.
    """
    g = gray.astype(np.float32) * (1.0 / 255.0)
    g = cv2.GaussianBlur(g, (0, 0), 0.7)  # damp sensor noise and JPEG blocking
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    h, w = lap.shape
    tile = max(8, min(tile, h, w))
    rows, cols = h // tile, w // tile
    if rows == 0 or cols == 0:
        return float(lap.var())
    blocks = lap[: rows * tile, : cols * tile].reshape(rows, tile, cols, tile)
    variances = blocks.var(axis=(1, 3)).ravel()
    count = max(1, math.ceil(variances.size * top))
    return float(np.partition(variances, variances.size - count)[-count:].mean())


def sharpness_raw(rgb: np.ndarray) -> float:
    """Resolution-independent focus measure of a whole frame."""
    gray = to_gray(rgb)
    h, w = gray.shape
    scale = SHARPNESS_SIZE / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return laplacian_energy(gray)


def sharpness_score(raw: float) -> float:
    """Map a Laplacian energy to 0 (blurred) .. 1 (crisp)."""
    if raw <= 0:
        return 0.0
    value = (math.log10(raw) - _SHARP_LOG_LOW) / (_SHARP_LOG_HIGH - _SHARP_LOG_LOW)
    return float(min(1.0, max(0.0, value)))


def contrast(rgb: np.ndarray) -> float:
    """Standard deviation of the luminance (0..1); used to make focus comparisons
    between frames of different brightness fair (detail energy scales with contrast²)."""
    gray = to_gray(rgb)
    h, w = gray.shape
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return float(gray.astype(np.float32).std() / 255.0)


def exposure(rgb: np.ndarray) -> Exposure:
    small = rgb
    h, w = rgb.shape[:2]
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    brightness = float(lum.mean() / 255.0)
    highlights = float(np.mean(small.max(axis=2) >= 250))
    shadows = float(np.mean(lum <= 8))
    penalty = 0.0
    penalty += min(1.0, max(0.0, highlights - 0.02) / 0.2) * 0.6
    penalty += min(1.0, max(0.0, shadows - 0.05) / 0.5) * 0.5
    penalty += min(1.0, max(0.0, abs(brightness - 0.47) - 0.23) / 0.2) * 0.6
    return Exposure(
        score=float(max(0.0, 1.0 - penalty)),
        brightness=brightness,
        highlights=highlights,
        shadows=shadows,
    )

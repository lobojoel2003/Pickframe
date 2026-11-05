from __future__ import annotations

import cv2
import numpy as np
import pytest

from pickframe.analysis import quality


def test_sharpness_drops_with_blur(astronaut_rgb):
    values = [quality.sharpness_raw(astronaut_rgb)]
    for sigma in (0.8, 1.5, 3.0, 6.0):
        values.append(quality.sharpness_raw(cv2.GaussianBlur(astronaut_rgb, (0, 0), sigma)))
    assert values == sorted(values, reverse=True)
    assert quality.sharpness_score(values[0]) > 0.9
    assert quality.sharpness_score(values[-1]) < 0.2


def test_sharpness_is_resolution_independent(astronaut_rgb):
    big = cv2.resize(astronaut_rgb, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)
    # Upscaling softens fine detail but the measurement still runs at 1024 px.
    assert quality.sharpness_raw(big) > 0
    assert quality.sharpness_raw(big) < quality.sharpness_raw(astronaut_rgb)


def test_sharpness_score_bounds():
    assert quality.sharpness_score(0) == 0.0
    assert quality.sharpness_score(-1) == 0.0
    assert quality.sharpness_score(1e-9) == 0.0
    assert quality.sharpness_score(10.0) == 1.0
    assert 0.0 < quality.sharpness_score(10**-2.3) < 1.0


def test_laplacian_energy_small_and_flat_inputs():
    flat = np.full((50, 50), 128, np.uint8)
    assert quality.laplacian_energy(flat) == pytest.approx(0.0, abs=1e-9)
    tiny = np.random.default_rng(0).integers(0, 255, (5, 5)).astype(np.uint8)
    assert quality.laplacian_energy(tiny) >= 0.0


def test_exposure_flags():
    black = np.zeros((300, 400, 3), np.uint8)
    white = np.full((300, 400, 3), 255, np.uint8)
    mid = np.full((300, 400, 3), 120, np.uint8)
    dark = quality.exposure(black)
    bright = quality.exposure(white)
    good = quality.exposure(mid)
    assert dark.underexposed
    assert not dark.overexposed
    assert bright.overexposed
    assert not bright.underexposed
    assert not good.overexposed
    assert not good.underexposed
    assert good.score > 0.9 > dark.score
    assert bright.score < 0.5
    assert bright.highlights == pytest.approx(1.0)
    assert dark.shadows == pytest.approx(1.0)


def test_exposure_downscales_large_images():
    big = np.full((2000, 3000, 3), 100, np.uint8)
    assert quality.exposure(big).brightness == pytest.approx(100 / 255, abs=0.01)


def test_contrast(astronaut_rgb):
    assert quality.contrast(np.full((20, 20, 3), 77, np.uint8)) == pytest.approx(0.0)
    full = quality.contrast(astronaut_rgb)
    halved = quality.contrast((astronaut_rgb // 2).astype(np.uint8))
    assert full > 0.1
    assert halved == pytest.approx(full / 2, rel=0.05)
    big = cv2.resize(astronaut_rgb, (1024, 1024))
    assert quality.contrast(big) == pytest.approx(full, rel=0.1)
    assert quality.to_gray(astronaut_rgb[..., 0]).ndim == 2

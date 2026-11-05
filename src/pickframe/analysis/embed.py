"""Image embeddings used to group near-duplicate shots.

Two backends share one interface (L2-normalised vectors compared with the
cosine similarity):

``clip``
    The CLIP ViT-B/32 image encoder. Understands content ("same couple, same
    pose") and is robust to recomposition and zoom. Optional ~88 MB download.

``classic``
    Built in, no download, very fast: a blend of a blurred thumbnail, low
    frequency DCT coefficients (a continuous perceptual hash) and a colour
    histogram. Excellent for bursts and bracketing; less forgiving when the
    photographer reframes between shots.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .. import models


class Embedder(Protocol):
    name: str
    #: Cosine similarity at which two consecutive shots count as the same scene.
    default_similarity: float

    def embed(self, rgb: np.ndarray) -> np.ndarray: ...


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


class ClassicEmbedder:
    name = "classic"
    default_similarity = 0.80

    # Weights of the three blocks; cosine of the concatenation is the weighted
    # mean of the per-block cosines because every block is unit length.
    _W_THUMB, _W_DCT, _W_COLOR = 0.45, 0.30, 0.25

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

        thumb = cv2.GaussianBlur(lab, (0, 0), 2.0)
        thumb = cv2.resize(thumb, (12, 12), interpolation=cv2.INTER_AREA)
        thumb[..., 0] -= thumb[..., 0].mean()  # brightness-invariant structure
        thumb[..., 1:] -= 128.0
        thumb_vec = _unit(thumb.ravel() * np.repeat([[1.0, 0.5, 0.5]], 144, axis=0).ravel())

        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
        coeffs = cv2.dct(gray.astype(np.float32))[:8, :8].ravel()[1:]
        dct_vec = _unit(coeffs)

        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        color_vec = _unit(np.sqrt(hist))

        return np.concatenate(
            [
                thumb_vec * np.sqrt(self._W_THUMB),
                dct_vec * np.sqrt(self._W_DCT),
                color_vec * np.sqrt(self._W_COLOR),
            ]
        ).astype(np.float32)


class ClipEmbedder:
    name = "clip"
    default_similarity = 0.88

    _MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    _STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    _SIZE = 224

    def __init__(self, threads: int = 0):
        path = models.remote_path("clip")
        if not path.is_file():
            raise models.ModelError("CLIP model not installed; run 'pickframe models download clip'")
        self._session = models.session(path, threads)
        self._input = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        names = [o.name for o in outputs]
        if "image_embeds" in names:
            self._output = "image_embeds"
        else:
            two_d = [o.name for o in outputs if len(o.shape) == 2]
            self._output = two_d[0] if two_d else names[0]

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        scale = self._SIZE / min(h, w)
        resized = cv2.resize(
            rgb, (max(self._SIZE, round(w * scale)), max(self._SIZE, round(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        rh, rw = resized.shape[:2]
        top, left = (rh - self._SIZE) // 2, (rw - self._SIZE) // 2
        crop = resized[top : top + self._SIZE, left : left + self._SIZE].astype(np.float32) / 255.0
        crop = (crop - self._MEAN) / self._STD
        return crop.transpose(2, 0, 1)[None].astype(np.float32)

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        out = self._session.run([self._output], {self._input: self.preprocess(rgb)})[0]
        vec = np.asarray(out, dtype=np.float32).reshape(out.shape[0], -1)[0]
        return _unit(vec)


def create_embedder(kind: str = "auto", threads: int = 0) -> Embedder:
    """``auto`` uses CLIP when it has been downloaded, otherwise ``classic``."""
    if kind == "classic":
        return ClassicEmbedder()
    if kind == "clip":
        return ClipEmbedder(threads)
    if kind == "auto":
        if models.is_installed("clip"):
            try:
                return ClipEmbedder(threads)
            except Exception:
                return ClassicEmbedder()
        return ClassicEmbedder()
    raise ValueError(f"unknown embedder {kind!r} (expected auto, clip or classic)")

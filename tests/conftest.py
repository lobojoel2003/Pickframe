"""Shared fixtures and helpers for the Pickframe test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Qt must not try to open a display, wherever the tests run.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
TOOLS = ROOT / "tools"


@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the analysis cache and downloaded models out of the user's folders."""
    base = tmp_path_factory.mktemp("pickframe-home")
    monkeypatch.setenv("PICKFRAME_CACHE_DIR", str(base / "cache"))
    monkeypatch.setenv("PICKFRAME_MODELS_DIR", str(base / "models"))
    monkeypatch.delenv("PICKFRAME_CLIP_URL", raising=False)
    monkeypatch.delenv("PICKFRAME_ORT_PROVIDERS", raising=False)


@pytest.fixture(scope="session")
def astronaut_path() -> Path:
    return DATA / "astronaut.jpg"


@pytest.fixture(scope="session")
def chelsea_path() -> Path:
    return DATA / "chelsea.jpg"


@pytest.fixture(scope="session")
def astronaut_rgb(astronaut_path: Path) -> np.ndarray:
    return np.asarray(Image.open(astronaut_path).convert("RGB"))


@pytest.fixture(scope="session")
def chelsea_rgb(chelsea_path: Path) -> np.ndarray:
    return np.asarray(Image.open(chelsea_path).convert("RGB"))


@pytest.fixture(scope="session")
def face_analyzer():
    from pickframe.analysis.faces import FaceAnalyzer

    return FaceAnalyzer()


@pytest.fixture(scope="session")
def make_demo():
    """The demo generator from tools/, which knows how to synthesise a blink."""
    sys.path.insert(0, str(TOOLS))
    try:
        import make_demo as module
    finally:
        sys.path.remove(str(TOOLS))
    return module

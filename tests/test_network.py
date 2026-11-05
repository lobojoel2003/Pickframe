"""Tests that need the internet (CLIP download, demo photos).

Run with ``PICKFRAME_NETWORK_TESTS=1 pytest -m network -v``; CI does this on
every push to main.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("PICKFRAME_NETWORK_TESTS") != "1",
        reason="set PICKFRAME_NETWORK_TESTS=1 to run",
    ),
]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def clip_model(tmp_path_factory):
    from pickframe import models

    folder = tmp_path_factory.mktemp("models")
    previous = os.environ.get("PICKFRAME_MODELS_DIR")
    os.environ["PICKFRAME_MODELS_DIR"] = str(folder)
    try:
        path = models.download("clip")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"\nCLIP model {path.stat().st_size} bytes, sha256 {digest}")
        if models.CLIP.sha256:
            assert digest == models.CLIP.sha256
        yield path
    finally:
        if previous is None:
            os.environ.pop("PICKFRAME_MODELS_DIR", None)
        else:
            os.environ["PICKFRAME_MODELS_DIR"] = previous


def test_clip_embeddings_separate_scenes(clip_model, monkeypatch):
    monkeypatch.setenv("PICKFRAME_MODELS_DIR", str(clip_model.parent))
    from pickframe.analysis.embed import ClipEmbedder
    from pickframe.imaging import load_image

    embedder = ClipEmbedder()
    astronaut = load_image(DATA / "astronaut.jpg").rgb
    cat = load_image(DATA / "chelsea.jpg").rgb
    a = embedder.embed(astronaut)
    shifted = embedder.embed(astronaut[10:, 12:])
    c = embedder.embed(cat)
    assert a.shape == (512,)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4
    same, different = float(a @ shifted), float(a @ c)
    print(f"\nCLIP similarity: shifted frame {same:.3f}, different scene {different:.3f}")
    assert same >= ClipEmbedder.default_similarity
    assert different < ClipEmbedder.default_similarity


def test_demo_shoot_with_clip(clip_model, monkeypatch, tmp_path):
    monkeypatch.setenv("PICKFRAME_MODELS_DIR", str(clip_model.parent))
    from pickframe.analysis.embed import ClipEmbedder
    from pickframe.config import Settings
    from pickframe.pipeline import Analyzer

    make_demo = _load_tool("make_demo")
    shoot = tmp_path / "shoot"
    assert make_demo.main([str(shoot)]) == 0

    result = Analyzer(Settings(embedder="clip"), use_cache=False).run([shoot])
    assert result.embedder == "clip"
    for shot in result.shots:
        d = result.decisions[shot.id]
        print(
            f"{shot.name}: group {d.group + 1} {d.status} rating={result.rating(shot.id)} {d.flags}"
        )

    # Similarities between consecutive frames, to calibrate the default threshold.
    embeddings = [result.analyses[s.id].embedding for s in result.shots]
    pairs = [float(x @ y) for x, y in itertools.pairwise(embeddings)]
    print("consecutive similarities:", " ".join(f"{p:.3f}" for p in pairs))
    print("default threshold:", ClipEmbedder.default_similarity)

    assert len(result.groups) == 5
    names = {s.name: result.decisions[s.id] for s in result.shots}
    assert names["DSC_0003.jpg"].status == "reject"  # blink
    assert names["DSC_0008.jpg"].status == "reject"  # missed focus
    assert sum(d.status == "pick" for d in result.decisions.values()) == 5

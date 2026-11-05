from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from helpers import crop_shift, save_jpeg
from pickframe.config import Settings
from pickframe.pipeline import Analyzer, Cancelled, CullResult, default_workers
from pickframe.results import BLURRY, PICK, REJECT, UNREADABLE

T0 = datetime(2026, 6, 20, 10, 0, 0)


def make_shoot(folder: Path, astronaut: np.ndarray, cat: np.ndarray) -> Path:
    save_jpeg(folder / "DSC_0001.jpg", crop_shift(astronaut), T0)
    save_jpeg(
        folder / "DSC_0002.jpg", crop_shift(astronaut, 0.01, 0.0, 1.02), T0 + timedelta(seconds=0.5)
    )
    blurred = cv2.GaussianBlur(crop_shift(astronaut, -0.01), (0, 0), 3.0)
    save_jpeg(folder / "DSC_0003.jpg", blurred, T0 + timedelta(seconds=1))
    save_jpeg(folder / "DSC_0004.jpg", cat, T0 + timedelta(minutes=2))
    return folder


@pytest.fixture
def shoot(tmp_path: Path, astronaut_rgb, chelsea_rgb) -> Path:
    return make_shoot(tmp_path / "shoot", astronaut_rgb, chelsea_rgb)


def test_run_groups_and_decides(shoot: Path):
    calls: list[tuple[int, int, str]] = []
    analyzer = Analyzer(Settings(workers=2))
    result = analyzer.run([shoot], progress=lambda d, t, n: calls.append((d, t, n)))
    analyzer.close()

    assert [s.name for s in result.shots] == [
        "DSC_0001.jpg",
        "DSC_0002.jpg",
        "DSC_0003.jpg",
        "DSC_0004.jpg",
    ]
    assert result.groups == [[0, 1, 2], [3]]
    d = result.decisions
    assert sum(d[i].status == PICK for i in (0, 1, 2)) == 1
    assert d[2].status == REJECT
    assert BLURRY in d[2].flags
    assert d[3].status == PICK
    assert result.embedder == "classic"
    assert result.cached == 0
    assert result.elapsed > 0
    assert result.counts() == {"shots": 4, "groups": 2, "pick": 2, "alternate": 1, "reject": 1}
    assert calls[-1][:2] == (4, 4)
    a = result.analyses[0]
    assert a.width == 460
    assert a.height == 460
    assert a.capture_time is not None
    assert a.camera == "Test Camera"
    assert len(a.faces) == 1
    assert a.embedding is not None
    assert a.embedder == "classic"


def test_cache_is_used_and_invalidated(shoot: Path):
    first = Analyzer().run([shoot])
    second = Analyzer().run([shoot])
    assert second.cached == 4
    assert [second.decisions[i].status for i in range(4)] == [
        first.decisions[i].status for i in range(4)
    ]
    assert second.analyses[0].faces[0].blink == pytest.approx(
        first.analyses[0].faces[0].blink, abs=1e-3
    )

    # touching a file invalidates its cache entry
    target = shoot / "DSC_0004.jpg"
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    third = Analyzer().run([shoot])
    assert third.cached == 3

    no_cache = Analyzer(use_cache=False).run([shoot])
    assert no_cache.cached == 0


def test_unreadable_file(tmp_path: Path, astronaut_rgb):
    save_jpeg(tmp_path / "good.jpg", astronaut_rgb)
    (tmp_path / "bad.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    result = Analyzer().run([tmp_path])
    by_name = {s.name: s.id for s in result.shots}
    bad = result.decisions[by_name["bad.jpg"]]
    assert bad.status == REJECT
    assert UNREADABLE in bad.flags
    assert result.analyses[by_name["bad.jpg"]].error
    assert result.decisions[by_name["good.jpg"]].status == PICK
    # errors are not cached
    assert Analyzer().run([tmp_path]).cached == 1


def test_save_load_roundtrip(shoot: Path, tmp_path: Path):
    result = Analyzer().run([shoot])
    result.overrides = {0: 3, 1: None, 3: -1}
    path = tmp_path / "session.json"
    result.save(path)
    loaded = CullResult.load(path)
    assert [s.files for s in loaded.shots] == [s.files for s in result.shots]
    assert loaded.groups == result.groups
    assert loaded.overrides == {0: 3, 1: None, 3: -1}
    assert loaded.rating(0) == 3
    assert loaded.rating(1) is None
    assert loaded.rating(2) == result.decisions[2].rating
    assert loaded.embedder == "classic"
    for i in range(4):
        assert loaded.decisions[i].status == result.decisions[i].status
        assert np.allclose(loaded.analyses[i].embedding, result.analyses[i].embedding)
    plain = result.to_dict(with_embeddings=False)
    assert "embedding" not in plain["shots"][0]["analysis"]


def test_regroup(shoot: Path):
    result = Analyzer().run([shoot])
    result.regroup(Settings(similarity=0.9999))
    assert len(result.groups) == 4
    result.regroup(Settings(similarity=0.0, max_gap=10_000))
    assert len(result.groups) == 1
    assert result.counts()["pick"] == 1
    result.regroup(Settings(picks_per_group=3, similarity=0.0, max_gap=10_000))
    assert result.counts()["pick"] == 3


def test_cancel(shoot: Path):
    import threading

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        Analyzer(use_cache=False).run([shoot], cancel=cancel)


def test_empty_run(tmp_path: Path):
    result = Analyzer().run([tmp_path])
    assert result.shots == []
    assert result.groups == []
    assert result.embedder == ""
    assert result.counts()["shots"] == 0


def test_analyze_image_failure_paths(tmp_path: Path, monkeypatch, astronaut_rgb):
    analyzer = Analyzer()
    path = save_jpeg(tmp_path / "x.jpg", astronaut_rgb)

    def boom(*args, **kwargs):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr("pickframe.pipeline.load_image", boom)
    assert "decoder exploded" in (analyzer.analyze_image(path).error or "")
    monkeypatch.undo()

    class BrokenFaces:
        def analyze(self, rgb):
            raise RuntimeError("model failed")

    analyzer._faces = BrokenFaces()  # type: ignore[assignment]
    assert "face analysis failed" in (analyzer.analyze_image(path).error or "")


def test_cache_unavailable(monkeypatch, shoot: Path):
    def broken(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr("pickframe.pipeline.AnalysisCache", broken)
    analyzer = Analyzer()
    assert analyzer.cache is None
    assert analyzer.run([shoot]).cached == 0


def test_default_workers():
    assert 1 <= default_workers() <= 8
    assert Analyzer(Settings(workers=3)).workers == 3

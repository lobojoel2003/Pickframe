from __future__ import annotations

import runpy
import sqlite3
import sys
from pathlib import Path

import pytest

import pickframe
from pickframe import _build, version
from pickframe.cache import AnalysisCache, default_path
from pickframe.config import RejectStyle, Settings, SidecarStyle
from pickframe.results import Analysis


def test_version_string_dev(monkeypatch):
    monkeypatch.setattr(_build, "BUILD_NUMBER", "0")
    monkeypatch.setattr(_build, "COMMIT", "")
    monkeypatch.setattr(_build, "BUILD_DATE", "")
    monkeypatch.setattr(_build, "CHANNEL", "dev")
    assert version.version_string() == f"Pickframe {pickframe.__version__} (dev)"
    assert version.build_number() == 0


def test_version_string_release(monkeypatch):
    monkeypatch.setattr(_build, "BUILD_NUMBER", "42")
    monkeypatch.setattr(_build, "COMMIT", "1a2b3c4d5e6f")
    monkeypatch.setattr(_build, "BUILD_DATE", "2026-11-05")
    text = version.version_string(verbose=True)
    assert text.startswith(f"Pickframe {pickframe.__version__} (build 42, 1a2b3c4, 2026-11-05)")
    assert "Python" in text
    assert "Frozen" in text
    assert version.build_number() == 42
    monkeypatch.setattr(_build, "BUILD_NUMBER", "abc")
    assert version.build_number() == 0


def test_version_is_semver():
    parts = pickframe.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pickframe", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("pickframe", run_name="__main__")
    assert exit_info.value.code == 0
    assert "Pickframe" in capsys.readouterr().out


def test_settings_roundtrip(tmp_path: Path):
    s = Settings(
        similarity=0.7,
        reject_style=RejectStyle.ONE_STAR,
        sidecar_style=SidecarStyle.LIGHTROOM,
        pick_label="Red",
    )
    data = s.to_dict()
    assert data["reject_style"] == "one-star"
    assert data["sidecar_style"] == "lightroom"
    again = Settings.from_dict({**data, "unknown_key": 1})
    assert again == s
    path = tmp_path / "conf" / "settings.json"
    s.save(path)
    assert Settings.load(path) == s
    path.write_text("{not json")
    assert Settings.load(path) == Settings()
    assert Settings.load(tmp_path / "missing.json") == Settings()
    assert Settings().analysis_key() != Settings(analysis_size=800).analysis_key()


def test_cache_roundtrip(tmp_path: Path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"data")
    with AnalysisCache(tmp_path / "c.sqlite3") as cache:
        assert cache.get(image, "k") is None
        cache.put(image, "k", Analysis(path=str(image), sharpness=0.5))
        hit = cache.get(image, "k")
        assert hit is not None
        assert hit.sharpness == 0.5
        assert cache.get(image, "other-key") is None
        cache.put(image, "k2", Analysis(path=str(image), error="bad"))  # errors are not stored
        assert cache.get(image, "k2") is None
        cache.put(tmp_path / "missing.jpg", "k", Analysis(path="x"))
        assert cache.get(tmp_path / "missing.jpg", "k") is None
        image.write_bytes(b"changed data")
        assert cache.get(image, "k") is None
        assert cache.clear() == 1


def test_cache_tolerates_corrupt_rows(tmp_path: Path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"data")
    db = tmp_path / "c.sqlite3"
    with AnalysisCache(db) as cache:
        cache.put(image, "k", Analysis(path=str(image)))
    con = sqlite3.connect(db)
    con.execute("UPDATE analyses SET data = '{broken'")
    con.commit()
    con.close()
    with AnalysisCache(db) as cache:
        assert cache.get(image, "k") is None


def test_default_cache_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PICKFRAME_CACHE_DIR", str(tmp_path))
    assert default_path().parent == tmp_path
    monkeypatch.delenv("PICKFRAME_CACHE_DIR")
    assert default_path().name.endswith(".sqlite3")


def test_analysis_dict_roundtrip():
    import numpy as np

    a = Analysis(
        path="p.jpg",
        width=10,
        height=20,
        capture_time=5.5,
        embedding=np.arange(4, dtype=np.float32),
    )
    b = Analysis.from_dict(a.to_dict())
    assert b.width == 10
    assert b.capture_time == 5.5
    assert b.embedding is not None
    assert b.embedding.tolist() == [0, 1, 2, 3]
    assert Analysis.from_dict(a.to_dict(with_embedding=False)).embedding is None

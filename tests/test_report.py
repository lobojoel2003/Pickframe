from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pickframe.pipeline import Analyzer, CullResult
from pickframe.report import render_html, write_report
from test_pipeline import make_shoot


@pytest.fixture
def result(tmp_path: Path, astronaut_rgb, chelsea_rgb) -> CullResult:
    return Analyzer().run([make_shoot(tmp_path / "shoot", astronaut_rgb, chelsea_rgb)])


def test_html(result: CullResult, tmp_path: Path):
    calls = []
    html = render_html(result, thumb_size=64, progress=lambda d, t, n: calls.append(d))
    assert html.startswith("<!DOCTYPE html>")
    assert html.count('class="card ') == 4
    assert html.count("data:image/jpeg;base64,") == 4
    assert 'class="face' in html  # face boxes drawn on the astronaut
    assert "<b>2</b> groups" in html
    assert "blurry" in html
    assert calls == [1, 2, 3, 4]
    path = tmp_path / "r.html"
    write_report(result, path)
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_html_with_missing_file(result: CullResult):
    result.shots[3].files[0].unlink()
    html = render_html(result, thumb_size=32)
    assert html.count("data:image/jpeg;base64,") == 3


def test_csv(result: CullResult, tmp_path: Path):
    path = tmp_path / "r.csv"
    write_report(result, path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 4
    assert {r["status"] for r in rows} == {"pick", "alternate", "reject"}
    assert rows[0]["group"] == "1"
    assert rows[0]["capture_time"].startswith("2026-06-20 10:00")
    assert any("blurry" in r["flags"] for r in rows)


def test_json(result: CullResult, tmp_path: Path):
    path = tmp_path / "r.json"
    write_report(result, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["shots"]) == 4
    assert data["groups"] == [[0, 1, 2], [3]]
    assert "embedding" not in data["shots"][0]["analysis"]
    assert data["shots"][0]["decision"]["status"] in ("pick", "alternate")


def test_bad_extension(result: CullResult, tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported report format"):
        write_report(result, tmp_path / "r.pdf")

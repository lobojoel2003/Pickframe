from __future__ import annotations

from pathlib import Path

import pytest

from helpers import save_jpeg, texture
from pickframe import xmp
from pickframe.config import RejectStyle, Settings, SidecarStyle
from pickframe.pipeline import Analyzer, CullResult
from pickframe.writer import plan, reset, write_result
from test_pipeline import make_shoot


@pytest.fixture
def result(tmp_path: Path, astronaut_rgb, chelsea_rgb) -> CullResult:
    folder = make_shoot(tmp_path / "shoot", astronaut_rgb, chelsea_rgb)
    (folder / "DSC_0004.NEF").write_bytes(b"raw data")  # RAW+JPEG pair
    return Analyzer().run([folder])


def _flags(result: CullResult) -> dict[str, str]:
    return {s.name: plan(result)[s.id].flag for s in result.shots}


def test_plan_flags(result: CullResult):
    updates = plan(result)
    for shot in result.shots:
        decision = result.decisions[shot.id]
        update = updates[shot.id]
        assert update.rating == decision.rating
        assert update.group == decision.group + 1
        expected = {"pick": "pick", "reject": "reject", "alternate": "none"}[decision.status]
        assert update.flag == expected
    result.settings = Settings(reject_style=RejectStyle.NONE)
    assert "reject" not in _flags(result).values()
    result.settings = Settings(write_flags=False)
    assert set(_flags(result).values()) == {"none"}
    result.settings = Settings(pick_label="Green")
    labels = {u.label for u in plan(result).values()}
    assert labels == {"Green", ""}


def test_plan_overrides(result: CullResult):
    pick = next(s.id for s in result.shots if result.decisions[s.id].status == "pick")
    result.overrides[pick] = -1
    assert plan(result)[pick].flag == "reject"
    result.overrides[pick] = 4
    assert plan(result)[pick].flag == "pick"
    assert plan(result)[pick].rating == 4
    result.overrides[pick] = None
    assert plan(result)[pick].flag == "none"
    assert plan(result)[pick].rating is None


def test_write_and_reset(result: CullResult):
    folder = result.shots[0].primary.parent
    calls = []
    report = write_result(result, progress=lambda d, t, n: calls.append(d))
    assert not report.errors
    names = sorted(p.name for p in report.written)
    assert "DSC_0004.NEF.xmp" in names
    assert "DSC_0004.xmp" in names
    assert "DSC_0004.jpg.xmp" in names
    assert len(names) == 6
    assert calls[-1] == len(result.shots)
    for shot in result.shots:
        assert xmp.read_rating(shot.primary) == result.decisions[shot.id].rating

    again = write_result(result)
    assert again.written == []
    assert len(again.unchanged) == 6

    removed = reset([folder])
    assert len(removed.removed) == 6
    assert not removed.errors
    assert not list(folder.glob("*.xmp"))


def test_dry_run_and_embed(result: CullResult):
    folder = result.shots[0].primary.parent
    report = write_result(result, dry_run=True, embed_jpeg=True)
    assert len(report.written) == 10  # 6 sidecars + 4 JPEGs
    assert not list(folder.glob("*.xmp"))
    assert xmp.read_jpeg_xmp(folder / "DSC_0001.jpg") is None

    write_result(result, embed_jpeg=True)
    assert xmp.read_jpeg_xmp(folder / "DSC_0001.jpg") is not None
    out = reset([folder], dry_run=True)
    assert len(out.written) == 4
    assert len(out.removed) == 6
    assert list(folder.glob("*.xmp"))
    reset([folder], include_jpeg=False)
    assert "pickframe" in (xmp.read_jpeg_xmp(folder / "DSC_0001.jpg") or "")
    reset([folder])
    assert "pickframe:Rating" not in (xmp.read_jpeg_xmp(folder / "DSC_0001.jpg") or "")


def test_manual_ratings_are_kept(result: CullResult):
    first = result.shots[0].primary
    sidecar = first.with_name(first.name + ".xmp")
    sidecar.write_text(  # rated 2 stars by hand in another program
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="2"/>'
        "</rdf:RDF></x:xmpmeta>"
    )
    report = write_result(result)
    assert sidecar in report.skipped_manual
    result.settings = Settings(overwrite_manual=True)
    assert sidecar in write_result(result).written


def test_lightroom_style_only_touches_raw(result: CullResult):
    result.settings = Settings(sidecar_style=SidecarStyle.LIGHTROOM)
    report = write_result(result)
    assert [p.name for p in report.written] == ["DSC_0004.xmp"]


def test_errors_are_reported(tmp_path: Path):
    save_jpeg(tmp_path / "a.jpg", texture())
    (tmp_path / "a.jpg.xmp").write_text("<not-xml")
    result = Analyzer().run([tmp_path])
    report = write_result(result)
    assert report.errors
    assert report.errors[0][0].name == "a.jpg.xmp"
    out = reset([tmp_path])
    assert out.errors
    assert out.errors[0][0].name == "a.jpg.xmp"
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff\xe1\x00\x01")
    assert any(p.name == "b.jpg" for p, _ in reset([tmp_path / "b.jpg"]).errors)

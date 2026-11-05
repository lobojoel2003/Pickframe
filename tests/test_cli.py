from __future__ import annotations

import json
from pathlib import Path

import pytest

from pickframe import cli, models
from pickframe.cache import default_path
from test_pipeline import make_shoot


@pytest.fixture
def shoot(tmp_path: Path, astronaut_rgb, chelsea_rgb) -> Path:
    return make_shoot(tmp_path / "shoot", astronaut_rgb, chelsea_rgb)


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Pickframe ")
    assert "Python" in out


def test_help(capsys):
    assert cli.main([]) == 0
    assert "cull" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["cull"])  # paths are required


def test_cull_dry_run_with_reports(shoot: Path, tmp_path: Path, capsys):
    html, csv_path, json_path = tmp_path / "r.html", tmp_path / "r.csv", tmp_path / "r.json"
    session = tmp_path / "s.json"
    code = cli.main(
        ["cull", str(shoot), "-n", "-l", "--report", str(html), "--report", str(csv_path),
         "--report", str(json_path), "--save-session", str(session), "--workers", "2"]
    )  # fmt: skip
    assert code == 0
    assert html.is_file()
    assert csv_path.is_file()
    assert json_path.is_file()
    assert session.is_file()
    assert not list(shoot.glob("*.xmp"))
    err = capsys.readouterr().err
    assert "Would write" in err
    assert "DSC_0003.jpg" in err  # --list prints every photo


def test_cull_writes_and_reset(shoot: Path, capsys):
    assert cli.main(["cull", str(shoot), "-q", "--label", "Green", "--reject-as", "one-star"]) == 0
    sidecars = sorted(p.name for p in shoot.glob("*.xmp"))
    assert sidecars == [f"DSC_000{i}.jpg.xmp" for i in range(1, 5)]
    assert 'xmp:Label="Green"' in (shoot / "DSC_0004.jpg.xmp").read_text()
    assert cli.main(["cull", str(shoot), "--sidecar", "lightroom"]) == 0
    assert "Lightroom ignores sidecars for JPEG" in capsys.readouterr().err
    assert cli.main(["reset", str(shoot), "-n"]) == 0
    assert len(list(shoot.glob("*.xmp"))) == 4
    assert cli.main(["reset", str(shoot)]) == 0
    assert not list(shoot.glob("*.xmp"))


def test_cull_reports_manual_ratings_and_errors(shoot: Path, capsys):
    (shoot / "DSC_0001.jpg.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="3"/>'
        "</rdf:RDF></x:xmpmeta>"
    )
    (shoot / "DSC_0002.jpg.xmp").write_text("<broken")
    assert cli.main(["cull", str(shoot)]) == 1
    err = capsys.readouterr().err
    assert "Kept 1 rating" in err
    assert "Error" in err


def test_cull_unreadable_files_are_listed(tmp_path: Path, capsys):
    for i in range(12):
        (tmp_path / f"bad{i:02d}.jpg").write_bytes(b"\xff\xd8 broken")
    assert cli.main(["cull", str(tmp_path), "-n", "--no-cache"]) == 0
    err = capsys.readouterr().err
    assert "Skipped" in err
    assert "and 2 more" in err


def test_cull_bad_report_extension(shoot: Path, tmp_path: Path, capsys):
    assert cli.main(["cull", str(shoot), "-n", "--report", str(tmp_path / "r.txt")]) == 1
    assert "Could not write report" in capsys.readouterr().err


def test_cull_errors(tmp_path: Path, capsys):
    assert cli.main(["cull", str(tmp_path / "missing")]) == 2
    assert cli.main(["cull", str(tmp_path)]) == 1  # empty folder
    assert cli.main(["cull", str(tmp_path), "--embedder", "clip"]) == 2
    err = capsys.readouterr().err
    assert "Not found" in err
    assert "No photos found" in err
    assert "CLIP model is not installed" in err
    with pytest.raises(SystemExit):
        cli.main(["cull", str(tmp_path), "--blur", "2"])
    with pytest.raises(SystemExit):
        cli.main(["cull", str(tmp_path), "--picks", "0"])


def test_info(astronaut_path: Path, tmp_path: Path, capsys):
    assert cli.main(["info", str(astronaut_path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert len(rows[0]["analysis"]["faces"]) == 1
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    assert cli.main(["info", str(astronaut_path), str(bad)]) == 0
    err = capsys.readouterr().err
    assert "Face 1" in err
    assert "Score" in err
    assert "cannot decode" in err


def test_models_commands(capsys):
    assert cli.main(["models"]) == 0
    assert cli.main(["models", "list"]) == 0
    assert "not installed" in capsys.readouterr().err
    assert cli.main(["models", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(models.models_dir())
    assert cli.main(["models", "remove", "clip"]) == 0
    assert "not installed" in capsys.readouterr().err


def test_models_download_failure(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise models.ModelError("network down")

    monkeypatch.setattr(models, "download", fail)
    assert cli.main(["models", "download", "clip"]) == 1
    assert "network down" in capsys.readouterr().err


def test_models_download_success(monkeypatch, tmp_path: Path, capsys):
    def fake(name, progress=None, force=False, cancel=None):
        if progress:
            progress(5, 10)
            progress(10, 10)
        return tmp_path / "clip.onnx"

    monkeypatch.setattr(models, "download", fake)
    assert cli.main(["models", "download"]) == 0
    assert "Saved to" in capsys.readouterr().err


def test_cache_commands(shoot: Path, capsys):
    assert cli.main(["cache", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(default_path())
    cli.main(["cull", str(shoot), "-n", "-q"])
    assert cli.main(["cache", "clear"]) == 0
    assert "Removed 4" in capsys.readouterr().err


def test_gui_command_without_display(monkeypatch, capsys):
    import pickframe.gui as gui

    monkeypatch.setattr(gui, "main", lambda argv=None: 7)
    assert cli.main(["gui"]) == 7

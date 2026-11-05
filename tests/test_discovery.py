from __future__ import annotations

from pathlib import Path

from helpers import save_jpeg, texture
from pickframe.discovery import Shot, discover, iter_images
from pickframe.formats import analysis_rank, is_image, is_jpeg, is_raw


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_format_helpers():
    assert is_image(Path("a.JPG"))
    assert is_image(Path("a.cr3"))
    assert is_image(Path("a.heic"))
    assert not is_image(Path("a.xmp"))
    assert is_raw(Path("x.NEF"))
    assert not is_raw(Path("x.jpg"))
    assert is_jpeg(Path("x.jpeg"))
    assert (
        analysis_rank(Path("a.jpg")) < analysis_rank(Path("a.png")) < analysis_rank(Path("a.cr2"))
    )


def test_raw_and_jpeg_are_one_shot(tmp_path: Path):
    save_jpeg(tmp_path / "IMG_0001.JPG", texture())
    _touch(tmp_path / "IMG_0001.CR3")
    _touch(tmp_path / "IMG_0001.CR3.xmp")  # sidecars are not photos
    save_jpeg(tmp_path / "IMG_0002.jpg", texture(seed=2))

    shots = discover([tmp_path])
    assert [s.id for s in shots] == [0, 1]
    first = shots[0]
    assert {p.name for p in first.files} == {"IMG_0001.JPG", "IMG_0001.CR3"}
    assert first.primary.name == "IMG_0001.JPG"
    assert first.label() == "IMG_0001 (CR3+JPG)"
    assert first.name == "IMG_0001.JPG"
    assert first.key.endswith("IMG_0001")
    assert shots[1].label() == "IMG_0002.jpg"


def test_recursive_and_hidden_files(tmp_path: Path):
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "sub" / "b.jpg")
    _touch(tmp_path / ".hidden" / "c.jpg")
    _touch(tmp_path / "sub" / "._b.jpg")  # macOS AppleDouble file
    _touch(tmp_path / "notes.txt")

    flat = [p.name for p in iter_images([tmp_path])]
    assert flat == ["a.jpg"]
    deep = sorted(p.name for p in iter_images([tmp_path], recursive=True))
    assert deep == ["a.jpg", "b.jpg"]


def test_files_and_duplicates(tmp_path: Path):
    a = _touch(tmp_path / "a.jpg")
    found = list(iter_images([a, tmp_path, str(a)]))
    assert found == [a.resolve()]
    assert list(iter_images([tmp_path / "missing.jpg"])) == []


def test_missing_folder_yields_nothing(tmp_path: Path):
    assert discover([tmp_path / "nope"]) == []


def test_shot_ids_are_sorted_by_folder_and_name(tmp_path: Path):
    for name in ("b.jpg", "a.jpg", "C.jpg"):
        _touch(tmp_path / name)
    shots = discover([tmp_path])
    assert [s.name for s in shots] == ["a.jpg", "b.jpg", "C.jpg"]
    assert isinstance(shots[0], Shot)

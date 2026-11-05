from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from helpers import save_jpeg, texture
from pickframe import xmp
from pickframe.config import SidecarStyle
from pickframe.xmp import XmpError, XmpUpdate

DARKTABLE_SIDECAR = """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:darktable="http://darktable.sf.net/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmp:Rating="0"
   darktable:xmp_version="5"
   darktable:history_end="2">
   <dc:subject><rdf:Bag><rdf:li>holiday</rdf:li></rdf:Bag></dc:subject>
   <darktable:history><rdf:Seq><rdf:li darktable:operation="exposure" darktable:enabled="1"/></rdf:Seq></darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

CHILD_ELEMENT_SIDECAR = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/">
   <xmp:Rating>3</xmp:Rating>
  </rdf:Description>
  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:Exposure2012="+0.50"/>
 </rdf:RDF>
</x:xmpmeta>
"""

PICK_UPDATE = XmpUpdate(rating=5, flag="pick", score=0.91, group=3, status="pick", notes=())
REJECT_UPDATE = XmpUpdate(
    rating=-1, flag="reject", score=0.2, group=3, status="reject", notes=("blurry",)
)


def _root(path: Path):
    root = xmp.read_sidecar(path)
    assert root is not None
    return root


def test_sidecar_paths():
    raw = Path("/p/IMG_0001.CR3")
    jpg = Path("/p/IMG_0001.JPG")
    dng = Path("/p/IMG_0001.DNG")
    assert xmp.sidecar_paths(raw) == [Path("/p/IMG_0001.CR3.xmp"), Path("/p/IMG_0001.xmp")]
    assert xmp.sidecar_paths(jpg) == [Path("/p/IMG_0001.JPG.xmp")]
    assert xmp.sidecar_paths(dng) == [Path("/p/IMG_0001.DNG.xmp")]
    assert xmp.sidecar_paths(raw, SidecarStyle.DARKTABLE) == [Path("/p/IMG_0001.CR3.xmp")]
    assert xmp.sidecar_paths(raw, SidecarStyle.LIGHTROOM) == [Path("/p/IMG_0001.xmp")]
    assert xmp.sidecar_paths(jpg, SidecarStyle.LIGHTROOM) == []


def test_new_sidecar(tmp_path: Path):
    sidecar = tmp_path / "IMG_0001.CR3.xmp"
    assert xmp.update_sidecar(sidecar, PICK_UPDATE) == "written"
    text = sidecar.read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert 'xmp:Rating="5"' in text
    assert 'xmpDM:pick="1"' in text
    assert 'xmpDM:good="True"' in text
    assert 'digiKam:PickLabel="3"' in text
    assert 'pickframe:Rating="5"' in text
    assert 'pickframe:Created="True"' in text
    assert 'pickframe:Score="0.910"' in text
    assert 'pickframe:Group="3"' in text
    root = _root(sidecar)
    assert xmp.get_property(root, xmp.RATING) == "5"
    assert xmp.has_pickframe(root)
    assert not xmp.is_manual(root)
    # writing the same verdict again changes nothing
    assert xmp.update_sidecar(sidecar, PICK_UPDATE) == "unchanged"


def test_reject_flags(tmp_path: Path):
    sidecar = tmp_path / "a.jpg.xmp"
    xmp.update_sidecar(sidecar, REJECT_UPDATE)
    root = _root(sidecar)
    assert xmp.get_property(root, xmp.RATING) == "-1"
    assert xmp.get_property(root, xmp.PICK) == "-1"
    assert xmp.get_property(root, xmp.GOOD) == "False"
    assert xmp.get_property(root, xmp.PICK_LABEL) == "1"
    assert xmp.get_property(root, xmp.PF_NOTES) == "blurry"
    # becoming an alternate clears the flags Pickframe set
    xmp.update_sidecar(sidecar, XmpUpdate(rating=2, flag="none", status="alternate"))
    root = _root(sidecar)
    assert xmp.get_property(root, xmp.PICK) is None
    assert xmp.get_property(root, xmp.PICK_LABEL) is None
    assert xmp.get_property(root, xmp.RATING) == "2"


def test_merge_preserves_darktable_data(tmp_path: Path):
    sidecar = tmp_path / "IMG_0002.NEF.xmp"
    sidecar.write_text(DARKTABLE_SIDECAR, encoding="utf-8")
    assert xmp.update_sidecar(sidecar, PICK_UPDATE) == "written"
    text = sidecar.read_text(encoding="utf-8")
    assert 'darktable:operation="exposure"' in text
    assert "<rdf:li>holiday</rdf:li>" in text
    assert 'darktable:history_end="2"' in text
    assert 'x:xmptk="XMP Core 4.4.0-Exiv2"' in text
    assert "ns0:" not in text  # prefixes are kept
    assert text.count("xmp:Rating") == 1
    root = _root(sidecar)
    assert xmp.get_property(root, xmp.PF_PREVIOUS) == "0"
    assert xmp.get_property(root, xmp.PF_CREATED) is None


def test_child_element_rating_updated_in_place(tmp_path: Path):
    sidecar = tmp_path / "IMG_0003.CR2.xmp"
    sidecar.write_text(CHILD_ELEMENT_SIDECAR, encoding="utf-8")
    # 3 stars set by hand: protected
    assert xmp.update_sidecar(sidecar, PICK_UPDATE) == "manual"
    assert xmp.update_sidecar(sidecar, PICK_UPDATE, overwrite_manual=True) == "written"
    text = sidecar.read_text(encoding="utf-8")
    assert "<xmp:Rating>5</xmp:Rating>" in text
    assert 'xmp:Rating="' not in text
    assert 'crs:Exposure2012="+0.50"' in text
    assert xmp.get_property(_root(sidecar), xmp.PF_PREVIOUS) == "3"


def test_manual_rating_after_pickframe(tmp_path: Path):
    sidecar = tmp_path / "x.jpg.xmp"
    xmp.update_sidecar(sidecar, PICK_UPDATE)
    sidecar.write_text(sidecar.read_text().replace('xmp:Rating="5"', 'xmp:Rating="2"'))
    assert xmp.is_manual(_root(sidecar))
    assert xmp.update_sidecar(sidecar, REJECT_UPDATE) == "manual"
    assert xmp.get_property(_root(sidecar), xmp.RATING) == "2"


def test_dry_run_writes_nothing(tmp_path: Path):
    sidecar = tmp_path / "x.jpg.xmp"
    assert xmp.update_sidecar(sidecar, PICK_UPDATE, dry_run=True) == "written"
    assert not sidecar.exists()


def test_rating_none_and_labels(tmp_path: Path):
    sidecar = tmp_path / "x.jpg.xmp"
    xmp.update_sidecar(sidecar, XmpUpdate(rating=4, flag="pick", label="Green"))
    root = _root(sidecar)
    assert xmp.get_property(root, xmp.LABEL) == "Green"
    # reject style "none": Pickframe's rating is withdrawn, the label too
    xmp.update_sidecar(sidecar, XmpUpdate(rating=None, flag="none"))
    root = _root(sidecar)
    assert xmp.get_property(root, xmp.RATING) is None
    assert xmp.get_property(root, xmp.LABEL) is None
    assert xmp.get_property(root, xmp.PF_RATING) == ""


def test_label_set_by_user_is_kept(tmp_path: Path):
    sidecar = tmp_path / "x.jpg.xmp"
    xmp.update_sidecar(sidecar, XmpUpdate(rating=4, flag="pick", label="Green"))
    sidecar.write_text(sidecar.read_text().replace('xmp:Label="Green"', 'xmp:Label="Red"'))
    xmp.update_sidecar(sidecar, XmpUpdate(rating=4, flag="pick"))
    assert xmp.get_property(_root(sidecar), xmp.LABEL) == "Red"


def test_reset_removes_created_sidecar(tmp_path: Path):
    sidecar = tmp_path / "x.jpg.xmp"
    xmp.update_sidecar(sidecar, XmpUpdate(rating=5, flag="pick", label="Green"))
    assert xmp.reset_sidecar(sidecar, dry_run=True) == "removed"
    assert sidecar.exists()
    assert xmp.reset_sidecar(sidecar) == "removed"
    assert not sidecar.exists()
    assert xmp.reset_sidecar(sidecar) == "untouched"


def test_reset_restores_previous_rating(tmp_path: Path):
    sidecar = tmp_path / "IMG_0002.NEF.xmp"
    sidecar.write_text(
        DARKTABLE_SIDECAR.replace('xmp:Rating="0"', 'xmp:Rating="1"'), encoding="utf-8"
    )
    xmp.update_sidecar(sidecar, REJECT_UPDATE, overwrite_manual=True)
    assert xmp.get_property(_root(sidecar), xmp.RATING) == "-1"
    assert xmp.reset_sidecar(sidecar) == "reset"
    root = _root(sidecar)
    assert xmp.get_property(root, xmp.RATING) == "1"
    assert xmp.get_property(root, xmp.PICK) is None
    assert not xmp.has_pickframe(root)
    assert 'darktable:operation="exposure"' in sidecar.read_text()
    assert xmp.reset_sidecar(sidecar) == "untouched"


def test_reset_keeps_user_changes(tmp_path: Path):
    sidecar = tmp_path / "x.jpg.xmp"
    xmp.update_sidecar(sidecar, PICK_UPDATE)
    sidecar.write_text(sidecar.read_text().replace('xmp:Rating="5"', 'xmp:Rating="2"'))
    assert (
        xmp.reset_sidecar(sidecar) == "reset"
    )  # created by Pickframe, but now holds the user's rating
    assert xmp.get_property(_root(sidecar), xmp.RATING) == "2"


def test_invalid_sidecars(tmp_path: Path):
    broken = tmp_path / "broken.xmp"
    broken.write_text("<x:xmpmeta")
    with pytest.raises(XmpError):
        xmp.update_sidecar(broken, PICK_UPDATE)
    with pytest.raises(XmpError):
        xmp.parse("<root/>")
    assert xmp.read_sidecar(tmp_path / "missing.xmp") is None


def test_bare_rdf_document():
    root = xmp.parse('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"></rdf:RDF>')
    xmp.set_property(root, xmp.RATING, "4")  # a Description is created on demand
    assert xmp.get_property(root, xmp.RATING) == "4"


def test_set_property_removes_duplicates():
    root = xmp.parse(CHILD_ELEMENT_SIDECAR)
    root.find(f".//{{{xmp.NS_RDF}}}Description").attrib[xmp.RATING] = "1"
    xmp.set_property(root, xmp.RATING, "4")
    text = xmp.serialize(root)
    assert text.count("Rating") == 1
    xmp.set_property(root, xmp.RATING, None)
    assert xmp.get_property(root, xmp.RATING) is None


def test_int_parsing():
    assert xmp._int("5") == 5
    assert xmp._int("4.6") == 5
    assert xmp._int(" ") is None
    assert xmp._int("abc") is None
    assert xmp._int(None) is None


def test_jpeg_embedding_roundtrip(tmp_path: Path):
    path = save_jpeg(tmp_path / "p.jpg", texture((120, 160)))
    before = Image.open(path)
    before.load()
    assert xmp.read_jpeg_xmp(path) is None
    assert xmp.update_jpeg(path, PICK_UPDATE) == "written"
    with Image.open(path) as im:
        im.load()
        assert im.size == before.size
        assert im.getexif().get(0x0110) == "Camera"  # Exif is untouched
        raw = im.info.get("xmp")  # Pillow finds the APP1 XMP segment
    assert raw is not None
    assert b'xmp:Rating="5"' in raw
    text = xmp.read_jpeg_xmp(path)
    assert text is not None
    assert 'xmp:Rating="5"' in text
    assert "<?xpacket" in text
    assert xmp.read_rating(path) == 5
    assert xmp.update_jpeg(path, PICK_UPDATE) == "unchanged"
    assert xmp.update_jpeg(path, REJECT_UPDATE) == "written"
    assert xmp.read_rating(path) == -1
    assert xmp.reset_jpeg(path) == "reset"
    assert xmp.read_rating(path) is None
    assert xmp.reset_jpeg(path) == "untouched"
    with Image.open(path) as im:
        im.load()


def test_jpeg_dry_run_and_manual(tmp_path: Path):
    path = save_jpeg(tmp_path / "p.jpg", texture((60, 80)))
    original = path.read_bytes()
    assert xmp.update_jpeg(path, PICK_UPDATE, dry_run=True) == "written"
    assert path.read_bytes() == original
    assert xmp.reset_jpeg(path) == "untouched"
    xmp.update_jpeg(path, PICK_UPDATE)
    data = path.read_bytes().replace(b'xmp:Rating="5"', b'xmp:Rating="3"')
    path.write_bytes(data)
    assert xmp.update_jpeg(path, REJECT_UPDATE) == "manual"
    assert xmp.update_jpeg(path, REJECT_UPDATE, overwrite_manual=True) == "written"


def test_jpeg_packet_too_large(tmp_path: Path):
    path = save_jpeg(tmp_path / "p.jpg", texture((60, 80)))
    huge = XmpUpdate(rating=5, flag="pick", notes=["x" * 70000])
    with pytest.raises(XmpError, match="too large"):
        xmp.update_jpeg(path, huge)


def test_jpeg_packet_without_padding_still_fits(tmp_path: Path):
    path = save_jpeg(tmp_path / "p.jpg", texture((60, 80)))
    big = XmpUpdate(rating=5, flag="pick", notes=["y" * 63000])
    assert xmp.update_jpeg(path, big) == "written"
    assert xmp.read_rating(path) == 5


def test_corrupt_jpeg(tmp_path: Path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not a jpeg")
    with pytest.raises(XmpError):
        xmp.update_jpeg(path, PICK_UPDATE)
    path.write_bytes(b"\xff\xd8\x00\x01\x02\x03")
    with pytest.raises(XmpError):
        xmp.update_jpeg(path, PICK_UPDATE)
    path.write_bytes(b"\xff\xd8\xff\xe1\x00\x01")
    with pytest.raises(XmpError):
        xmp.update_jpeg(path, PICK_UPDATE)
    path.write_bytes(b"\xff\xd8\xff\xe1\xff\x00abc")
    with pytest.raises(XmpError):
        xmp.update_jpeg(path, PICK_UPDATE)
    assert xmp.read_rating(path) is None


def test_read_rating_sidecars(tmp_path: Path):
    image = tmp_path / "IMG_9.CR3"
    image.write_bytes(b"raw")
    assert xmp.read_rating(image) is None
    xmp.update_sidecar(tmp_path / "IMG_9.xmp", XmpUpdate(rating=4, flag="pick"))
    assert xmp.read_rating(image) == 4
    (tmp_path / "IMG_9.CR3.xmp").write_text("<broken")
    assert xmp.read_rating(image) == 4  # unreadable sidecars are skipped


def test_write_report_merge():
    a, b = xmp.WriteReport(), xmp.WriteReport()
    a.written.append(Path("a"))
    b.errors.append((Path("b"), "boom"))
    b.removed.append(Path("c"))
    a.merge(b)
    assert a.written == [Path("a")]
    assert a.errors == [(Path("b"), "boom")]
    assert a.removed == [Path("c")]

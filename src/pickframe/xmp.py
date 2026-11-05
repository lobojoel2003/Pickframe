"""Read and write star ratings and pick/reject flags in XMP metadata.

What gets written, and who reads it:

=====================  =========================================================
``xmp:Rating``         Stars 1-5, ``-1`` = rejected. Lightroom, darktable,
                       digiKam, Capture One, Bridge, Photo Mechanic, ...
``xmpDM:pick/good``    Pick (1/True) and reject (-1/False) flags as written by
                       Lightroom Classic 13.2 and later.
``digiKam:PickLabel``  digiKam pick label: 3 = accepted, 1 = rejected.
``xmp:Label``          Optional colour label for picks (e.g. ``Green``).
``pickframe:*``        Score, group and what Pickframe wrote, so a later run can
                       tell your own edits apart and ``pickframe reset`` can
                       undo everything.
=====================  =========================================================

Sidecar names: ``IMG_0001.CR3.xmp`` (darktable, digiKam) and, for raw files,
``IMG_0001.xmp`` (Lightroom, Capture One, Bridge). Existing sidecars are
merged, never replaced: develop settings, keywords and other metadata stay.
Lightroom ignores sidecars for JPEG files; for those, Pickframe can embed the
XMP packet in the JPEG itself (opt-in).
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from . import __version__
from .config import SidecarStyle
from .formats import is_jpeg, is_raw

NS_X = "adobe:ns:meta/"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_XMP = "http://ns.adobe.com/xap/1.0/"
NS_XMPDM = "http://ns.adobe.com/xmp/1.0/DynamicMedia/"
NS_DIGIKAM = "http://www.digikam.org/ns/1.0/"
NS_PICKFRAME = "https://github.com/lobojoel2003/Pickframe/ns/xmp/1.0/"

_PREFERRED_PREFIXES = {
    NS_X: "x",
    NS_RDF: "rdf",
    NS_XMP: "xmp",
    NS_XMPDM: "xmpDM",
    NS_DIGIKAM: "digiKam",
    NS_PICKFRAME: "pickframe",
}

RATING = f"{{{NS_XMP}}}Rating"
LABEL = f"{{{NS_XMP}}}Label"
PICK = f"{{{NS_XMPDM}}}pick"
GOOD = f"{{{NS_XMPDM}}}good"
PICK_LABEL = f"{{{NS_DIGIKAM}}}PickLabel"
PF_RATING = f"{{{NS_PICKFRAME}}}Rating"
PF_PREVIOUS = f"{{{NS_PICKFRAME}}}PreviousRating"
PF_LABEL = f"{{{NS_PICKFRAME}}}Label"
PF_FLAG = f"{{{NS_PICKFRAME}}}Flag"
PF_CREATED = f"{{{NS_PICKFRAME}}}Created"
PF_SCORE = f"{{{NS_PICKFRAME}}}Score"
PF_GROUP = f"{{{NS_PICKFRAME}}}Group"
PF_STATUS = f"{{{NS_PICKFRAME}}}Status"
PF_NOTES = f"{{{NS_PICKFRAME}}}Flags"
PF_VERSION = f"{{{NS_PICKFRAME}}}Version"
_ABOUT = f"{{{NS_RDF}}}about"
_DESCRIPTION = f"{{{NS_RDF}}}Description"
_RDF = f"{{{NS_RDF}}}RDF"

JPEG_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
_MAX_JPEG_XMP = 65502  # APP1 segment payload limit minus the namespace header

# ElementTree keeps one global prefix registry; guard parse + serialise.
_ET_LOCK = threading.Lock()


class XmpError(Exception):
    """Raised for unreadable XMP or files that cannot be updated."""


@dataclass
class XmpUpdate:
    """What to write for one file."""

    rating: int | None  # -1..5, None = remove Pickframe's rating
    flag: str = "none"  # "pick", "reject" or "none"
    label: str = ""
    score: float | None = None
    group: int | None = None
    status: str = ""
    notes: Iterable[str] = ()


@dataclass
class WriteReport:
    written: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    skipped_manual: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)

    def merge(self, other: WriteReport) -> None:
        self.written += other.written
        self.unchanged += other.unchanged
        self.skipped_manual += other.skipped_manual
        self.removed += other.removed
        self.errors += other.errors


# -- sidecar names -------------------------------------------------------------
def sidecar_paths(image: Path, style: SidecarStyle = SidecarStyle.AUTO) -> list[Path]:
    """Sidecar files to write for ``image`` under the given naming style."""
    darktable = image.with_name(image.name + ".xmp")
    lightroom = image.with_suffix(".xmp")
    # Lightroom reads IMG_0001.xmp only for proprietary raw files (not DNG/JPEG/TIFF).
    lightroom_ok = is_raw(image) and image.suffix.lower() != ".dng"
    if style == SidecarStyle.DARKTABLE:
        return [darktable]
    if style == SidecarStyle.LIGHTROOM:
        return [lightroom] if lightroom_ok else []
    return [darktable, lightroom] if lightroom_ok else [darktable]


# -- XML helpers -----------------------------------------------------------------
def _register_namespaces(text: str) -> None:
    for uri, prefix in _PREFERRED_PREFIXES.items():
        ET.register_namespace(prefix, uri)
    try:
        for _event, (prefix, uri) in ET.iterparse(StringIO(text), events=("start-ns",)):
            if uri in _PREFERRED_PREFIXES or not prefix:
                continue
            with contextlib.suppress(ValueError):  # reserved "nsN" prefixes
                ET.register_namespace(prefix, uri)
    except ET.ParseError:
        pass


def new_document() -> ET.Element:
    root = ET.Element(f"{{{NS_X}}}xmpmeta", {f"{{{NS_X}}}xmptk": f"Pickframe {__version__}"})
    rdf = ET.SubElement(root, _RDF)
    ET.SubElement(rdf, _DESCRIPTION, {_ABOUT: ""})
    return root


def parse(text: str) -> ET.Element:
    text = text.lstrip("\ufeff")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise XmpError(f"invalid XMP: {exc}") from exc
    if root.find(f".//{_RDF}") is None and root.tag != _RDF:
        raise XmpError("XMP document has no rdf:RDF element")
    return root


def _descriptions(root: ET.Element) -> list[ET.Element]:
    rdf = root if root.tag == _RDF else root.find(f".//{_RDF}")
    if rdf is None:
        return []
    found = rdf.findall(_DESCRIPTION)
    if not found:
        found = [ET.SubElement(rdf, _DESCRIPTION, {_ABOUT: ""})]
    return found


def get_property(root: ET.Element, name: str) -> str | None:
    """Read a simple property stored as an attribute or a child element."""
    for desc in _descriptions(root):
        if name in desc.attrib:
            return desc.attrib[name]
        child = desc.find(name)
        if child is not None:
            return (child.text or "").strip()
    return None


def set_property(root: ET.Element, name: str, value: str | None) -> None:
    """Set (or with ``None`` remove) a simple property, wherever it lives."""
    descriptions = _descriptions(root)
    placed = False
    for desc in descriptions:
        if name in desc.attrib:
            if value is None or placed:
                del desc.attrib[name]
            else:
                desc.attrib[name] = value
                placed = True
        for child in desc.findall(name):
            if value is None or placed:
                desc.remove(child)
            else:
                child.text = value
                child.attrib.clear()
                for sub in list(child):
                    child.remove(sub)
                placed = True
    if value is not None and not placed:
        descriptions[0].attrib[name] = value


def _int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return round(float(value))
    except ValueError:
        return None


def serialize(root: ET.Element) -> str:
    ET.indent(root, space=" ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


# -- merging -----------------------------------------------------------------------
def is_manual(root: ET.Element) -> bool:
    """True when the current rating was not written by Pickframe."""
    current = _int(get_property(root, RATING))
    if current is None or current == 0:
        return False
    ours = _int(get_property(root, PF_RATING))
    return ours is None or ours != current


def apply_update(root: ET.Element, update: XmpUpdate, created: bool) -> None:
    """Write Pickframe's verdict into a parsed XMP document."""
    first_time = get_property(root, PF_RATING) is None and get_property(root, PF_FLAG) is None
    if first_time:
        previous = get_property(root, RATING)
        set_property(root, PF_PREVIOUS, previous if previous is not None else "")
        if created:
            set_property(root, PF_CREATED, "True")

    if update.rating is None:
        if not first_time and _int(get_property(root, RATING)) == _int(
            get_property(root, PF_RATING)
        ):
            previous = get_property(root, PF_PREVIOUS)
            set_property(root, RATING, previous or None)
        set_property(root, PF_RATING, "")
    else:
        set_property(root, RATING, str(update.rating))
        set_property(root, PF_RATING, str(update.rating))

    previous_flag = get_property(root, PF_FLAG)
    if update.flag in ("pick", "reject"):
        set_property(root, PICK, "1" if update.flag == "pick" else "-1")
        set_property(root, GOOD, "True" if update.flag == "pick" else "False")
        set_property(root, PICK_LABEL, "3" if update.flag == "pick" else "1")
    elif previous_flag in ("pick", "reject"):
        set_property(root, PICK, None)
        set_property(root, GOOD, None)
        set_property(root, PICK_LABEL, None)
    set_property(root, PF_FLAG, update.flag)

    previous_label = get_property(root, PF_LABEL)
    if update.label:
        set_property(root, LABEL, update.label)
        set_property(root, PF_LABEL, update.label)
    elif previous_label:
        if get_property(root, LABEL) == previous_label:
            set_property(root, LABEL, None)
        set_property(root, PF_LABEL, None)

    set_property(root, PF_SCORE, None if update.score is None else f"{update.score:.3f}")
    set_property(root, PF_GROUP, None if update.group is None else str(update.group))
    set_property(root, PF_STATUS, update.status or None)
    notes = ",".join(update.notes)
    set_property(root, PF_NOTES, notes or None)
    set_property(root, PF_VERSION, __version__)


_PICKFRAME_PROPS = (
    PF_RATING,
    PF_PREVIOUS,
    PF_LABEL,
    PF_FLAG,
    PF_CREATED,
    PF_SCORE,
    PF_GROUP,
    PF_STATUS,
    PF_NOTES,
    PF_VERSION,
)


def remove_pickframe(root: ET.Element) -> bool:
    """Undo Pickframe's changes. Returns True if the document is now empty."""
    ours = get_property(root, PF_RATING)
    if ours is not None and _int(get_property(root, RATING)) == _int(ours) and ours != "":
        previous = get_property(root, PF_PREVIOUS)
        set_property(root, RATING, previous or None)
    if get_property(root, PF_FLAG) in ("pick", "reject"):
        for name in (PICK, GOOD, PICK_LABEL):
            set_property(root, name, None)
    label = get_property(root, PF_LABEL)
    if label and get_property(root, LABEL) == label:
        set_property(root, LABEL, None)
    created = get_property(root, PF_CREATED) == "True"
    for name in _PICKFRAME_PROPS:
        set_property(root, name, None)
    if not created:
        return False
    for desc in _descriptions(root):
        meaningful = {k for k in desc.attrib if k != _ABOUT}
        if meaningful or len(desc):
            return False
    return True


def has_pickframe(root: ET.Element) -> bool:
    return any(get_property(root, name) is not None for name in (PF_RATING, PF_FLAG))


# -- files ---------------------------------------------------------------------
def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if path.exists():
            with contextlib.suppress(OSError):
                os.chmod(tmp, path.stat().st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_sidecar(path: Path) -> ET.Element | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    with _ET_LOCK:
        _register_namespaces(text)
        return parse(text)


def read_rating(image: Path, style: SidecarStyle = SidecarStyle.AUTO) -> int | None:
    """Current rating of ``image`` from its sidecar(s) or embedded XMP."""
    for sidecar in [*sidecar_paths(image, style), image.with_suffix(".xmp")]:
        try:
            root = read_sidecar(sidecar)
        except (OSError, XmpError):
            continue
        if root is not None:
            value = _int(get_property(root, RATING))
            if value is not None:
                return value
    if is_jpeg(image):
        try:
            text = read_jpeg_xmp(image)
        except (OSError, XmpError):
            return None
        if text:
            with _ET_LOCK:
                _register_namespaces(text)
                return _int(get_property(_parse_packet(text), RATING))
    return None


def update_sidecar(
    sidecar: Path, update: XmpUpdate, overwrite_manual: bool = False, dry_run: bool = False
) -> str:
    """Merge ``update`` into ``sidecar``. Returns written/unchanged/manual."""
    exists = sidecar.is_file()
    text = sidecar.read_text(encoding="utf-8", errors="replace") if exists else ""
    with _ET_LOCK:
        if exists:
            _register_namespaces(text)
            root = parse(text)
        else:
            _register_namespaces("")
            root = new_document()
        if update.rating is not None and not overwrite_manual and is_manual(root):
            return "manual"
        before = serialize(root) if exists else None
        apply_update(root, update, created=not exists)
        after = serialize(root)
    if after == before:
        return "unchanged"
    if not dry_run:
        _atomic_write(sidecar, after.encode("utf-8"))
    return "written"


def reset_sidecar(sidecar: Path, dry_run: bool = False) -> str:
    """Remove Pickframe's changes from a sidecar: removed/reset/untouched."""
    if not sidecar.is_file():
        return "untouched"
    text = sidecar.read_text(encoding="utf-8", errors="replace")
    with _ET_LOCK:
        _register_namespaces(text)
        root = parse(text)
        if not has_pickframe(root):
            return "untouched"
        empty = remove_pickframe(root)
        after = serialize(root)
    if dry_run:
        return "removed" if empty else "reset"
    if empty:
        sidecar.unlink()
        return "removed"
    _atomic_write(sidecar, after.encode("utf-8"))
    return "reset"


# -- XMP embedded in JPEG files ---------------------------------------------------
def _jpeg_segments(data: bytes, partial: bool = False) -> list[tuple[int, int, int]]:
    """(marker, start, end) of the header segments before the image data.

    With ``partial`` the data may be a truncated read of the file; parsing then
    stops quietly at the end of the buffer.
    """
    if data[:2] != b"\xff\xd8":
        raise XmpError("not a JPEG file")
    segments = []
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise XmpError("corrupt JPEG header")
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker in (0xDA, 0xD9):
            break
        length = int.from_bytes(data[pos + 2 : pos + 4], "big")
        if length < 2:
            raise XmpError("corrupt JPEG segment")
        if pos + 2 + length > len(data):
            if partial:
                break
            raise XmpError("corrupt JPEG segment")
        segments.append((marker, pos, pos + 2 + length))
        pos += 2 + length
    return segments


def read_jpeg_xmp(path: Path) -> str | None:
    with path.open("rb") as fh:
        head = fh.read(1 << 20)
    for marker, start, end in _jpeg_segments(head, partial=True):
        payload = head[start + 4 : end]
        if marker == 0xE1 and payload.startswith(JPEG_XMP_HEADER):
            return payload[len(JPEG_XMP_HEADER) :].decode("utf-8", errors="replace")
    return None


def _packet(xml_text: str, padding: bool = True) -> bytes:
    body = xml_text.split("?>", 1)[1].strip() if xml_text.startswith("<?xml") else xml_text.strip()
    pad = "\n".join([" " * 99] * 20) + "\n" if padding else ""  # room for in-place edits
    packet = f'<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n{body}\n{pad}<?xpacket end="w"?>'
    return packet.encode("utf-8")


def _embedded(data: bytes) -> tuple[tuple[int, int, str] | None, int]:
    """Existing XMP segment (start, end, text) and where a new one should go."""
    existing = None
    insert_at = 2
    for marker, start, end in _jpeg_segments(data):
        payload = data[start + 4 : end]
        if marker == 0xE1 and payload.startswith(JPEG_XMP_HEADER):
            existing = (
                start,
                end,
                payload[len(JPEG_XMP_HEADER) :].decode("utf-8", errors="replace"),
            )
            break
        if marker in (0xE0, 0xE1):
            insert_at = end  # after JFIF / Exif, as the XMP specification recommends
    return existing, insert_at


def _replace_embedded(
    data: bytes, existing: tuple[int, int, str] | None, insert_at: int, xml_text: str
) -> bytes:
    packet = _packet(xml_text)
    if len(packet) > _MAX_JPEG_XMP:
        packet = _packet(xml_text, padding=False)
        if len(packet) > _MAX_JPEG_XMP:
            raise XmpError("XMP packet too large to embed in a JPEG")
    payload = JPEG_XMP_HEADER + packet
    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    if existing is not None:
        return data[: existing[0]] + segment + data[existing[1] :]
    return data[:insert_at] + segment + data[insert_at:]


def _parse_packet(text: str) -> ET.Element:
    return parse(re.sub(r"<\?xpacket[^>]*\?>", "", text).strip())


def update_jpeg(
    path: Path, update: XmpUpdate, overwrite_manual: bool = False, dry_run: bool = False
) -> str:
    """Merge ``update`` into the XMP packet embedded in a JPEG file."""
    data = path.read_bytes()
    existing, insert_at = _embedded(data)
    with _ET_LOCK:
        if existing is not None:
            _register_namespaces(existing[2])
            root = _parse_packet(existing[2])
        else:
            _register_namespaces("")
            root = new_document()
        if update.rating is not None and not overwrite_manual and is_manual(root):
            return "manual"
        before = serialize(root) if existing is not None else None
        apply_update(root, update, created=False)
        after = serialize(root)
    if after == before:
        return "unchanged"
    if not dry_run:
        _atomic_write(path, _replace_embedded(data, existing, insert_at, after))
    return "written"


def reset_jpeg(path: Path, dry_run: bool = False) -> str:
    """Remove Pickframe's changes from XMP embedded in a JPEG: reset/untouched."""
    data = path.read_bytes()
    existing, insert_at = _embedded(data)
    if existing is None:
        return "untouched"
    with _ET_LOCK:
        _register_namespaces(existing[2])
        root = _parse_packet(existing[2])
        if not has_pickframe(root):
            return "untouched"
        remove_pickframe(root)
        after = serialize(root)
    if not dry_run:
        _atomic_write(path, _replace_embedded(data, existing, insert_at, after))
    return "reset"

"""Read capture time and orientation from photos, including camera raw files.

Pillow covers JPEG, TIFF, PNG, WebP and HEIF. Most raw formats (CR2, NEF,
ARW, DNG, ORF, RW2, PEF, ...) are TIFF containers, which the small parser
below reads directly; Canon CR3 keeps its EXIF in ``CMT1``/``CMT2`` boxes.
"""

from __future__ import annotations

import calendar
import struct
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

_TAG_ORIENTATION = 0x0112
_TAG_DATETIME = 0x0132
_TAG_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_SUBSEC_ORIGINAL = 0x9291
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110

_TIFF_MAGIC = {42, 0x4F52, 0x5352, 0x55}  # TIFF, Olympus ORF ("RO"/"SR"), Panasonic RW2
_HEAD_BYTES = 1 << 20


@dataclass
class ExifInfo:
    capture_time: float | None = None  # seconds since epoch (camera local time)
    orientation: int = 1
    camera: str = ""


def parse_exif_datetime(value: object, subsec: object = None) -> float | None:
    """Convert ``"2024:05:01 14:03:22"`` (+ sub-seconds) to a timestamp."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip("\x00").strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text[:19], fmt)
            break
        except ValueError:
            continue
    else:
        return None
    stamp = float(calendar.timegm(dt.timetuple()))
    if isinstance(subsec, bytes):
        subsec = subsec.decode("ascii", "ignore")
    if isinstance(subsec, (str, int)):
        digits = "".join(ch for ch in str(subsec) if ch.isdigit())
        if digits:
            stamp += int(digits) / 10 ** len(digits)
    return stamp


def camera_name(make: str, model: str) -> str:
    """``("NIKON CORPORATION", "NIKON D3S")`` -> ``"NIKON D3S"``; ``("Canon", "EOS R5")`` -> ``"Canon EOS R5"``."""
    make, model = make.strip(), model.strip()
    brand = make.split()[0].lower() if make else ""
    if not model:
        return make
    if brand and brand in model.lower():
        return model
    return f"{make} {model}".strip()


def _from_pillow_exif(exif: Image.Exif) -> ExifInfo:
    info = ExifInfo()
    try:
        sub = exif.get_ifd(_TAG_EXIF_IFD)
    except Exception:
        sub = {}
    info.capture_time = parse_exif_datetime(
        sub.get(_TAG_DATETIME_ORIGINAL), sub.get(_TAG_SUBSEC_ORIGINAL)
    ) or parse_exif_datetime(exif.get(_TAG_DATETIME))
    orientation = exif.get(_TAG_ORIENTATION, 1)
    info.orientation = orientation if isinstance(orientation, int) and 1 <= orientation <= 8 else 1
    make = str(exif.get(_TAG_MAKE, "") or "").strip("\x00 ")
    model = str(exif.get(_TAG_MODEL, "") or "").strip("\x00 ")
    info.camera = camera_name(make, model)
    return info


def read_pillow(image: Image.Image) -> ExifInfo:
    try:
        return _from_pillow_exif(image.getexif())
    except Exception:
        return ExifInfo()


def read_jpeg_bytes(data: bytes) -> ExifInfo:
    try:
        with Image.open(BytesIO(data)) as im:
            return read_pillow(im)
    except Exception:
        return ExifInfo()


class _Tiff:
    """Just enough TIFF to read a handful of ASCII/SHORT tags."""

    def __init__(self, data: bytes, base: int = 0):
        self.data = data
        self.base = base
        order = data[base : base + 2]
        if order == b"II":
            self.endian = "<"
        elif order == b"MM":
            self.endian = ">"
        else:
            raise ValueError("not a TIFF header")
        (magic,) = struct.unpack_from(self.endian + "H", data, base + 2)
        if magic not in _TIFF_MAGIC:
            raise ValueError("bad TIFF magic")
        (self.first_ifd,) = struct.unpack_from(self.endian + "I", data, base + 4)

    def entries(self, offset: int) -> dict[int, object]:
        data, e, base = self.data, self.endian, self.base
        out: dict[int, object] = {}
        pos = base + offset
        if pos + 2 > len(data):
            return out
        (count,) = struct.unpack_from(e + "H", data, pos)
        for i in range(min(count, 512)):
            entry = pos + 2 + i * 12
            if entry + 12 > len(data):
                break
            tag, typ, n = struct.unpack_from(e + "HHI", data, entry)
            if typ == 2:  # ASCII
                if n <= 4:
                    raw = data[entry + 8 : entry + 8 + n]
                else:
                    (ptr,) = struct.unpack_from(e + "I", data, entry + 8)
                    raw = data[base + ptr : base + ptr + n]
                out[tag] = raw.split(b"\x00", 1)[0].decode("ascii", "ignore")
            elif typ == 3 and n >= 1:  # SHORT
                (out[tag],) = struct.unpack_from(e + "H", data, entry + 8)
            elif typ in (4, 13) and n >= 1:  # LONG / IFD
                (out[tag],) = struct.unpack_from(e + "I", data, entry + 8)
        return out


def _from_tiff(data: bytes, base: int = 0, exif_ifd_only: bool = False) -> ExifInfo | None:
    try:
        tiff = _Tiff(data, base)
    except (ValueError, struct.error):
        return None
    info = ExifInfo()
    try:
        ifd0 = tiff.entries(tiff.first_ifd)
        sub = ifd0 if exif_ifd_only else {}
        pointer = ifd0.get(_TAG_EXIF_IFD)
        if isinstance(pointer, int) and not exif_ifd_only:
            sub = tiff.entries(pointer)
    except struct.error:
        return None
    info.capture_time = parse_exif_datetime(
        sub.get(_TAG_DATETIME_ORIGINAL), sub.get(_TAG_SUBSEC_ORIGINAL)
    ) or (None if exif_ifd_only else parse_exif_datetime(ifd0.get(_TAG_DATETIME)))
    orientation = ifd0.get(_TAG_ORIENTATION)
    if not exif_ifd_only and isinstance(orientation, int) and 1 <= orientation <= 8:
        info.orientation = orientation
    make = str(ifd0.get(_TAG_MAKE, "")).strip()
    model = str(ifd0.get(_TAG_MODEL, "")).strip()
    info.camera = camera_name(make, model)
    return info


def _from_cr3(head: bytes) -> ExifInfo | None:
    info = ExifInfo()
    found = False
    cmt1 = head.find(b"CMT1")
    if cmt1 > 0:
        ifd0 = _from_tiff(head, cmt1 + 4)
        if ifd0:
            info.orientation, info.camera = ifd0.orientation, ifd0.camera
            info.capture_time = ifd0.capture_time
            found = True
    cmt2 = head.find(b"CMT2")
    if cmt2 > 0:
        exif = _from_tiff(head, cmt2 + 4, exif_ifd_only=True)
        if exif and exif.capture_time:
            info.capture_time = exif.capture_time
            found = True
    return info if found else None


def read_raw_head(path: Path) -> ExifInfo | None:
    """Parse EXIF from the first megabyte of a raw file, if possible."""
    try:
        with path.open("rb") as fh:
            head = fh.read(_HEAD_BYTES)
    except OSError:
        return None
    if head[4:8] == b"ftyp" and b"crx " in head[8:24]:
        return _from_cr3(head)
    if head.startswith(b"FUJIFILM") and len(head) >= 92:
        (offset, length) = struct.unpack_from(">II", head, 84)
        if offset + length <= len(head):
            return read_jpeg_bytes(head[offset : offset + length])
        try:
            with path.open("rb") as fh:
                fh.seek(offset)
                return read_jpeg_bytes(fh.read(length))
        except OSError:
            return None
    return _from_tiff(head, 0)


def read_file(path: Path) -> ExifInfo:
    """Best-effort EXIF for any supported file."""
    from .formats import is_raw

    if is_raw(path):
        info = read_raw_head(path)
        if info is not None:
            return info
        return ExifInfo()
    try:
        with Image.open(path) as im:
            return read_pillow(im)
    except Exception:
        return ExifInfo()

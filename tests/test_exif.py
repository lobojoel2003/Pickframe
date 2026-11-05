from __future__ import annotations

import calendar
import struct
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from helpers import jpeg_bytes, save_jpeg, texture
from pickframe import exif


def _ts(text: str) -> float:
    return float(calendar.timegm(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timetuple()))


def test_parse_exif_datetime():
    assert exif.parse_exif_datetime("2024:05:01 14:03:22") == _ts("2024-05-01 14:03:22")
    assert exif.parse_exif_datetime(b"2024:05:01 14:03:22\x00") == _ts("2024-05-01 14:03:22")
    assert exif.parse_exif_datetime("2024-05-01T14:03:22") == _ts("2024-05-01 14:03:22")
    assert (
        exif.parse_exif_datetime("2024:05:01 14:03:22", "25") == _ts("2024-05-01 14:03:22") + 0.25
    )
    assert exif.parse_exif_datetime("2024:05:01 14:03:22", b"5") == _ts("2024-05-01 14:03:22") + 0.5
    assert exif.parse_exif_datetime("2024:05:01 14:03:22", 7) == _ts("2024-05-01 14:03:22") + 0.7
    assert exif.parse_exif_datetime("0000:00:00 00:00:00") is None
    assert exif.parse_exif_datetime("garbage") is None
    assert exif.parse_exif_datetime(None) is None
    assert exif.parse_exif_datetime(12345) is None


def test_read_file_jpeg(tmp_path: Path):
    path = save_jpeg(
        tmp_path / "a.jpg", texture(), datetime(2025, 1, 2, 3, 4, 5), orientation=6, subsec="5"
    )
    info = exif.read_file(path)
    assert info.capture_time == _ts("2025-01-02 03:04:05") + 0.5
    assert info.orientation == 6
    assert info.camera == "Test Camera"


def test_read_file_without_exif(tmp_path: Path):
    path = tmp_path / "plain.png"
    Image.fromarray(texture()).save(path)
    info = exif.read_file(path)
    assert info.capture_time is None
    assert info.orientation == 1
    assert exif.read_file(tmp_path / "missing.jpg").capture_time is None


def test_datetime_fallback_to_ifd0(tmp_path: Path):
    e = Image.Exif()
    e[0x0132] = "2023:07:08 09:10:11"
    e[0x010F] = "Canon"
    e[0x0110] = "Canon EOS R5"
    path = tmp_path / "x.jpg"
    Image.fromarray(texture()).save(path, exif=e)
    info = exif.read_file(path)
    assert info.capture_time == _ts("2023-07-08 09:10:11")
    assert info.camera == "Canon EOS R5"  # model already contains the make


def build_tiff(
    endian: str = "<",
    orientation: int | None = 8,
    make: str = "NIKON CORPORATION",
    model: str = "NIKON Z 6",
    when: str | None = None,
    subsec: str | None = None,
) -> bytes:
    """A minimal TIFF: IFD0 (orientation, make, model, Exif pointer) + Exif IFD."""
    order = b"II" if endian == "<" else b"MM"

    def ascii_entries(
        entries: list[tuple[int, int, int, int | str]],
    ) -> list[tuple[int, int, int, int | bytes]]:
        return [
            (t, ty, n, v.encode() + b"\x00" if isinstance(v, str) else v) for t, ty, n, v in entries
        ]

    ifd0: list[tuple[int, int, int, int | str]] = []
    if orientation is not None:
        ifd0.append((0x0112, 3, 1, orientation))
    ifd0.append((0x010F, 2, len(make) + 1, make))
    ifd0.append((0x0110, 2, len(model) + 1, model))
    exif_entries: list[tuple[int, int, int, int | str]] = []
    if when is not None:
        exif_entries.append((0x9003, 2, len(when) + 1, when))
    if subsec is not None:
        exif_entries.append((0x9291, 2, len(subsec) + 1, subsec))
    has_exif = bool(exif_entries)
    n0 = len(ifd0) + (1 if has_exif else 0)
    ifd0_at = 8
    exif_at = ifd0_at + 2 + 12 * n0 + 4
    data_at = exif_at + (2 + 12 * len(exif_entries) + 4 if has_exif else 0)
    blobs = b""

    def pack_ifd(entries, extra=None):
        nonlocal blobs
        out = struct.pack(endian + "H", len(entries) + (1 if extra else 0))
        for tag, typ, count, value in ascii_entries(entries):
            if isinstance(value, bytes):
                if len(value) <= 4:
                    out += struct.pack(endian + "HHI", tag, typ, count) + value.ljust(4, b"\x00")
                else:
                    out += struct.pack(endian + "HHII", tag, typ, count, data_at + len(blobs))
                    blobs += value
            else:
                out += struct.pack(endian + "HHIHH", tag, typ, count, value, 0)
        if extra:
            out += struct.pack(endian + "HHII", 0x8769, 4, 1, extra)
        return out + struct.pack(endian + "I", 0)

    body = pack_ifd(ifd0, exif_at if has_exif else None)
    if has_exif:
        body += pack_ifd(exif_entries)
    return order + struct.pack(endian + "HI", 42, ifd0_at) + body + blobs


def test_raw_tiff_container(tmp_path: Path):
    path = tmp_path / "DSC_0001.NEF"
    path.write_bytes(build_tiff(when="2022:12:24 18:00:01", subsec="12"))
    info = exif.read_raw_head(path)
    assert info is not None
    assert info.capture_time == _ts("2022-12-24 18:00:01") + 0.12
    assert info.orientation == 8
    assert "Z 6" in info.camera
    # read_file routes raw extensions through the same parser
    assert exif.read_file(path).capture_time == info.capture_time


def test_raw_big_endian_tiff():
    info = exif._from_tiff(build_tiff(endian=">", orientation=3, when="2021:03:04 05:06:07"))
    assert info is not None
    assert info.orientation == 3
    assert info.capture_time == _ts("2021-03-04 05:06:07")
    assert info.camera == "NIKON Z 6"  # "NIKON CORPORATION" is redundant


@pytest.mark.parametrize(
    ("make", "model", "expected"),
    [
        ("NIKON CORPORATION", "NIKON D3S", "NIKON D3S"),
        ("Canon", "Canon EOS R5", "Canon EOS R5"),
        ("Canon", "EOS R5", "Canon EOS R5"),
        ("SONY", "ILCE-7M4", "SONY ILCE-7M4"),
        ("FUJIFILM", "X-T5", "FUJIFILM X-T5"),
        ("Apple", "", "Apple"),
        ("", "Pixel 9", "Pixel 9"),
    ],
)
def test_camera_name(make, model, expected):
    assert exif.camera_name(make, model) == expected


def test_tiff_without_exif_ifd_uses_nothing():
    info = exif._from_tiff(build_tiff(orientation=None, make="Sony", model="ILCE-7M4"))
    assert info is not None
    assert info.capture_time is None
    assert info.orientation == 1
    assert info.camera == "Sony ILCE-7M4"


def test_invalid_tiff_header():
    assert exif._from_tiff(b"XX\x00\x00") is None
    assert exif._from_tiff(b"II" + struct.pack("<HI", 99, 8)) is None


def test_cr3_boxes(tmp_path: Path):
    ifd0 = build_tiff(orientation=6, when="2020:01:01 00:00:00")
    # CMT2 holds the Exif IFD directly as the first IFD of a TIFF structure.
    when = b"2024:06:20 14:02:10\x00"
    cmt2 = b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", 1)
    cmt2 += struct.pack("<HHII", 0x9003, 2, len(when), 8 + 2 + 12 + 4) + struct.pack("<I", 0) + when
    head = struct.pack(">I", 24) + b"ftypcrx " + b"\x00" * 12
    head += struct.pack(">I", len(ifd0) + 8) + b"CMT1" + ifd0
    head += struct.pack(">I", len(cmt2) + 8) + b"CMT2" + cmt2
    path = tmp_path / "IMG_0001.CR3"
    path.write_bytes(head)
    info = exif.read_raw_head(path)
    assert info is not None
    assert info.capture_time == _ts("2024-06-20 14:02:10")
    assert info.orientation == 6


def test_cr3_without_metadata(tmp_path: Path):
    path = tmp_path / "IMG_0002.CR3"
    path.write_bytes(struct.pack(">I", 24) + b"ftypcrx " + b"\x00" * 40)
    assert exif.read_raw_head(path) is None
    assert exif.read_file(path).capture_time is None


def _jpeg_with_exif(when: str) -> bytes:
    e = Image.Exif()
    e.get_ifd(0x8769)[0x9003] = when
    buffer = BytesIO()
    Image.fromarray(texture((64, 64))).save(buffer, format="JPEG", exif=e)
    return buffer.getvalue()


def test_raf_header_points_to_jpeg(tmp_path: Path):
    preview = _jpeg_with_exif("2019:09:09 09:09:09")
    offset = 160
    header = b"FUJIFILMCCD-RAW 0201FF383501" + b"\x00" * (84 - 28)
    header += struct.pack(">II", offset, len(preview))
    header += b"\x00" * (offset - len(header))
    path = tmp_path / "DSCF0001.RAF"
    path.write_bytes(header + preview + b"\x00" * 100)
    info = exif.read_raw_head(path)
    assert info is not None
    assert info.capture_time == _ts("2019-09-09 09:09:09")


def test_raf_preview_beyond_head(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(exif, "_HEAD_BYTES", 200)
    preview = _jpeg_with_exif("2018:08:08 08:08:08")
    offset = 400
    header = b"FUJIFILMCCD-RAW 0201FF383501" + b"\x00" * (84 - 28)
    header += struct.pack(">II", offset, len(preview))
    path = tmp_path / "DSCF0002.RAF"
    path.write_bytes(header + b"\x00" * (offset - len(header)) + preview)
    info = exif.read_raw_head(path)
    assert info is not None
    assert info.capture_time == _ts("2018-08-08 08:08:08")


def test_unreadable_raw(tmp_path: Path):
    assert exif.read_raw_head(tmp_path / "missing.nef") is None
    junk = tmp_path / "junk.nef"
    junk.write_bytes(b"\x00" * 64)
    assert exif.read_raw_head(junk) is None


def test_read_jpeg_bytes():
    info = exif.read_jpeg_bytes(_jpeg_with_exif("2017:07:07 07:07:07"))
    assert info.capture_time == _ts("2017-07-07 07:07:07")
    assert exif.read_jpeg_bytes(b"not a jpeg").capture_time is None
    assert exif.read_jpeg_bytes(jpeg_bytes(texture((32, 32)))).capture_time is None

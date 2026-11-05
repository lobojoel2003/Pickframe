"""File types Pickframe understands."""

from __future__ import annotations

from pathlib import Path

#: Formats decoded directly with Pillow / OpenCV.
RASTER_EXTENSIONS = frozenset({".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".webp", ".bmp"})

#: HEIF/HEIC/AVIF need the optional ``pillow-heif`` package.
HEIF_EXTENSIONS = frozenset({".heic", ".heif", ".hif", ".avif"})

#: Camera raw formats. Pickframe analyses their embedded full-size preview.
RAW_EXTENSIONS = frozenset(
    {
        ".3fr", ".ari", ".arw", ".bay", ".cr2", ".cr3", ".crw", ".dcr", ".dng",
        ".erf", ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mef", ".mos", ".mrw",
        ".nef", ".nrw", ".orf", ".pef", ".ptx", ".raf", ".raw", ".rw2", ".rwl",
        ".sr2", ".srf", ".srw", ".x3f",
    }
)  # fmt: skip

IMAGE_EXTENSIONS = RASTER_EXTENSIONS | HEIF_EXTENSIONS | RAW_EXTENSIONS

#: Preferred file to analyse when several files share a name (RAW+JPEG pairs):
#: camera JPEGs decode fastest and match what the photographer saw.
_ANALYSIS_PREFERENCE = (".jpg", ".jpeg", ".jpe", ".heic", ".heif", ".hif", ".tif", ".tiff")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def is_jpeg(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".jpe"}


def analysis_rank(path: Path) -> int:
    """Lower is better: which member of a RAW+JPEG pair to decode."""
    suffix = path.suffix.lower()
    if suffix in _ANALYSIS_PREFERENCE:
        return _ANALYSIS_PREFERENCE.index(suffix)
    if suffix in RASTER_EXTENSIONS or suffix in HEIF_EXTENSIONS:
        return len(_ANALYSIS_PREFERENCE)
    return len(_ANALYSIS_PREFERENCE) + 1

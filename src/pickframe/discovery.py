"""Find photos on disk and pair RAW+JPEG files into shots."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .formats import analysis_rank, is_image

_SKIP_DIRS = {".git", ".svn", "__pycache__", "@eaDir", ".thumbnails", ".Trash", "$RECYCLE.BIN"}


@dataclass
class Shot:
    """One photograph: a single file, or a RAW+JPEG (+HEIC, ...) set sharing a name."""

    id: int
    files: list[Path] = field(default_factory=list)

    @property
    def primary(self) -> Path:
        """The file Pickframe decodes for analysis."""
        return min(self.files, key=lambda p: (analysis_rank(p), p.name))

    @property
    def name(self) -> str:
        return self.primary.name

    @property
    def key(self) -> str:
        return str(self.primary.parent / self.primary.stem)

    def label(self) -> str:
        if len(self.files) == 1:
            return self.files[0].name
        exts = "+".join(sorted({p.suffix.lstrip(".").upper() for p in self.files}))
        return f"{self.primary.stem} ({exts})"


def _walk(root: Path, recursive: bool) -> Iterator[Path]:
    if not recursive:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_file() and not entry.name.startswith("._"):
                yield entry
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("._"):
                yield Path(dirpath) / name


def iter_images(paths: Iterable[Path | str], recursive: bool = False) -> Iterator[Path]:
    """Yield image files from a mix of files and folders, without duplicates."""
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        candidates: Iterable[Path] = _walk(path, recursive) if path.is_dir() else [path]
        for candidate in candidates:
            if not is_image(candidate) or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def discover(paths: Iterable[Path | str], recursive: bool = False) -> list[Shot]:
    """Collect images and merge files with the same folder + stem into one shot.

    ``IMG_0001.CR3`` and ``IMG_0001.JPG`` become a single shot; ratings are
    written for both files but the JPEG is the one that gets analysed.
    """
    groups: dict[tuple[Path, str], list[Path]] = {}
    for path in iter_images(paths, recursive):
        groups.setdefault((path.parent, path.stem.lower()), []).append(path)
    shots = [
        Shot(id=0, files=sorted(files, key=lambda p: (analysis_rank(p), p.name)))
        for files in groups.values()
    ]
    shots.sort(key=lambda s: (str(s.primary.parent), s.primary.name.lower()))
    for index, shot in enumerate(shots):
        shot.id = index
    return shots

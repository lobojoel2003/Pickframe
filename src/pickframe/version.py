"""Version and build information."""

from __future__ import annotations

import platform
import sys

from . import __version__, _build


def build_number() -> int:
    try:
        return int(_build.BUILD_NUMBER)
    except ValueError:
        return 0


def version_string(verbose: bool = False) -> str:
    """Return e.g. ``Pickframe 1.0.0 (build 42, 1a2b3c4, 2026-11-05)``."""
    parts = [f"build {build_number()}"] if build_number() else [_build.CHANNEL]
    if _build.COMMIT:
        parts.append(_build.COMMIT[:7])
    if _build.BUILD_DATE:
        parts.append(_build.BUILD_DATE)
    text = f"Pickframe {__version__} ({', '.join(parts)})"
    if verbose:
        text += f"\nPython {platform.python_version()} on {platform.platform()}"
        text += f"\nFrozen: {bool(getattr(sys, 'frozen', False))}"
    return text

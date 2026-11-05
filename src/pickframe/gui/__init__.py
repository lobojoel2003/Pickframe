"""Pickframe desktop app (needs ``pip install pickframe[gui]``)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from .app import run
    except ImportError as exc:  # PySide6 missing
        message = f"The Pickframe desktop app needs PySide6 ({exc}).\nInstall it with: pip install 'pickframe[gui]'"
        print(message, file=sys.stderr)
        return 1
    return run(argv)


__all__ = ["main"]

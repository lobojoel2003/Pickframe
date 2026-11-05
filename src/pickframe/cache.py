"""SQLite cache of per-image analysis results.

Re-running Pickframe on a folder (for example after changing the grouping
or blur thresholds) only analyses files that are new or have changed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from platformdirs import user_cache_dir

from .results import Analysis

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    path TEXT NOT NULL,
    key TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (path, key)
)
"""


def default_path() -> Path:
    override = os.environ.get("PICKFRAME_CACHE_DIR")
    base = Path(override).expanduser() if override else Path(user_cache_dir("pickframe", False))
    return base / "analysis-v1.sqlite3"


class AnalysisCache:
    def __init__(self, path: Path | None = None):
        self.path = path or default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(_SCHEMA)
            self._db.commit()

    @staticmethod
    def _stat(path: Path) -> tuple[int, int] | None:
        try:
            st = path.stat()
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns

    def get(self, path: Path, key: str) -> Analysis | None:
        stat = self._stat(path)
        if stat is None:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT size, mtime_ns, data FROM analyses WHERE path = ? AND key = ?",
                (str(path), key),
            ).fetchone()
        if row is None or (row[0], row[1]) != stat:
            return None
        try:
            return Analysis.from_dict(json.loads(row[2]))
        except (ValueError, KeyError, TypeError):
            return None

    def put(self, path: Path, key: str, analysis: Analysis) -> None:
        stat = self._stat(path)
        if stat is None or analysis.error:
            return
        payload = json.dumps(analysis.to_dict(with_embedding=True), separators=(",", ":"))
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO analyses (path, key, size, mtime_ns, data) VALUES (?, ?, ?, ?, ?)",
                (str(path), key, stat[0], stat[1], payload),
            )
            self._db.commit()

    def clear(self) -> int:
        with self._lock:
            count = self._db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            self._db.execute("DELETE FROM analyses")
            self._db.commit()
            self._db.execute("VACUUM")
        return int(count)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> AnalysisCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

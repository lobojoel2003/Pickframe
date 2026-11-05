"""Group near-duplicate shots (bursts, brackets, repeated poses).

Photographers shoot similar frames back to back, so Pickframe walks through
each folder in capture order and starts a new group whenever a frame is not
similar enough to the last few frames of the current group, or when too much
time has passed since the previous frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class GroupItem:
    shot_id: int
    folder: str
    name: str
    capture_time: float | None
    embedding: np.ndarray | None


def _order(items: Sequence[GroupItem]) -> list[GroupItem]:
    """Capture order when every shot has a time stamp, otherwise file name order."""
    if items and all(item.capture_time is not None for item in items):
        return sorted(items, key=lambda i: (i.capture_time, i.name.lower()))
    return sorted(items, key=lambda i: i.name.lower())


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


def group_shots(
    items: Sequence[GroupItem],
    threshold: float,
    max_gap: float = 60.0,
    lookback: int = 3,
) -> list[list[int]]:
    """Return groups of shot ids, each in capture order."""
    by_folder: dict[str, list[GroupItem]] = {}
    for item in items:
        by_folder.setdefault(item.folder, []).append(item)

    groups: list[list[int]] = []
    for folder in sorted(by_folder):
        current: list[GroupItem] = []
        for item in _order(by_folder[folder]):
            if current:
                previous = current[-1]
                gap_ok = (
                    item.capture_time is None
                    or previous.capture_time is None
                    or abs(item.capture_time - previous.capture_time) <= max_gap
                )
                best = max(similarity(item.embedding, m.embedding) for m in current[-lookback:])
                if gap_ok and best >= threshold:
                    current.append(item)
                    continue
                groups.append([m.shot_id for m in current])
            current = [item]
        if current:
            groups.append([m.shot_id for m in current])
    return groups

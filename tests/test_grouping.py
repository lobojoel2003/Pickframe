from __future__ import annotations

import numpy as np

from pickframe.grouping import GroupItem, group_shots, similarity


def _vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


A, A2, B, C = _vec(1, 0, 0), _vec(0.97, 0.24, 0), _vec(0, 1, 0), _vec(0, 0, 1)


def item(shot_id, emb, t=None, folder="f", name=None) -> GroupItem:
    return GroupItem(
        shot_id=shot_id,
        folder=folder,
        name=name or f"{shot_id:04d}.jpg",
        capture_time=t,
        embedding=emb,
    )


def test_similarity():
    assert similarity(A, A) == np.float32(1.0)
    assert similarity(A, B) == 0.0
    assert similarity(None, A) == 0.0
    assert similarity(A, np.ones(4, np.float32)) == 0.0


def test_sequential_groups():
    items = [item(0, A, 0), item(1, A2, 1), item(2, B, 2), item(3, B, 3), item(4, C, 4)]
    assert group_shots(items, threshold=0.9) == [[0, 1], [2, 3], [4]]


def test_time_gap_splits_groups():
    items = [item(0, A, 0), item(1, A, 5), item(2, A, 500)]
    assert group_shots(items, threshold=0.9, max_gap=60) == [[0, 1], [2]]
    assert group_shots(items, threshold=0.9, max_gap=600) == [[0, 1, 2]]


def test_capture_order_beats_name_order():
    items = [
        item(0, A, 10, name="a.jpg"),
        item(1, B, 0, name="b.jpg"),
        item(2, A, 11, name="c.jpg"),
    ]
    assert group_shots(items, threshold=0.9) == [[1], [0, 2]]


def test_name_order_when_times_are_missing():
    items = [
        item(0, A, None, name="b.jpg"),
        item(1, A, 5, name="a.jpg"),
        item(2, B, None, name="c.jpg"),
    ]
    assert group_shots(items, threshold=0.9) == [[1, 0], [2]]


def test_folders_never_mix():
    items = [item(0, A, 0, folder="x"), item(1, A, 1, folder="y"), item(2, A, 2, folder="x")]
    assert group_shots(items, threshold=0.9) == [[0, 2], [1]]


def test_lookback_bridges_one_odd_frame():
    items = [item(0, A, 0), item(1, A, 1), item(2, _vec(1, 0.3, 0), 2), item(3, A, 3)]
    # frame 2 is close enough to join; frame 3 matches frame 1 via the lookback
    assert group_shots(items, threshold=0.95, lookback=3) == [[0, 1, 2, 3]]
    items = [item(0, A, 0), item(1, B, 1), item(2, A, 2)]
    assert group_shots(items, threshold=0.9, lookback=3) == [[0], [1], [2]]


def test_empty_and_missing_embeddings():
    assert group_shots([], threshold=0.5) == []
    items = [item(0, None, 0), item(1, None, 1)]
    assert group_shots(items, threshold=0.5) == [[0], [1]]

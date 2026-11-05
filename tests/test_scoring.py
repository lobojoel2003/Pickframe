from __future__ import annotations

import pytest

from pickframe import scoring
from pickframe.analysis.faces import Face
from pickframe.config import RejectStyle, Settings
from pickframe.results import (
    ALTERNATE,
    BLURRY,
    DUPLICATE,
    EYES_CLOSED,
    OVEREXPOSED,
    PICK,
    REJECT,
    SOFT,
    UNDEREXPOSED,
    UNREADABLE,
    Analysis,
    Decision,
)


def face(
    closed: bool = False,
    width: float = 0.2,
    pixels: float = 200.0,
    frontal: float = 1.0,
    sharpness: float = 1.0,
    focus_log: float = -1.0,
    smile: float = 0.5,
) -> Face:
    ratio = 0.05 if closed else 0.3
    blink = 0.9 if closed else 0.05
    return Face(
        box=(0.3, 0.2, width, width), confidence=0.9, presence=1.0, blink=(blink, blink),
        eye_ratio=(ratio, ratio), smile=smile, frontal=frontal, sharpness=sharpness,
        focus_log=focus_log, pixels=pixels,
    )  # fmt: skip


def analysis(
    sharpness: float = 1.0,
    log: float = -1.2,
    exposure: float = 1.0,
    faces: list[Face] | None = None,
    brightness: float = 0.45,
    highlights: float = 0.0,
    shadows: float = 0.0,
    contrast: float = 0.25,
    error: str | None = None,
) -> Analysis:
    return Analysis(
        path="x.jpg", sharpness=sharpness, sharpness_raw=10**log, exposure=exposure,
        brightness=brightness, highlights=highlights, shadows=shadows, contrast=contrast,
        faces=faces or [], error=error,
    )  # fmt: skip


def decide(*items: Analysis, settings: Settings | None = None, groups=None) -> list[Decision]:
    analyses = dict(enumerate(items))
    result = scoring.decide(groups or [list(analyses)], analyses, settings or Settings())
    return [result[i] for i in range(len(items))]


def test_single_sharp_shot_is_a_five_star_pick():
    (d,) = decide(analysis())
    assert d.status == PICK
    assert d.rating == 5
    assert d.flags == []
    assert d.summary == "looks good"
    assert d.rank == 1
    assert d.group == 0
    assert d.group_size == 1


def test_blurry_shot_is_rejected():
    (d,) = decide(analysis(sharpness=0.1, log=-2.9))
    assert d.status == REJECT
    assert BLURRY in d.flags
    assert d.rating == -1


def test_closed_eyes_rejected():
    (d,) = decide(analysis(faces=[face(closed=True)]))
    assert d.status == REJECT
    assert EYES_CLOSED in d.flags
    assert d.eyes_closed == 1
    assert d.faces == 1
    assert "eyes closed" in d.summary


def test_profile_faces_never_count_as_blinking():
    profile = face(closed=True, frontal=0.1)
    assert not scoring.face_closed(profile, Settings())
    (d,) = decide(analysis(faces=[profile]))
    assert EYES_CLOSED not in d.flags


def test_small_faces_need_stronger_evidence():
    s = Settings()
    assert scoring.eyes_threshold(face(pixels=200), s.eyes_threshold) == s.eyes_threshold
    assert scoring.eyes_threshold(face(pixels=40), s.eyes_threshold) == pytest.approx(
        s.eyes_threshold + 0.15
    )
    assert scoring.eyes_threshold(face(pixels=60), s.eyes_threshold) == pytest.approx(
        s.eyes_threshold + 0.075
    )
    assert scoring.eyes_threshold(face(pixels=0), s.eyes_threshold) == s.eyes_threshold
    borderline = face()
    borderline.eye_ratio = (0.13, 0.13)  # probability ~0.62 + blink part
    borderline.blink = (0.3, 0.3)
    p = borderline.eyes_closed_probability
    assert s.eyes_threshold < p < s.eyes_threshold + 0.15
    borderline.pixels = 200
    assert scoring.face_closed(borderline, s)
    borderline.pixels = 30
    assert not scoring.face_closed(borderline, s)


def test_background_faces_are_ignored():
    main = face(width=0.3)
    tiny = face(closed=True, width=0.01)  # narrower than MIN_FACE_WIDTH
    small = face(closed=True, width=0.1)  # less than 20 % of the main face's area
    assert scoring.prominent_faces([main, tiny, small]) == [main]
    assert scoring.prominent_faces([tiny]) == []
    (d,) = decide(analysis(faces=[main, tiny, small]))
    assert d.status == PICK
    assert d.faces == 1


def test_face_quality_and_smile_weight():
    s = Settings(smile_weight=0.0)
    assert scoring.face_quality(face(smile=0.0), s) == pytest.approx(1.0)
    s2 = Settings(smile_weight=0.15)
    assert scoring.face_quality(face(smile=1.0), s2) > scoring.face_quality(face(smile=0.0), s2)


def test_blurry_face_on_sharp_background_is_rejected():
    (d,) = decide(analysis(sharpness=1.0, faces=[face(sharpness=0.0, focus_log=-3.0)]))
    assert d.subject_sharpness == pytest.approx(0.35)
    assert d.status == PICK  # 0.35 is above the default blur threshold of 0.30
    (d,) = decide(analysis(sharpness=0.5, faces=[face(sharpness=0.0)]))
    assert d.status == REJECT
    assert BLURRY in d.flags


def test_exposure_flags():
    (over,) = decide(analysis(highlights=0.3, exposure=0.5))
    (under,) = decide(analysis(brightness=0.05, exposure=0.5))
    assert OVEREXPOSED in over.flags
    assert over.status == PICK
    assert UNDEREXPOSED in under.flags
    (crushed,) = decide(analysis(shadows=0.7))
    assert UNDEREXPOSED in crushed.flags


def test_stars_for():
    assert scoring.stars_for(0.95) == 5
    assert scoring.stars_for(0.85) == 5
    assert scoring.stars_for(0.75) == 4
    assert scoring.stars_for(0.5) == 3


def test_reject_styles():
    assert scoring.reject_rating(Settings(reject_style=RejectStyle.REJECT)) == -1
    assert scoring.reject_rating(Settings(reject_style=RejectStyle.ONE_STAR)) == 1
    assert scoring.reject_rating(Settings(reject_style=RejectStyle.NONE)) is None
    (d,) = decide(
        analysis(sharpness=0.0, log=-3.5), settings=Settings(reject_style=RejectStyle.NONE)
    )
    assert d.status == REJECT
    assert d.rating is None


def test_group_pick_and_alternates():
    shots = decide(analysis(), analysis(exposure=0.95), analysis(exposure=0.4, highlights=0.3))
    assert [d.status for d in shots] == [PICK, ALTERNATE, ALTERNATE]
    assert shots[0].rating == 5
    assert shots[1].rating == 2  # within 0.10 of the pick
    assert shots[2].rating == 1  # clearly worse
    assert DUPLICATE in shots[1].flags
    assert shots[1].summary == "similar to a better frame"
    assert [d.rank for d in shots] == [1, 2, 3]


def test_picks_per_group():
    shots = decide(
        analysis(), analysis(), analysis(exposure=0.8), settings=Settings(picks_per_group=2)
    )
    assert [d.status for d in shots] == [PICK, PICK, ALTERNATE]


def test_rejected_frames_are_never_picked():
    shots = decide(analysis(faces=[face(closed=True)]), analysis(faces=[face()], exposure=0.6))
    assert shots[0].status == REJECT
    assert shots[1].status == PICK


def test_all_rejected_group_has_no_pick():
    shots = decide(analysis(sharpness=0.1, log=-2.9), analysis(sharpness=0.05, log=-3.0))
    assert all(d.status == REJECT for d in shots)


def test_relative_softness_in_a_burst():
    sharp = analysis(log=-1.0)
    soft = analysis(log=-1.6)  # 0.6 log drop: soft, still usable
    blurred = analysis(log=-2.0)  # 1.0 log drop: rejected
    shots = decide(sharp, soft, blurred)
    assert shots[0].status == PICK
    assert SOFT in shots[1].flags
    assert shots[1].status == ALTERNATE
    assert BLURRY in shots[2].flags
    assert shots[2].status == REJECT
    assert shots[0].score > shots[1].score > shots[2].score


def test_relative_drop_needs_a_crisp_reference():
    shots = decide(analysis(sharpness=0.45, log=-2.3), analysis(sharpness=0.31, log=-3.2))
    # the sharpest frame is itself soft, so the weaker one is only "soft"
    assert SOFT in shots[1].flags
    assert BLURRY not in shots[1].flags


def test_contrast_normalisation_for_exposure_brackets():
    normal = analysis(log=-1.0, contrast=0.3)
    dark = analysis(log=-1.5, contrast=0.17)  # less detail energy only because it is darker
    shots = decide(normal, dark)
    assert SOFT not in shots[1].flags


def test_unreadable_shot():
    shots = decide(analysis(error="cannot decode"), analysis())
    assert shots[0].status == REJECT
    assert UNREADABLE in shots[0].flags
    assert shots[1].status == PICK
    assert shots[0].to_dict()["flags"] == [UNREADABLE]


def test_decision_to_dict():
    (d,) = decide(analysis(faces=[face()]))
    data = d.to_dict()
    assert data["status"] == PICK
    assert data["faces"] == 1
    assert 0 <= data["score"] <= 1
    assert data["face_score"] is not None
    assert d.summary == "looks good"


def test_multiple_groups():
    shots = decide(analysis(), analysis(), groups=[[0], [1]])
    assert [d.group for d in shots] == [0, 1]
    assert all(d.status == PICK for d in shots)
    assert shots[0].summary == "looks good"

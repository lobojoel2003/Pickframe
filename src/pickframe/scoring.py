"""Turn measurements into scores, flags, picks and star ratings.

Per shot
    * **subject sharpness**: focus around the eyes of the main faces (blended
      with whole-frame focus), or whole-frame focus when there are no faces.
    * **face score**: eyes open, eye focus, facing the camera, smiling.
    * **score**: subject sharpness, exposure and face score combined (0-1).

Per group of similar shots
    * the best ``picks_per_group`` shots without a hard flag are **picks**
      (3-5 stars by score);
    * other usable shots are **alternates** (2 stars if close to the pick,
      otherwise 1 star);
    * blurry shots, closed eyes and unreadable files are **rejects**.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .analysis.faces import Face
from .config import RejectStyle, Settings
from .results import (
    ALTERNATE,
    BLURRY,
    DUPLICATE,
    EYES_CLOSED,
    HARD_FLAGS,
    OVEREXPOSED,
    PICK,
    REJECT,
    SOFT,
    UNDEREXPOSED,
    UNREADABLE,
    Analysis,
    Decision,
)

#: Faces smaller than this share of the largest face are treated as background.
PROMINENT_SHARE = 0.2
#: Faces narrower than this fraction of the image width are ignored.
MIN_FACE_WIDTH = 0.02
#: Detail-energy drop (log10) versus the sharpest frame of the group that makes a
#: frame "soft" (about 2.8x less fine detail) ...
SOFT_DROP = 0.45
#: ... and the drop that rejects it as blurry, when the sharpest frame is crisp.
BLURRY_DROP = 0.8
#: Eye-state estimates get less reliable below this face width (working pixels).
SMALL_FACE_PIXELS = 80.0

STAR_THRESHOLDS = ((0.85, 5), (0.70, 4))  # otherwise 3 stars


def prominent_faces(faces: Sequence[Face]) -> list[Face]:
    candidates = [f for f in faces if f.box[2] >= MIN_FACE_WIDTH]
    if not candidates:
        return []
    largest = max(f.area for f in candidates)
    return [f for f in candidates if f.area >= PROMINENT_SHARE * largest]


def eyes_threshold(face: Face, base: float) -> float:
    """Small faces need stronger evidence before they count as blinking."""
    if face.pixels <= 0:
        return base
    shortfall = min(1.0, max(0.0, (SMALL_FACE_PIXELS - face.pixels) / 40.0))
    return min(0.95, base + 0.15 * shortfall)


def face_closed(face: Face, settings: Settings) -> bool:
    if face.frontal < 0.3:  # profiles: one eye is hidden, the estimate is unreliable
        return False
    return face.eyes_closed_probability >= eyes_threshold(face, settings.eyes_threshold)


def face_quality(face: Face, settings: Settings) -> float:
    eyes_open = 1.0 - min(1.0, face.eyes_closed_probability)
    total = 0.45 * eyes_open + 0.30 * face.sharpness + 0.10 * face.frontal
    total += settings.smile_weight * face.smile
    return total / (0.85 + settings.smile_weight)


def score_shot(analysis: Analysis, settings: Settings) -> Decision:
    """Score one shot in isolation (group fields are filled in later)."""
    decision = Decision(
        shot_id=-1, group=-1, group_size=1, rank=1, score=0.0, status=ALTERNATE, rating=None
    )
    if analysis.error:
        decision.flags.append(UNREADABLE)
        return decision

    faces = prominent_faces(analysis.faces)
    decision.faces = len(faces)
    subject = analysis.sharpness
    global_log = math.log10(analysis.sharpness_raw) if analysis.sharpness_raw > 0 else -6.0
    decision.subject_log = global_log
    if faces:
        qualities = [face_quality(f, settings) for f in faces]
        decision.face_score = 0.6 * min(qualities) + 0.4 * sum(qualities) / len(qualities)
        face_sharp = sum(f.sharpness for f in faces) / len(faces)
        subject = 0.35 * analysis.sharpness + 0.65 * face_sharp
        face_log = sum(f.focus_log for f in faces) / len(faces)
        decision.subject_log = 0.35 * global_log + 0.65 * face_log
        decision.eyes_closed = sum(face_closed(f, settings) for f in faces)
        score = 0.35 * subject + 0.15 * analysis.exposure + 0.50 * decision.face_score
    else:
        score = 0.70 * subject + 0.30 * analysis.exposure
    if analysis.contrast > 0:  # brightness-independent: detail energy scales with contrast²
        decision.subject_log -= 2.0 * math.log10(analysis.contrast)
    decision.subject_sharpness = subject

    if subject < settings.blur_threshold:
        decision.flags.append(BLURRY)
        score *= 0.6
    if decision.eyes_closed:
        decision.flags.append(EYES_CLOSED)
        score *= 0.6
    if analysis.highlights > 0.15 or analysis.brightness > 0.85:
        decision.flags.append(OVEREXPOSED)
    elif analysis.brightness < 0.1 or analysis.shadows > 0.6:
        decision.flags.append(UNDEREXPOSED)
    decision.score = float(min(1.0, max(0.0, score)))
    return decision


def stars_for(score: float) -> int:
    for threshold, stars in STAR_THRESHOLDS:
        if score >= threshold:
            return stars
    return 3


def reject_rating(settings: Settings) -> int | None:
    if settings.reject_style == RejectStyle.REJECT:
        return -1
    if settings.reject_style == RejectStyle.ONE_STAR:
        return 1
    return None


def decide(
    groups: Sequence[Sequence[int]],
    analyses: Mapping[int, Analysis],
    settings: Settings,
) -> dict[int, Decision]:
    """Score every shot, then choose picks, alternates and rejects per group."""
    decisions: dict[int, Decision] = {}
    for group_index, members in enumerate(groups):
        scored = []
        for shot_id in members:
            d = score_shot(analyses[shot_id], settings)
            d.shot_id, d.group, d.group_size = shot_id, group_index, len(members)
            scored.append(d)

        readable = [d for d in scored if UNREADABLE not in d.flags]
        if len(readable) > 1:
            # Frames of one burst share their content, so differences in fine
            # detail are differences in focus or camera shake.
            best = max(readable, key=lambda d: d.subject_log)
            for d in readable:
                drop = best.subject_log - d.subject_log
                d.score *= 1.0 - 0.25 * min(1.0, max(0.0, drop))
                if BLURRY in d.flags or drop < SOFT_DROP:
                    continue
                if drop >= BLURRY_DROP and best.subject_sharpness >= 0.5:
                    d.flags.append(BLURRY)
                    d.score *= 0.6
                else:
                    d.flags.append(SOFT)

        ranked = sorted(scored, key=lambda d: (-d.score, d.shot_id))
        for rank, d in enumerate(ranked, start=1):
            d.rank = rank
        picks = [d for d in ranked if not HARD_FLAGS.intersection(d.flags)][
            : settings.picks_per_group
        ]
        best_score = picks[0].score if picks else 0.0
        for d in scored:
            if HARD_FLAGS.intersection(d.flags):
                d.status = REJECT
                d.rating = reject_rating(settings)
            elif d in picks:
                d.status = PICK
                d.rating = stars_for(d.score)
            else:
                d.status = ALTERNATE
                d.flags.append(DUPLICATE)
                d.rating = 2 if d.score >= best_score - 0.10 else 1
            decisions[d.shot_id] = d
    return decisions

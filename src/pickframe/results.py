"""Data produced by an analysis run."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

from .analysis.faces import Face


@dataclass
class Analysis:
    """Measurements for one shot (independent of the other shots)."""

    path: str
    width: int = 0
    height: int = 0
    capture_time: float | None = None
    camera: str = ""
    sharpness: float = 0.0  # whole-frame focus score, 0..1
    sharpness_raw: float = 0.0  # Laplacian energy behind the score
    contrast: float = 0.0  # luminance standard deviation, 0..1
    exposure: float = 0.0  # 0..1
    brightness: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    faces: list[Face] = field(default_factory=list)
    embedding: np.ndarray | None = None
    embedder: str = ""
    source: str = ""
    error: str | None = None

    def to_dict(self, with_embedding: bool = True) -> dict:
        data = {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "capture_time": self.capture_time,
            "camera": self.camera,
            "sharpness": round(self.sharpness, 4),
            "sharpness_raw": self.sharpness_raw,
            "contrast": round(self.contrast, 5),
            "exposure": round(self.exposure, 4),
            "brightness": round(self.brightness, 4),
            "highlights": round(self.highlights, 4),
            "shadows": round(self.shadows, 4),
            "faces": [f.to_dict() for f in self.faces],
            "embedder": self.embedder,
            "source": self.source,
            "error": self.error,
        }
        if with_embedding and self.embedding is not None:
            data["embedding"] = base64.b64encode(self.embedding.astype(np.float32).tobytes()).decode()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Analysis:
        embedding = None
        if data.get("embedding"):
            embedding = np.frombuffer(base64.b64decode(data["embedding"]), dtype=np.float32).copy()
        return cls(
            path=data["path"],
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            capture_time=data.get("capture_time"),
            camera=data.get("camera", ""),
            sharpness=float(data.get("sharpness", 0.0)),
            sharpness_raw=float(data.get("sharpness_raw", 0.0)),
            contrast=float(data.get("contrast", 0.0)),
            exposure=float(data.get("exposure", 0.0)),
            brightness=float(data.get("brightness", 0.0)),
            highlights=float(data.get("highlights", 0.0)),
            shadows=float(data.get("shadows", 0.0)),
            faces=[Face.from_dict(f) for f in data.get("faces", [])],
            embedding=embedding,
            embedder=data.get("embedder", ""),
            source=data.get("source", ""),
            error=data.get("error"),
        )


# Status values of a decision.
PICK = "pick"
ALTERNATE = "alternate"
REJECT = "reject"

# Flags. "Hard" flags reject a shot; the others only explain the ranking.
BLURRY = "blurry"
EYES_CLOSED = "eyes-closed"
UNREADABLE = "unreadable"
SOFT = "soft"
OVEREXPOSED = "overexposed"
UNDEREXPOSED = "underexposed"
DUPLICATE = "duplicate"
HARD_FLAGS = frozenset({BLURRY, EYES_CLOSED, UNREADABLE})

FLAG_TEXT = {
    BLURRY: "blurry",
    EYES_CLOSED: "eyes closed",
    UNREADABLE: "could not be read",
    SOFT: "softer than the sharpest frame",
    OVEREXPOSED: "overexposed",
    UNDEREXPOSED: "underexposed",
    DUPLICATE: "similar to a better frame",
}


@dataclass
class Decision:
    """Pickframe's verdict for one shot."""

    shot_id: int
    group: int
    group_size: int
    rank: int  # 1 = best in its group
    score: float  # overall quality, 0..1
    status: str  # pick / alternate / reject
    rating: int | None  # XMP rating to write (-1..5), None = leave unrated
    flags: list[str] = field(default_factory=list)
    subject_sharpness: float = 0.0
    subject_log: float = -6.0  # log10 detail energy of the subject, for comparing frames
    face_score: float | None = None
    eyes_closed: int = 0  # number of prominent faces with closed eyes
    faces: int = 0  # number of prominent faces

    @property
    def summary(self) -> str:
        if not self.flags:
            return "best of its group" if self.status == PICK and self.group_size > 1 else "looks good"
        return ", ".join(FLAG_TEXT.get(flag, flag) for flag in self.flags)

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "group_size": self.group_size,
            "rank": self.rank,
            "score": round(self.score, 4),
            "status": self.status,
            "rating": self.rating,
            "flags": list(self.flags),
            "subject_sharpness": round(self.subject_sharpness, 4),
            "face_score": None if self.face_score is None else round(self.face_score, 4),
            "eyes_closed": self.eyes_closed,
            "faces": self.faces,
        }

"""User-adjustable settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path


class RejectStyle(str, Enum):
    """How rejected shots are written to XMP."""

    REJECT = "reject"  # xmp:Rating -1 + Lightroom/digiKam reject flags
    ONE_STAR = "one-star"  # xmp:Rating 1 (for apps that ignore -1)
    NONE = "none"  # leave rejected shots unrated


class SidecarStyle(str, Enum):
    """Which XMP sidecar names to write."""

    AUTO = "auto"  # both styles below (raw files) / darktable style (other files)
    DARKTABLE = "darktable"  # IMG_0001.CR3.xmp: darktable, digiKam (default naming)
    LIGHTROOM = "lightroom"  # IMG_0001.xmp: Lightroom, Capture One, Bridge (raw files)


@dataclass
class Settings:
    # --- analysis -----------------------------------------------------------
    #: Long side of the working image in pixels.
    analysis_size: int = 1600
    #: Long side used for face detection (larger finds smaller faces, slower).
    face_detect_size: int = 1280
    #: Face detector confidence threshold.
    face_confidence: float = 0.6
    #: ``auto`` (CLIP if downloaded, else classic), ``clip`` or ``classic``.
    embedder: str = "auto"
    #: Worker threads (0 = automatic).
    workers: int = 0

    # --- grouping -----------------------------------------------------------
    #: Similarity needed to join a group (``None`` = embedder default).
    similarity: float | None = None
    #: Maximum time between consecutive shots of one group, in seconds.
    max_gap: float = 60.0

    # --- decisions ----------------------------------------------------------
    #: Sharpness score (0-1) below which a shot is flagged as blurry.
    blur_threshold: float = 0.30
    #: Eyes-closed probability (0-1) above which a face counts as blinking.
    eyes_threshold: float = 0.55
    #: How many shots to pick from each group.
    picks_per_group: int = 1
    #: Weight of smiles in the face score (0 disables).
    smile_weight: float = 0.15
    #: Write rejects as ``reject``, ``one-star`` or ``none``.
    reject_style: RejectStyle = RejectStyle.REJECT

    # --- output -------------------------------------------------------------
    sidecar_style: SidecarStyle = SidecarStyle.AUTO
    #: Also write pick/reject flags (Lightroom 13.2+ and digiKam).
    write_flags: bool = True
    #: Colour label for picks (e.g. ``Green``); empty = none.
    pick_label: str = ""
    #: Overwrite ratings that were set by hand or another program.
    overwrite_manual: bool = False

    def analysis_key(self) -> str:
        """Settings that change per-image analysis results (used by the cache)."""
        return f"s{self.analysis_size}-f{self.face_detect_size}-c{self.face_confidence:.2f}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reject_style"] = self.reject_style.value
        data["sidecar_style"] = self.sidecar_style.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        if "reject_style" in values:
            values["reject_style"] = RejectStyle(values["reject_style"])
        if "sidecar_style" in values:
            values["sidecar_style"] = SidecarStyle(values["sidecar_style"])
        return cls(**values)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Settings:
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()

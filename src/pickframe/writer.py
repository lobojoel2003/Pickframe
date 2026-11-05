"""Write a culling result to XMP, or undo Pickframe's changes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .config import RejectStyle, SidecarStyle
from .discovery import iter_images
from .formats import is_jpeg
from .pipeline import CullResult
from .results import PICK, REJECT
from .xmp import (
    WriteReport,
    XmpError,
    XmpUpdate,
    reset_jpeg,
    reset_sidecar,
    sidecar_paths,
    update_jpeg,
    update_sidecar,
)

ProgressFn = Callable[[int, int, str], None]


def plan(result: CullResult) -> dict[int, XmpUpdate]:
    """The XMP update for every shot, including manual overrides."""
    settings = result.settings
    updates: dict[int, XmpUpdate] = {}
    for shot in result.shots:
        decision = result.decisions[shot.id]
        rating = result.rating(shot.id)
        if shot.id in result.overrides:
            flag = "reject" if rating == -1 else ("pick" if decision.status == PICK and rating else "none")
        elif decision.status == PICK:
            flag = "pick"
        elif decision.status == REJECT and settings.reject_style != RejectStyle.NONE:
            flag = "reject"
        else:
            flag = "none"
        if not settings.write_flags:
            flag = "none"
        updates[shot.id] = XmpUpdate(
            rating=rating,
            flag=flag,
            label=settings.pick_label if flag == "pick" else "",
            score=decision.score,
            group=decision.group + 1,
            status=decision.status,
            notes=decision.flags,
        )
    return updates


def write_result(
    result: CullResult,
    dry_run: bool = False,
    embed_jpeg: bool = False,
    progress: ProgressFn | None = None,
) -> WriteReport:
    """Write ratings for every file of every shot."""
    settings = result.settings
    report = WriteReport()
    updates = plan(result)
    total = len(result.shots)
    for index, shot in enumerate(result.shots, start=1):
        update = updates[shot.id]
        for image in shot.files:
            targets: list[tuple[str, Path]] = [("sidecar", p) for p in sidecar_paths(image, settings.sidecar_style)]
            if embed_jpeg and is_jpeg(image):
                targets.append(("jpeg", image))
            for kind, target in targets:
                try:
                    if kind == "jpeg":
                        outcome = update_jpeg(target, update, settings.overwrite_manual, dry_run)
                    else:
                        outcome = update_sidecar(target, update, settings.overwrite_manual, dry_run)
                except (OSError, XmpError) as exc:
                    report.errors.append((target, str(exc)))
                    continue
                if outcome == "written":
                    report.written.append(target)
                elif outcome == "manual":
                    report.skipped_manual.append(target)
                else:
                    report.unchanged.append(target)
        if progress:
            progress(index, total, shot.name)
    return report


def reset(
    paths: Iterable[Path | str],
    recursive: bool = False,
    include_jpeg: bool = True,
    dry_run: bool = False,
) -> WriteReport:
    """Remove ratings and flags written by Pickframe, restoring earlier ratings."""
    report = WriteReport()
    seen: set[Path] = set()  # IMG_0001.xmp is shared by IMG_0001.CR3 and IMG_0001.JPG
    for image in iter_images(paths, recursive):
        candidates = set(sidecar_paths(image, SidecarStyle.AUTO)) | {image.with_suffix(".xmp")}
        for sidecar in sorted(candidates - seen):
            seen.add(sidecar)
            try:
                outcome = reset_sidecar(sidecar, dry_run)
            except (OSError, XmpError) as exc:
                report.errors.append((sidecar, str(exc)))
                continue
            if outcome == "removed":
                report.removed.append(sidecar)
            elif outcome == "reset":
                report.written.append(sidecar)
        if include_jpeg and is_jpeg(image):
            try:
                if reset_jpeg(image, dry_run) == "reset":
                    report.written.append(image)
            except (OSError, XmpError) as exc:
                report.errors.append((image, str(exc)))
    return report

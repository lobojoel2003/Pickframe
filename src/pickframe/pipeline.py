"""Run a full culling pass: discover, analyse, group, score."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .analysis.embed import Embedder, create_embedder
from .analysis.faces import FaceAnalyzer
from .analysis.quality import contrast, exposure, sharpness_raw, sharpness_score
from .cache import AnalysisCache
from .config import Settings
from .discovery import Shot, discover
from .grouping import GroupItem, group_shots
from .imaging import ImageLoadError, load_image
from .results import Analysis, Decision
from .scoring import decide

#: Bump when measurements change so cached results are recomputed.
ANALYSIS_VERSION = "3"

ProgressFn = Callable[[int, int, str], None]


class Cancelled(Exception):
    """Raised when a run is cancelled through its ``cancel`` event."""


@dataclass
class CullResult:
    shots: list[Shot]
    analyses: dict[int, Analysis]
    groups: list[list[int]]
    decisions: dict[int, Decision]
    settings: Settings
    embedder: str = ""
    elapsed: float = 0.0
    cached: int = 0
    #: Ratings chosen by hand in the desktop app (shot id -> rating or None).
    overrides: dict[int, int | None] = field(default_factory=dict)

    def rating(self, shot_id: int) -> int | None:
        if shot_id in self.overrides:
            return self.overrides[shot_id]
        return self.decisions[shot_id].rating

    def regroup(self, settings: Settings | None = None) -> None:
        """Recompute groups and decisions after a settings change (no re-analysis)."""
        if settings is not None:
            self.settings = settings
        threshold = self.settings.similarity
        if threshold is None:
            threshold = _default_similarity(self.embedder)
        items = [
            GroupItem(
                shot_id=shot.id,
                folder=str(shot.primary.parent),
                name=shot.primary.name,
                capture_time=self.analyses[shot.id].capture_time,
                embedding=self.analyses[shot.id].embedding,
            )
            for shot in self.shots
        ]
        self.groups = group_shots(items, threshold, self.settings.max_gap)
        self.decisions = decide(self.groups, self.analyses, self.settings)

    def counts(self) -> dict[str, int]:
        out = {"shots": len(self.shots), "groups": len(self.groups), "pick": 0, "alternate": 0, "reject": 0}
        for decision in self.decisions.values():
            out[decision.status] += 1
        return out

    def to_dict(self, with_embeddings: bool = True) -> dict:
        return {
            "pickframe": __version__,
            "embedder": self.embedder,
            "elapsed": round(self.elapsed, 2),
            "settings": self.settings.to_dict(),
            "groups": self.groups,
            "shots": [
                {
                    "id": shot.id,
                    "files": [str(p) for p in shot.files],
                    "analysis": self.analyses[shot.id].to_dict(with_embedding=with_embeddings),
                    "decision": self.decisions[shot.id].to_dict(),
                    "override": _encode_override(self.overrides, shot.id),
                }
                for shot in self.shots
            ],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CullResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        shots, analyses, overrides = [], {}, {}
        for entry in data["shots"]:
            shot = Shot(id=int(entry["id"]), files=[Path(p) for p in entry["files"]])
            shots.append(shot)
            analyses[shot.id] = Analysis.from_dict(entry["analysis"])
            if entry.get("override") is not None:
                value = entry["override"]
                overrides[shot.id] = None if value == "none" else int(value)
        result = cls(
            shots=shots,
            analyses=analyses,
            groups=[],
            decisions={},
            settings=Settings.from_dict(data.get("settings", {})),
            embedder=data.get("embedder", ""),
            elapsed=float(data.get("elapsed", 0.0)),
            overrides=overrides,
        )
        result.regroup()
        return result


def _encode_override(overrides: dict[int, int | None], shot_id: int) -> int | str | None:
    if shot_id not in overrides:
        return None
    value = overrides[shot_id]
    return "none" if value is None else value


def _default_similarity(embedder_name: str) -> float:
    from .analysis.embed import ClassicEmbedder, ClipEmbedder

    return ClipEmbedder.default_similarity if embedder_name == "clip" else ClassicEmbedder.default_similarity


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2)))


class Analyzer:
    """Analyse images. Models are loaded lazily and shared between threads."""

    def __init__(self, settings: Settings | None = None, use_cache: bool = True):
        self.settings = settings or Settings()
        self.workers = self.settings.workers or default_workers()
        self._threads = max(1, (os.cpu_count() or 2) // self.workers)
        self._faces: FaceAnalyzer | None = None
        self._embedder: Embedder | None = None
        self._init_lock = threading.Lock()
        self._use_cache = use_cache
        self._cache: AnalysisCache | None = None

    @property
    def faces(self) -> FaceAnalyzer:
        with self._init_lock:
            if self._faces is None:
                self._faces = FaceAnalyzer(
                    detect_size=self.settings.face_detect_size,
                    min_confidence=self.settings.face_confidence,
                    threads=self._threads,
                )
            return self._faces

    @property
    def embedder(self) -> Embedder:
        with self._init_lock:
            if self._embedder is None:
                self._embedder = create_embedder(self.settings.embedder, self._threads)
            return self._embedder

    @property
    def cache(self) -> AnalysisCache | None:
        if self._use_cache and self._cache is None:
            try:
                self._cache = AnalysisCache()
            except Exception:  # read-only home, locked database, ...: run without cache
                self._use_cache = False
        return self._cache

    def cache_key(self) -> str:
        return f"v{ANALYSIS_VERSION}|{self.settings.analysis_key()}|{self.embedder.name}"

    def analyze_image(self, path: Path) -> Analysis:
        analysis = Analysis(path=str(path), embedder=self.embedder.name)
        try:
            loaded = load_image(path, self.settings.analysis_size)
        except ImageLoadError as exc:
            analysis.error = str(exc)
            return analysis
        except Exception as exc:  # never let one bad file stop a run
            analysis.error = f"{path.name}: {exc}"
            return analysis
        rgb = loaded.rgb
        analysis.width, analysis.height = loaded.width, loaded.height
        analysis.capture_time = loaded.exif.capture_time
        analysis.camera = loaded.exif.camera
        analysis.source = loaded.source
        analysis.sharpness_raw = sharpness_raw(rgb)
        analysis.sharpness = sharpness_score(analysis.sharpness_raw)
        analysis.contrast = contrast(rgb)
        exp = exposure(rgb)
        analysis.exposure = exp.score
        analysis.brightness, analysis.highlights, analysis.shadows = exp.brightness, exp.highlights, exp.shadows
        try:
            analysis.faces = self.faces.analyze(rgb)
        except Exception as exc:
            analysis.error = f"face analysis failed: {exc}"
            return analysis
        analysis.embedding = self.embedder.embed(rgb)
        return analysis

    def analyze(
        self,
        shots: list[Shot],
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
    ) -> tuple[dict[int, Analysis], int]:
        """Analyse every shot's primary file; returns results and the cache hit count."""
        results: dict[int, Analysis] = {}
        todo: list[Shot] = []
        cache = self.cache
        key = self.cache_key()
        for shot in shots:
            hit = cache.get(shot.primary, key) if cache else None
            if hit is not None:
                results[shot.id] = hit
            else:
                todo.append(shot)
        cached = len(results)
        total = len(shots)
        if progress:
            progress(cached, total, "cached" if cached else "starting")
        if todo:
            # Load the models before starting threads so failures surface once.
            _ = self.faces, self.embedder
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="pickframe") as pool:
                futures = {pool.submit(self.analyze_image, shot.primary): shot for shot in todo}
                try:
                    for future in as_completed(futures):
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        shot = futures[future]
                        analysis = future.result()
                        results[shot.id] = analysis
                        if cache:
                            cache.put(shot.primary, key, analysis)
                        if progress:
                            progress(len(results), total, shot.primary.name)
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise
        return results, cached

    def run(
        self,
        paths: Iterable[Path | str],
        recursive: bool = False,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
    ) -> CullResult:
        start = time.perf_counter()
        shots = discover(paths, recursive=recursive)
        analyses, cached = self.analyze(shots, progress, cancel)
        result = CullResult(
            shots=shots,
            analyses=analyses,
            groups=[],
            decisions={},
            settings=self.settings,
            embedder=self.embedder.name if shots else "",
            cached=cached,
        )
        result.regroup()
        result.elapsed = time.perf_counter() - start
        return result

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

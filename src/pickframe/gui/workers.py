"""Background work for the desktop app: analysis, writing, thumbnails, downloads."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThread, Signal
from PySide6.QtGui import QImage

from .. import models
from ..config import Settings
from ..imaging import load_thumbnail
from ..pipeline import Analyzer, Cancelled, CullResult
from ..writer import write_result


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class AnalysisJob(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)  # CullResult
    failed = Signal(str)

    def __init__(self, paths: list[Path], settings: Settings, recursive: bool):
        super().__init__()
        self.paths = paths
        self.settings = settings
        self.recursive = recursive
        self.cancel = threading.Event()

    def run(self) -> None:
        analyzer = Analyzer(self.settings)
        try:
            result = analyzer.run(
                self.paths, recursive=self.recursive, progress=self.progress.emit, cancel=self.cancel
            )
        except Cancelled:
            self.failed.emit("Analysis cancelled.")
        except Exception as exc:
            self.failed.emit(f"Analysis failed: {exc}")
        else:
            self.finished.emit(result)
        finally:
            analyzer.close()


class WriteJob(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)  # WriteReport
    failed = Signal(str)

    def __init__(self, result: CullResult, embed_jpeg: bool):
        super().__init__()
        self.result = result
        self.embed_jpeg = embed_jpeg

    def run(self) -> None:
        try:
            report = write_result(self.result, embed_jpeg=self.embed_jpeg, progress=self.progress.emit)
        except Exception as exc:
            self.failed.emit(f"Writing XMP failed: {exc}")
        else:
            self.finished.emit(report)


class DownloadJob(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            path = models.download(
                self.name, lambda done, total: self.progress.emit(done, total or 0), cancel=self.cancel
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(str(path))


def start(job: QObject, parent: QObject) -> QThread:
    """Run ``job.run`` on a new thread that cleans itself up."""
    thread = QThread(parent)
    job.moveToThread(thread)
    thread.started.connect(job.run)  # type: ignore[attr-defined]
    for signal_name in ("finished", "failed"):
        getattr(job, signal_name).connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread


class ThumbnailSignals(QObject):
    loaded = Signal(int, QImage)


class ThumbnailTask(QRunnable):
    def __init__(self, shot_id: int, path: Path, size: int, signals: ThumbnailSignals):
        super().__init__()
        self.shot_id = shot_id
        self.path = path
        self.size = size
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            image = to_qimage(load_thumbnail(self.path, self.size))
        except Exception:
            image = QImage()
        self.signals.loaded.emit(self.shot_id, image)

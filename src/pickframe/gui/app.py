"""Pickframe desktop app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, QSize, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..config import Settings
from ..pipeline import CullResult
from ..version import version_string
from . import theme
from .widgets import THUMB, DetailPanel, SettingsDialog, ShotDelegate, ShotModel, ShotRole
from .workers import AnalysisJob, DownloadJob, ThumbnailSignals, ThumbnailTask, WriteJob, start

REPO_URL = "https://github.com/lobojoel2003/Pickframe"
ICON_PATH = Path(__file__).resolve().parent / "icon.png"
FILTERS = ("All photos", "Picks", "Alternates", "Rejects")
GROUP_ICON = QSize(56, 42)


def group_icon(pixmap: QPixmap | None) -> QIcon:
    """Thumbnail centred on a fixed-size canvas so group names line up."""
    canvas = QPixmap(GROUP_ICON)
    canvas.fill(QColor(theme.CARD))
    if pixmap is not None and not pixmap.isNull():
        scaled = pixmap.scaled(GROUP_ICON, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(canvas)
        painter.drawPixmap((GROUP_ICON.width() - scaled.width()) // 2, (GROUP_ICON.height() - scaled.height()) // 2, scaled)
        painter.end()
    return QIcon(canvas)


class WelcomePage(QWidget):
    def __init__(self, open_folder) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch(2)
        logo = QLabel()
        if ICON_PATH.is_file():
            logo.setPixmap(QPixmap(str(ICON_PATH)).scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio,
                                                          Qt.TransformationMode.SmoothTransformation))  # fmt: skip
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Pickframe")
        title.setObjectName("hero")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QLabel(
            "Free, local AI photo culling.<br>Groups near-duplicates, flags blur and closed eyes, "
            "and writes star ratings<br>that Lightroom, darktable and digiKam can read."
            "<br><br>Drop a folder here or open one to start. Nothing leaves your computer."
        )
        text.setObjectName("muted")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button = QPushButton("Open folder…")
        button.setDefault(True)
        button.setFixedWidth(200)
        button.clicked.connect(open_folder)
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addSpacing(12)
        layout.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(3)


class MainWindow(QMainWindow):
    def __init__(self, store: QSettings | None = None) -> None:
        super().__init__()
        self.store = store or QSettings("Pickframe", "Pickframe")
        self.settings = Settings.from_dict(json.loads(str(self.store.value("settings", "{}") or "{}")))
        self.embed_jpeg = self.store.value("embed_jpeg", False, type=bool)
        self.folder: Path | None = None
        self.result: CullResult | None = None
        self.job = None
        self.thread = None
        self.requested: set[int] = set()
        self.previews: dict[int, QPixmap] = {}
        self.pool = QThreadPool.globalInstance()
        self.thumb_signals = ThumbnailSignals()
        self.thumb_signals.loaded.connect(self._thumb_loaded)
        self.preview_signals = ThumbnailSignals()
        self.preview_signals.loaded.connect(self._preview_loaded)
        self.unsaved = False

        self.setWindowTitle("Pickframe")
        if ICON_PATH.is_file():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setAcceptDrops(True)
        self.resize(1440, 900)
        self._build_toolbar()
        self._build_views()
        self._build_statusbar()
        self._build_shortcuts()
        geometry = self.store.value("geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        self._update_actions()

    # -- construction -----------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QSize(18, 18))
        self.addToolBar(bar)
        self.act_open = QAction("Open folder…", self, shortcut=QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.choose_folder)
        self.act_analyse = QAction("Analyse", self, shortcut=QKeySequence("F5"))
        self.act_analyse.triggered.connect(self.analyse)
        self.act_write = QAction("Write XMP ratings", self, shortcut=QKeySequence.StandardKey.Save)
        self.act_write.triggered.connect(self.write_xmp)
        self.act_settings = QAction("Settings", self, shortcut=QKeySequence("Ctrl+,"))
        self.act_settings.triggered.connect(self.open_settings)
        self.act_report = QAction("Export report…", self, shortcut=QKeySequence("Ctrl+E"))
        self.act_report.triggered.connect(self.export_report)
        self.act_about = QAction("About", self)
        self.act_about.triggered.connect(self.about)
        bar.addAction(self.act_open)
        self.recursive = QCheckBox("Include sub-folders")
        self.recursive.setChecked(self.store.value("recursive", False, type=bool))
        bar.addWidget(self.recursive)
        bar.addSeparator()
        bar.addAction(self.act_analyse)
        bar.addAction(self.act_write)
        write_button = bar.widgetForAction(self.act_write)
        if isinstance(write_button, QToolButton):
            write_button.setObjectName("primary")
        bar.addSeparator()
        self.filter = QComboBox()
        self.filter.addItems(FILTERS)
        self.filter.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(QLabel(" Show "))
        bar.addWidget(self.filter)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy())
        bar.addWidget(spacer)
        bar.addAction(self.act_report)
        bar.addAction(self.act_settings)
        bar.addAction(self.act_about)

    def _build_views(self) -> None:
        self.model = ShotModel()
        self.grid = QListView()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setSpacing(4)
        self.grid.setModel(self.model)
        self.grid.setItemDelegate(ShotDelegate(self.model, self._request_thumb))
        self.grid.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.grid.selectionModel().currentChanged.connect(lambda *_: self._show_current())

        self.stack = QStackedWidget()
        self.stack.addWidget(WelcomePage(self.choose_folder))
        self.stack.addWidget(self.grid)
        self.setCentralWidget(self.stack)

        self.groups = QListWidget()
        self.groups.setIconSize(GROUP_ICON)
        self.groups.currentRowChanged.connect(lambda _row: self._apply_filter())
        left = QDockWidget("Groups", self)
        left.setWidget(self.groups)
        left.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        left.setMinimumWidth(270)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left)

        self.details = DetailPanel()
        self.details.rating_requested.connect(self.set_rating)
        right = QDockWidget("Details", self)
        right.setWidget(self.details)
        right.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        right.setMinimumWidth(400)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right)

    def _build_statusbar(self) -> None:
        self.status_label = QLabel("Open a folder to begin.")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(280)
        self.progress.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel)
        bar = self.statusBar()
        bar.addWidget(self.status_label, 1)
        bar.addPermanentWidget(self.progress)
        bar.addPermanentWidget(self.cancel_button)

    def _build_shortcuts(self) -> None:
        for key, value in (("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("X", -1), ("U", 99)):
            QShortcut(QKeySequence(key), self, activated=lambda v=value: self.set_rating(v))

    # -- helpers ------------------------------------------------------------------
    def _busy(self) -> bool:
        return self.thread is not None

    def _update_actions(self) -> None:
        busy = self._busy()
        self.act_open.setEnabled(not busy)
        self.act_analyse.setEnabled(not busy and self.folder is not None)
        self.act_write.setEnabled(not busy and self.result is not None and bool(self.result.shots))
        self.act_report.setEnabled(not busy and self.result is not None and bool(self.result.shots))
        self.act_settings.setEnabled(not busy)

    def _save_store(self) -> None:
        self.store.setValue("settings", json.dumps(self.settings.to_dict()))
        self.store.setValue("embed_jpeg", self.embed_jpeg)
        self.store.setValue("recursive", self.recursive.isChecked())
        self.store.setValue("geometry", self.saveGeometry())
        if self.folder:
            self.store.setValue("last_folder", str(self.folder))

    def _summary_text(self) -> str:
        if not self.result:
            return ""
        c = self.result.counts()
        return (
            f"{c['shots']} photos · {c['groups']} groups · {c['pick']} picks · "
            f"{c['alternate']} alternates · {c['reject']} rejects · grouping: {self.result.embedder}"
        )

    # -- folder & analysis ------------------------------------------------------
    def choose_folder(self) -> None:
        start_dir = str(self.folder or self.store.value("last_folder", str(Path.home())))
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder of photos", start_dir)
        if folder:
            self.open_folder(Path(folder))

    def open_folder(self, folder: Path, analyse: bool = True) -> None:
        self.folder = folder
        self.setWindowTitle(f"Pickframe — {folder.name}")
        self._save_store()
        self._update_actions()
        if analyse:
            self.analyse()

    def analyse(self) -> None:
        if self.folder is None or self._busy():
            return
        self.job = AnalysisJob([self.folder], self.settings, self.recursive.isChecked())
        self.job.progress.connect(self._progress)
        self.job.finished.connect(self._analysis_done)
        self.job.failed.connect(self._job_failed)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.status_label.setText(f"Analysing {self.folder}…")
        self.thread = start(self.job, self)
        self.thread.finished.connect(self._job_cleanup)
        self._update_actions()

    def cancel(self) -> None:
        if isinstance(self.job, AnalysisJob):
            self.job.cancel.set()
            self.status_label.setText("Cancelling…")

    def _progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(f"Analysing {done}/{total} · {name}")

    def _job_cleanup(self) -> None:
        self.thread = None
        self.job = None
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self._update_actions()

    def _job_failed(self, message: str) -> None:
        self.status_label.setText(message)
        if "cancelled" not in message.lower():
            QMessageBox.warning(self, "Pickframe", message)

    def _analysis_done(self, result: CullResult) -> None:
        self.set_result(result)
        if not result.shots:
            self.status_label.setText("No photos found in this folder. Try “Include sub-folders”.")

    def set_result(self, result: CullResult) -> None:
        self.result = result
        self.unsaved = bool(result.shots)
        self.requested.clear()
        self.previews.clear()
        self.model.set_result(result)
        self.stack.setCurrentIndex(1)
        self._fill_groups()
        self._apply_filter()
        self.status_label.setText(self._summary_text())
        self._update_actions()

    # -- views ----------------------------------------------------------------------
    def _fill_groups(self) -> None:
        self.groups.blockSignals(True)
        self.groups.clear()
        result = self.result
        if result is None:
            self.groups.blockSignals(False)
            return
        all_item = QListWidgetItem(f"All photos\n{len(result.shots)} photos in {len(result.groups)} groups")
        all_item.setData(Qt.ItemDataRole.UserRole, -1)
        all_item.setIcon(group_icon(None))
        self.groups.addItem(all_item)
        for index, members in enumerate(result.groups):
            picks = sum(result.decisions[i].status == "pick" for i in members)
            rejects = sum(result.decisions[i].status == "reject" for i in members)
            text = f"Group {index + 1}\n{len(members)} shot{'s' if len(members) != 1 else ''}"
            text += f" · {picks} pick" + (f" · {rejects} reject{'s' if rejects != 1 else ''}" if rejects else "")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, index)
            best = min(members, key=lambda i: result.decisions[i].rank)
            item.setIcon(group_icon(self.model.thumbs.get(best)))
            self.groups.addItem(item)
            self._request_thumb(best)
        self.groups.setCurrentRow(0)
        self.groups.blockSignals(False)

    def _apply_filter(self) -> None:
        result = self.result
        if result is None:
            return
        item = self.groups.currentItem()
        group = item.data(Qt.ItemDataRole.UserRole) if item is not None else -1
        ids = [i for members in result.groups for i in sorted(members, key=lambda s: result.decisions[s].rank)]
        if group is not None and group >= 0:
            ids = [i for i in ids if result.decisions[i].group == group]
        status = {1: "pick", 2: "alternate", 3: "reject"}.get(self.filter.currentIndex())
        if status:
            ids = [i for i in ids if result.decisions[i].status == status]
        current = self._current_id()
        self.model.set_ids(ids)
        row = self.model.row_of(current) if current is not None else -1
        self.grid.setCurrentIndex(self.model.index(row if row >= 0 else 0))
        self._show_current()

    def _current_id(self) -> int | None:
        index = self.grid.currentIndex()
        return index.data(ShotRole) if index.isValid() else None

    def _show_current(self) -> None:
        shot_id = self._current_id()
        if self.result is None or shot_id is None:
            self.details.show_shot(None, None, None)
            return
        pixmap = self.previews.get(shot_id)
        if pixmap is None:
            pixmap = self.model.thumbs.get(shot_id)
            self.pool.start(ThumbnailTask(shot_id, self.result.shots[shot_id].primary, 1400, self.preview_signals))
        self.details.show_shot(self.result, shot_id, pixmap)

    def _request_thumb(self, shot_id: int) -> None:
        if self.result is None or shot_id in self.requested:
            return
        self.requested.add(shot_id)
        self.pool.start(ThumbnailTask(shot_id, self.result.shots[shot_id].primary, THUMB * 2, self.thumb_signals))

    def _thumb_loaded(self, shot_id: int, image: QImage) -> None:
        self.model.thumbnail_ready(shot_id, image)
        if self.result is None or image.isNull():
            return
        for row in range(self.groups.count()):
            item = self.groups.item(row)
            group = item.data(Qt.ItemDataRole.UserRole)
            if group is None or group < 0 or group >= len(self.result.groups):
                continue
            members = self.result.groups[group]
            best = min(members, key=lambda i: self.result.decisions[i].rank)  # type: ignore[union-attr]
            if best == shot_id:
                item.setIcon(group_icon(QPixmap.fromImage(image)))
        if shot_id == self._current_id() and shot_id not in self.previews:
            self._show_current()

    def _preview_loaded(self, shot_id: int, image: QImage) -> None:
        if image.isNull():
            return
        if len(self.previews) > 24:
            self.previews.pop(next(iter(self.previews)))
        self.previews[shot_id] = QPixmap.fromImage(image)
        if shot_id == self._current_id():
            self._show_current()

    # -- editing ----------------------------------------------------------------------
    def set_rating(self, value: int) -> None:
        shot_id = self._current_id()
        if self.result is None or shot_id is None:
            return
        if value == 99:
            self.result.overrides.pop(shot_id, None)
        else:
            self.result.overrides[shot_id] = value or None
        self.unsaved = True
        self.model.refresh()
        self._show_current()
        if value != 99:  # move on, like a light table
            index = self.grid.currentIndex()
            nxt = self.model.index(min(index.row() + 1, self.model.rowCount() - 1))
            self.grid.setCurrentIndex(nxt)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self.embed_jpeg, self)
        dialog.download_requested.connect(lambda: self._download_clip(dialog))
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        new, self.embed_jpeg = dialog.values()
        needs_analysis = new.embedder != self.settings.embedder or new.analysis_key() != self.settings.analysis_key()
        self.settings = new
        self._save_store()
        if self.result is None:
            return
        if needs_analysis:
            self.analyse()
        else:
            self.result.regroup(new)
            self.set_result(self.result)

    def _download_clip(self, dialog: SettingsDialog) -> None:
        job = DownloadJob("clip")
        job.progress.connect(dialog.download_progress)
        job.finished.connect(lambda _path: dialog.download_done())
        job.failed.connect(lambda msg: (dialog.download_done(), QMessageBox.warning(dialog, "Download failed", msg)))
        self._download_job = job
        start(job, self)

    def write_xmp(self) -> None:
        if self.result is None or self._busy():
            return
        c = self.result.counts()
        answer = QMessageBox.question(
            self,
            "Write XMP ratings",
            f"Write ratings for {c['shots']} photos ({c['pick']} picks, {c['alternate']} alternates, "
            f"{c['reject']} rejects) to XMP sidecar files next to your photos?\n\n"
            "Existing metadata is kept, ratings you set by hand in other apps are not overwritten, "
            "and “pickframe reset” undoes everything.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.result.settings = self.settings
        self.job = WriteJob(self.result, self.embed_jpeg)
        self.job.progress.connect(self._write_progress)
        self.job.finished.connect(self._write_done)
        self.job.failed.connect(self._job_failed)
        self.progress.setVisible(True)
        self.thread = start(self.job, self)
        self.thread.finished.connect(self._job_cleanup)
        self._update_actions()

    def _write_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(f"Writing {done}/{total} · {name}")

    def _write_done(self, report) -> None:
        self.unsaved = False
        lines = [f"Updated {len(report.written)} file(s); {len(report.unchanged)} were already up to date."]
        if report.skipped_manual:
            lines.append(f"Kept {len(report.skipped_manual)} rating(s) you set by hand.")
        if report.errors:
            lines.append(f"{len(report.errors)} error(s), e.g. {report.errors[0][0].name}: {report.errors[0][1]}")
        self.status_label.setText(" ".join(lines))
        QMessageBox.information(self, "Pickframe", "\n".join(lines))

    def export_report(self) -> None:
        if self.result is None:
            return
        start_path = str((self.folder or Path.home()) / "pickframe-report.html")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", start_path, "HTML report (*.html);;CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return
        from ..report import write_report

        try:
            write_report(self.result, Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Pickframe", f"Could not write the report: {exc}")
            return
        if Path(path).suffix.lower() in (".html", ".htm"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def about(self) -> None:
        QMessageBox.about(
            self,
            "About Pickframe",
            f"<h3>Pickframe {__version__}</h3><p>{version_string().replace('Pickframe ', '')}</p>"
            "<p>Free, local AI photo culling. Licensed under the GNU GPL v3 or later.</p>"
            f'<p><a href="{REPO_URL}">{REPO_URL}</a></p>'
            "<p>Face models: YuNet (MIT) and MediaPipe Face Landmarker (Apache-2.0).</p>",
        )

    # -- events -----------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self.open_folder(path)
                return
            if path.is_file():
                self.open_folder(path.parent)
                return

    def closeEvent(self, event) -> None:
        if self.unsaved and self.result is not None and not self._busy():
            answer = QMessageBox.question(
                self,
                "Pickframe",
                "Ratings have not been written to XMP yet. Quit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if isinstance(self.job, AnalysisJob):
            self.job.cancel.set()
        self._save_store()
        self.pool.waitForDone(2000)
        super().closeEvent(event)


def create_app(argv: list[str]) -> QApplication:
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Pickframe")
    app.setOrganizationName("Pickframe")
    app.setApplicationVersion(__version__)
    if ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    theme.apply(app)  # type: ignore[arg-type]
    return app  # type: ignore[return-value]


def run(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    app = create_app([sys.argv[0], *args])
    window = MainWindow()
    window.show()
    folder = next((Path(a) for a in args if not a.startswith("-")), None)
    if folder is not None and folder.exists():
        QTimer.singleShot(0, lambda: window.open_folder(folder if folder.is_dir() else folder.parent))
    return app.exec()

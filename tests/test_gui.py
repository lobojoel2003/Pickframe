from __future__ import annotations

from pathlib import Path

import pytest

from test_pipeline import make_shoot

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from pickframe.config import RejectStyle, Settings, SidecarStyle  # noqa: E402
from pickframe.gui import app as gui_app  # noqa: E402
from pickframe.gui import theme  # noqa: E402
from pickframe.gui.widgets import SettingsDialog, ShotRole, stars_text  # noqa: E402
from pickframe.gui.workers import (  # noqa: E402
    AnalysisJob,
    ThumbnailSignals,
    ThumbnailTask,
    WriteJob,
    to_qimage,
)
from pickframe.pipeline import Analyzer, CullResult  # noqa: E402


@pytest.fixture
def shoot(tmp_path: Path, astronaut_rgb, chelsea_rgb) -> Path:
    return make_shoot(tmp_path / "shoot", astronaut_rgb, chelsea_rgb)


@pytest.fixture
def result(shoot: Path) -> CullResult:
    return Analyzer().run([shoot])


@pytest.fixture
def window(qtbot, tmp_path: Path, monkeypatch):
    gui_app.create_app([])
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = gui_app.MainWindow(store)
    qtbot.addWidget(win)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    # pytest-qt closes the window before fixture teardown: never block on "quit anyway?"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    yield win
    win.unsaved = False
    win.close()


def test_stars_text():
    assert stars_text(None) == "unrated"
    assert stars_text(0) == "unrated"
    assert stars_text(-1) == "✕"
    assert stars_text(3) == "★★★☆☆"


def test_window_shows_result(window, result: CullResult, qtbot):
    window.show()
    assert window.stack.currentIndex() == 0  # welcome page
    window.set_result(result)
    assert window.stack.currentIndex() == 1
    assert window.groups.count() == len(result.groups) + 1
    assert window.model.rowCount() == len(result.shots)
    assert window.act_write.isEnabled()
    assert "photos" in window.status_label.text()

    window.filter.setCurrentIndex(3)  # rejects
    rejects = [i for i, d in result.decisions.items() if d.status == "reject"]
    assert window.model.ids == rejects
    window.filter.setCurrentIndex(1)  # picks
    assert all(result.decisions[i].status == "pick" for i in window.model.ids)
    window.filter.setCurrentIndex(0)
    window.groups.setCurrentRow(2)  # second group only
    assert window.model.ids == result.groups[1]
    window.groups.setCurrentRow(0)

    qtbot.waitUntil(lambda: len(window.model.thumbs) == len(result.shots), timeout=10000)
    image = window.grid.grab()  # paints every card through the delegate
    assert not image.isNull()
    assert not window.details.grab().isNull()


def test_rating_override_and_revert(window, result: CullResult):
    window.set_result(result)
    window.grid.setCurrentIndex(window.model.index(0))
    shot_id = window._current_id()
    window.set_rating(4)
    assert result.overrides[shot_id] == 4
    assert window._current_id() != shot_id  # moved on to the next photo
    window.grid.setCurrentIndex(window.model.index(window.model.row_of(shot_id)))
    window.set_rating(0)
    assert result.overrides[shot_id] is None
    window.set_rating(-1)
    window.grid.setCurrentIndex(window.model.index(window.model.row_of(shot_id)))
    window.set_rating(99)
    assert shot_id not in result.overrides
    assert window.unsaved


def test_thumbnail_and_preview_slots(window, result: CullResult, astronaut_rgb):
    window.set_result(result)
    image = to_qimage(astronaut_rgb)
    assert image.width() == 512
    assert image.height() == 512
    current = window._current_id()
    window._thumb_loaded(current, image)
    window._preview_loaded(current, image)
    assert current in window.previews
    window._preview_loaded(current, QImage())  # ignored
    window._thumb_loaded(current, QImage())
    assert window.model.thumbs[current].isNull()


def test_thumbnail_task(qtbot, astronaut_path: Path, tmp_path: Path):
    signals = ThumbnailSignals()
    got = []
    signals.loaded.connect(lambda shot_id, image: got.append((shot_id, image)))
    ThumbnailTask(7, astronaut_path, 64, signals).run()
    ThumbnailTask(8, tmp_path / "missing.jpg", 64, signals).run()
    assert got[0][0] == 7
    assert max(got[0][1].width(), got[0][1].height()) == 64
    assert got[1][0] == 8
    assert got[1][1].isNull()


def test_jobs_run_synchronously(shoot: Path):
    finished: list[object] = []
    job = AnalysisJob([shoot], Settings(), recursive=False)
    job.finished.connect(finished.append)
    job.run()
    assert isinstance(finished[0], CullResult)

    reports: list[object] = []
    writer = WriteJob(finished[0], embed_jpeg=False)  # type: ignore[arg-type]
    writer.finished.connect(reports.append)
    writer.run()
    assert len(reports[0].written) == 4  # type: ignore[attr-defined]

    failures: list[str] = []
    # a different analysis size misses the cache, so the cancel flag is checked
    job = AnalysisJob([shoot], Settings(analysis_size=800), recursive=False)
    job.failed.connect(failures.append)
    job.cancel.set()
    job.run()
    assert failures == ["Analysis cancelled."]
    failures.clear()
    job.cancel.clear()
    job.settings = Settings(embedder="clip")  # not installed -> failure is reported, not raised
    job.run()
    assert failures
    assert "failed" in failures[0].lower()


def test_open_folder_analyses_in_background(window, shoot: Path, qtbot):
    window.open_folder(shoot)
    assert window.windowTitle().endswith("shoot")
    qtbot.waitUntil(lambda: window.result is not None and window.thread is None, timeout=30000)
    assert len(window.result.shots) == 4
    assert window.act_analyse.isEnabled()


def test_write_xmp_from_window(window, result: CullResult, qtbot, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.set_result(result)
    window.write_xmp()
    qtbot.waitUntil(lambda: window.thread is None, timeout=30000)
    folder = result.shots[0].primary.parent
    assert len(list(folder.glob("*.xmp"))) == 4
    assert not window.unsaved


def test_write_xmp_cancelled(window, result: CullResult, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    window.set_result(result)
    window.write_xmp()
    assert window.thread is None
    assert not list(result.shots[0].primary.parent.glob("*.xmp"))


def test_export_report(window, result: CullResult, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "report.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.set_result(result)
    window.export_report()
    assert target.is_file()


def test_settings_dialog_roundtrip(qtbot):
    gui_app.create_app([])
    settings = Settings(
        blur_threshold=0.4, eyes_threshold=0.6, picks_per_group=2, embedder="classic", similarity=0.7,
        max_gap=30, sidecar_style=SidecarStyle.DARKTABLE, reject_style=RejectStyle.ONE_STAR,
        write_flags=False, pick_label="Green", overwrite_manual=True, smile_weight=0.2,
    )  # fmt: skip
    dialog = SettingsDialog(settings, embed_jpeg=True)
    qtbot.addWidget(dialog)
    values, embed = dialog.values()
    assert embed is True
    for name in ("blur_threshold", "eyes_threshold", "picks_per_group", "embedder", "max_gap",
                 "sidecar_style", "reject_style", "write_flags", "pick_label", "overwrite_manual"):  # fmt: skip
        assert getattr(values, name) == getattr(settings, name), name
    assert values.similarity == pytest.approx(0.7)
    assert values.smile_weight == pytest.approx(0.2)
    dialog.similarity.setValue(0.0)
    assert dialog.values()[0].similarity is None  # "model default"
    dialog.download_progress(5, 10)
    assert dialog.clip_progress.isVisible() or not dialog.isVisible()
    dialog.download_done()
    assert not dialog.clip_progress.isVisible()


def test_settings_change_regroups(window, result: CullResult, monkeypatch):
    window.set_result(result)
    groups_before = window.groups.count()

    class FakeDialog:
        DialogCode = SettingsDialog.DialogCode

        def __init__(self, settings, embed, parent):
            self.download_requested = type("S", (), {"connect": lambda self, f: None})()
            self._settings = Settings.from_dict(settings.to_dict())
            self._settings.similarity = 0.0
            self._settings.max_gap = 100000

        def exec(self):
            return SettingsDialog.DialogCode.Accepted

        def values(self):
            return self._settings, False

    monkeypatch.setattr(gui_app, "SettingsDialog", FakeDialog)
    window.open_settings()
    assert window.groups.count() == 2 < groups_before  # "All photos" + one group


def test_drop_and_close(window, shoot: Path, qtbot):
    from PySide6.QtCore import QMimeData, QUrl

    window.set_result(Analyzer().run([shoot]))
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(shoot / "DSC_0001.jpg"))])

    class Event:
        def mimeData(self):
            return mime

        def acceptProposedAction(self):
            pass

    window.dragEnterEvent(Event())
    window.dropEvent(Event())
    assert window.folder == shoot
    qtbot.waitUntil(lambda: window.thread is None, timeout=30000)
    window.unsaved = False
    assert window.close()


def test_close_asks_when_unsaved(window, result: CullResult, monkeypatch):
    window.set_result(result)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert window.unsaved
    assert not window.close()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert window.close()


def test_about_and_theme(window, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "about", lambda *a, **k: shown.append(a[2]))
    window.about()
    assert "Pickframe" in shown[0]
    assert theme.STATUS_COLORS["pick"] == theme.PICK


def test_shot_role(window, result: CullResult):
    window.set_result(result)
    index = window.model.index(0)
    assert index.data(ShotRole) == window.model.ids[0]
    assert "Score" in index.data(Qt.ItemDataRole.ToolTipRole)
    assert index.data(Qt.ItemDataRole.DisplayRole) == result.shots[window.model.ids[0]].label()
    assert window.model.row_of(9999) == -1

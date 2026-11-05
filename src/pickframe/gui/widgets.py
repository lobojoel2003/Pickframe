"""Custom widgets: thumbnail grid, preview, detail panel and settings dialog."""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from .. import models
from ..config import RejectStyle, Settings, SidecarStyle
from ..pipeline import CullResult
from ..results import FLAG_TEXT, HARD_FLAGS
from ..scoring import face_closed, prominent_faces
from . import theme

THUMB = 196
CARD_W, CARD_H = THUMB + 16, THUMB + 58

ShotRole = Qt.ItemDataRole.UserRole + 1


def stars_text(rating: int | None) -> str:
    if rating is None or rating == 0:
        return "unrated"
    if rating < 0:
        return "✕"
    return "★" * rating + "☆" * (5 - rating)


class ShotModel(QAbstractListModel):
    """The shots currently shown in the grid (filtered by group and status)."""

    def __init__(self) -> None:
        super().__init__()
        self.result: CullResult | None = None
        self.ids: list[int] = []
        self.thumbs: dict[int, QPixmap] = {}

    def set_result(self, result: CullResult | None) -> None:
        self.beginResetModel()
        self.result = result
        self.ids = []
        self.thumbs = {}
        self.endResetModel()

    def set_ids(self, ids: list[int]) -> None:
        self.beginResetModel()
        self.ids = ids
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self.ids)

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ):
        if not index.isValid() or self.result is None:
            return None
        shot_id = self.ids[index.row()]
        if role == ShotRole:
            return shot_id
        if role == Qt.ItemDataRole.DisplayRole:
            return self.result.shots[shot_id].label()
        if role == Qt.ItemDataRole.ToolTipRole:
            d = self.result.decisions[shot_id]
            return f"{self.result.shots[shot_id].primary}\nScore {d.score * 100:.0f} · {d.summary}"
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(CARD_W, CARD_H)
        return None

    def row_of(self, shot_id: int) -> int:
        try:
            return self.ids.index(shot_id)
        except ValueError:
            return -1

    def thumbnail_ready(self, shot_id: int, image: QImage) -> None:
        self.thumbs[shot_id] = QPixmap.fromImage(image) if not image.isNull() else QPixmap()
        row = self.row_of(shot_id)
        if row >= 0:
            idx = self.index(row)
            self.dataChanged.emit(idx, idx)

    def refresh(self) -> None:
        if self.ids:
            self.dataChanged.emit(self.index(0), self.index(len(self.ids) - 1))


class ShotDelegate(QStyledItemDelegate):
    """Paints a card: thumbnail, status border, stars, badges, file name."""

    def __init__(self, model: ShotModel, request_thumb):
        super().__init__()
        self.model = model
        self.request_thumb = request_thumb

    def sizeHint(self, option, index) -> QSize:
        return QSize(CARD_W, CARD_H)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        result = self.model.result
        if result is None:
            return
        shot_id = index.data(ShotRole)
        decision = result.decisions[shot_id]
        rating = result.rating(shot_id)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRect(option.rect).adjusted(6, 6, -6, -6)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 10, 10)
        painter.fillPath(path, QColor(theme.CARD))
        color = QColor(theme.STATUS_COLORS.get(decision.status, theme.ALTERNATE))
        if selected:
            painter.setPen(QPen(QColor(theme.ACCENT), 3))
        elif decision.status in ("pick", "reject"):
            painter.setPen(QPen(color, 2))
        else:
            painter.setPen(QPen(QColor(theme.LINE), 1))
        painter.drawPath(path)

        image_rect = QRect(rect.left() + 6, rect.top() + 6, rect.width() - 12, THUMB - 8)
        pixmap = self.model.thumbs.get(shot_id)
        if pixmap is None:
            self.request_thumb(shot_id)
            painter.fillRect(image_rect, QColor(theme.BG))
        elif not pixmap.isNull():
            scaled = pixmap.scaled(
                image_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = image_rect.left() + (image_rect.width() - scaled.width()) // 2
            y = image_rect.top() + (image_rect.height() - scaled.height()) // 2
            if decision.status == "reject":
                painter.setOpacity(0.55)
            painter.drawPixmap(x, y, scaled)
            painter.setOpacity(1.0)

        # status badge
        badge_font = QFont(painter.font())
        badge_font.setPointSizeF(max(7.0, badge_font.pointSizeF() - 2))
        badge_font.setBold(True)
        painter.setFont(badge_font)
        label = decision.status.upper()
        metrics = painter.fontMetrics()
        badge = QRect(
            image_rect.left() + 6, image_rect.top() + 6, metrics.horizontalAdvance(label) + 14, 18
        )
        bpath = QPainterPath()
        bpath.addRoundedRect(QRectF(badge), 9, 9)
        painter.fillPath(bpath, color)
        painter.setPen(QColor("#0f1115"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, label)

        # stars
        if rating:
            text = "✕" if rating < 0 else "★" * rating
            painter.setFont(badge_font)
            width = painter.fontMetrics().horizontalAdvance(text) + 14
            srect = QRect(image_rect.right() - width - 6, image_rect.top() + 6, width, 18)
            spath = QPainterPath()
            spath.addRoundedRect(QRectF(srect), 9, 9)
            painter.fillPath(spath, QColor(0, 0, 0, 160))
            painter.setPen(QColor(theme.REJECT if rating < 0 else theme.ACCENT))
            painter.drawText(srect, Qt.AlignmentFlag.AlignCenter, text)

        # name and reason
        text_font = QFont(option.font)
        painter.setFont(text_font)
        painter.setPen(QColor(theme.TEXT))
        name_rect = QRect(rect.left() + 10, image_rect.bottom() + 6, rect.width() - 20, 18)
        name = painter.fontMetrics().elidedText(
            result.shots[shot_id].label(), Qt.TextElideMode.ElideMiddle, name_rect.width()
        )
        painter.drawText(
            name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name
        )
        small = QFont(text_font)
        small.setPointSizeF(max(7.0, text_font.pointSizeF() - 1.5))
        painter.setFont(small)
        hard = [f for f in decision.flags if f in HARD_FLAGS]
        painter.setPen(QColor(theme.REJECT if hard else theme.MUTED))
        reason = f"{decision.score * 100:.0f} · {decision.summary}"
        reason_rect = QRect(name_rect.left(), name_rect.bottom() + 1, name_rect.width(), 16)
        reason = painter.fontMetrics().elidedText(
            reason, Qt.TextElideMode.ElideRight, reason_rect.width()
        )
        painter.drawText(
            reason_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, reason
        )
        painter.restore()


class Preview(QWidget):
    """Large preview with face boxes (green = eyes open, red = eyes closed)."""

    def __init__(self) -> None:
        super().__init__()
        self.pixmap: QPixmap | None = None
        self.boxes: list[tuple[tuple[float, float, float, float], bool]] = []
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, pixmap: QPixmap | None, boxes=()) -> None:
        self.pixmap = pixmap
        self.boxes = list(boxes)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.BG))
        if not self.pixmap or self.pixmap.isNull():
            painter.setPen(QColor(theme.MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No photo selected")
            return
        scaled = self.pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        for (bx, by, bw, bh), closed in self.boxes:
            painter.setPen(QPen(QColor(theme.REJECT if closed else theme.PICK), 2))
            painter.drawRoundedRect(
                QRectF(x + bx * scaled.width(), y + by * scaled.height(), bw * scaled.width(), bh * scaled.height()),
                4, 4,
            )  # fmt: skip


class Meter(QWidget):
    """A labelled horizontal bar for a 0..1 value."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self.value: float | None = None
        self.setMinimumHeight(30)

    def set_value(self, value: float | None) -> None:
        self.value = value
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(theme.MUTED))
        painter.drawText(QRect(0, 0, self.width(), 14), Qt.AlignmentFlag.AlignLeft, self.label)
        text = "–" if self.value is None else f"{self.value * 100:.0f}"
        painter.setPen(QColor(theme.TEXT))
        painter.drawText(QRect(0, 0, self.width(), 14), Qt.AlignmentFlag.AlignRight, text)
        track = QRectF(0, 19, self.width(), 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.LINE))
        painter.drawRoundedRect(track, 3, 3)
        if self.value is not None:
            v = max(0.0, min(1.0, self.value))
            color = QColor(theme.REJECT if v < 0.35 else theme.ACCENT if v < 0.65 else theme.PICK)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(0, 19, self.width() * v, 6), 3, 3)


class DetailPanel(QWidget):
    rating_requested = Signal(int)  # -1..5, or 99 = revert to Pickframe's choice

    def __init__(self) -> None:
        super().__init__()
        self.preview = Preview()
        self.title = QLabel("")
        self.title.setObjectName("title")
        self.title.setWordWrap(True)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.meta = QLabel("")
        self.meta.setObjectName("muted")
        self.meta.setWordWrap(True)
        self.meters = {
            "score": Meter("Overall score"),
            "subject": Meter("Subject focus"),
            "exposure": Meter("Exposure"),
            "eyes": Meter("Eyes open"),
            "smile": Meter("Smile"),
        }
        buttons = QHBoxLayout()
        for text, value, tip in (
            ("✕", -1, "Reject (X)"), ("1", 1, "1 star"), ("2", 2, "2 stars"), ("3", 3, "3 stars"),
            ("4", 4, "4 stars"), ("5", 5, "5 stars"), ("↺", 99, "Use Pickframe's rating (U)"),
        ):  # fmt: skip
            button = QPushButton(text)
            button.setToolTip(tip)
            button.setFixedWidth(38)
            button.clicked.connect(lambda _=False, v=value: self.rating_requested.emit(v))
            buttons.addWidget(button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        layout.addWidget(self.preview, 3)
        layout.addWidget(self.title)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        for meter in self.meters.values():
            layout.addWidget(meter)
        layout.addWidget(self.meta)
        layout.addStretch(1)

    def show_shot(
        self, result: CullResult | None, shot_id: int | None, pixmap: QPixmap | None
    ) -> None:
        if result is None or shot_id is None:
            self.preview.set_image(None)
            self.title.setText("")
            self.status.setText("")
            self.meta.setText("")
            for meter in self.meters.values():
                meter.set_value(None)
            return
        shot = result.shots[shot_id]
        d = result.decisions[shot_id]
        a = result.analyses[shot_id]
        faces = prominent_faces(a.faces)
        self.preview.set_image(pixmap, [(f.box, face_closed(f, result.settings)) for f in faces])
        self.title.setText(shot.label())
        color = theme.STATUS_COLORS.get(d.status, theme.ALTERNATE)
        override = " (set by you)" if shot_id in result.overrides else ""
        flags = ", ".join(FLAG_TEXT.get(f, f) for f in d.flags) or d.summary
        self.status.setText(
            f'<span style="color:{color};font-weight:600">{d.status.upper()}</span> &nbsp; '
            f'<span style="color:{theme.ACCENT}">{stars_text(result.rating(shot_id))}</span>{override}'
            f'<br><span style="color:{theme.MUTED}">Group {d.group + 1} · rank {d.rank} of {d.group_size} · {flags}</span>'
        )
        self.meters["score"].set_value(d.score)
        self.meters["subject"].set_value(d.subject_sharpness)
        self.meters["exposure"].set_value(a.exposure)
        if faces:
            self.meters["eyes"].set_value(1 - max(f.eyes_closed_probability for f in faces))
            self.meters["smile"].set_value(sum(f.smile for f in faces) / len(faces))
        else:
            self.meters["eyes"].set_value(None)
            self.meters["smile"].set_value(None)
        when = (
            datetime.fromtimestamp(a.capture_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if a.capture_time is not None
            else "unknown time"
        )
        files = " + ".join(p.name for p in shot.files) + f"<br>{shot.primary.parent}"
        self.meta.setText(
            f"{a.width} × {a.height} · {a.camera or 'unknown camera'} · {when}<br>"
            f"{len(a.faces)} face(s) detected · {len(faces)} main<br>{files}"
            + (f'<br><span style="color:{theme.REJECT}">{a.error}</span>' if a.error else "")
        )


class SettingsDialog(QDialog):
    """Thresholds, output options and the optional CLIP model."""

    download_requested = Signal()

    def __init__(self, settings: Settings, embed_jpeg: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Pickframe settings")
        self.setMinimumWidth(520)
        s = settings

        def slider(value: float, lo: float = 0.0, hi: float = 1.0):
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(0.05)
            box.setDecimals(2)
            box.setValue(value)
            return box

        culling = QGroupBox("Culling")
        form = QFormLayout(culling)
        self.blur = slider(s.blur_threshold)
        self.blur.setToolTip("Shots with subject focus below this are rejected as blurry.")
        self.eyes = slider(s.eyes_threshold)
        self.eyes.setToolTip("Faces whose eyes-closed probability is above this reject the shot.")
        self.smile = slider(s.smile_weight)
        self.picks = QSpinBox()
        self.picks.setRange(1, 20)
        self.picks.setValue(s.picks_per_group)
        form.addRow("Blur threshold", self.blur)
        form.addRow("Closed-eyes threshold", self.eyes)
        form.addRow("Smile weight", self.smile)
        form.addRow("Picks per group", self.picks)

        grouping = QGroupBox("Grouping")
        gform = QFormLayout(grouping)
        self.embedder = QComboBox()
        self.embedder.addItems(["auto", "clip", "classic"])
        self.embedder.setCurrentText(s.embedder)
        self.similarity = slider(s.similarity if s.similarity is not None else 0.0)
        self.similarity.setSpecialValueText("model default")
        self.gap = QDoubleSpinBox()
        self.gap.setRange(1, 3600)
        self.gap.setSuffix(" s")
        self.gap.setValue(s.max_gap)
        self.clip_state = QLabel()
        self.clip_progress = QProgressBar()
        self.clip_progress.setVisible(False)
        self.clip_button = QPushButton("Download CLIP model (89 MB)")
        self.clip_button.clicked.connect(self.download_requested.emit)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip_state, 1)
        clip_row.addWidget(self.clip_button)
        gform.addRow("Similarity model", self.embedder)
        gform.addRow("Similarity threshold", self.similarity)
        gform.addRow("Max time between shots", self.gap)
        gform.addRow(clip_row)
        gform.addRow(self.clip_progress)
        self.refresh_clip_state()

        output = QGroupBox("XMP output")
        oform = QFormLayout(output)
        self.sidecar = QComboBox()
        for style, text in (
            (SidecarStyle.AUTO, "Both (recommended)"),
            (SidecarStyle.DARKTABLE, "IMG_0001.CR3.xmp (darktable, digiKam)"),
            (SidecarStyle.LIGHTROOM, "IMG_0001.xmp (Lightroom, Capture One)"),
        ):
            self.sidecar.addItem(text, style)
        self.sidecar.setCurrentIndex(self.sidecar.findData(s.sidecar_style))
        self.reject = QComboBox()
        for style, text in (
            (RejectStyle.REJECT, "Rejected (rating −1 + reject flag)"),
            (RejectStyle.ONE_STAR, "1 star"),
            (RejectStyle.NONE, "Leave unrated"),
        ):
            self.reject.addItem(text, style)
        self.reject.setCurrentIndex(self.reject.findData(s.reject_style))
        self.flags = QCheckBox("Write pick/reject flags (Lightroom 13.2+, digiKam)")
        self.flags.setChecked(s.write_flags)
        self.embed = QCheckBox("Also embed XMP in JPEG files (Lightroom + JPEG)")
        self.embed.setChecked(embed_jpeg)
        self.overwrite = QCheckBox("Overwrite ratings I set by hand")
        self.overwrite.setChecked(s.overwrite_manual)
        self.label = QLineEdit(s.pick_label)
        self.label.setPlaceholderText("e.g. Green (optional)")
        oform.addRow("Sidecar names", self.sidecar)
        oform.addRow("Write rejects as", self.reject)
        oform.addRow("Colour label for picks", self.label)
        oform.addRow(self.flags)
        oform.addRow(self.embed)
        oform.addRow(self.overwrite)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject_dialog)
        layout = QVBoxLayout(self)
        layout.addWidget(culling)
        layout.addWidget(grouping)
        layout.addWidget(output)
        layout.addWidget(buttons)
        self._base = s

    def reject_dialog(self) -> None:
        QDialog.reject(self)

    def refresh_clip_state(self) -> None:
        installed = models.is_installed("clip")
        self.clip_state.setText(
            "CLIP model installed ✓" if installed else "CLIP model not installed"
        )
        self.clip_button.setVisible(not installed)

    def download_progress(self, done: int, total: int) -> None:
        self.clip_progress.setVisible(True)
        self.clip_button.setEnabled(False)
        self.clip_progress.setRange(0, total or 0)
        self.clip_progress.setValue(done)

    def download_done(self) -> None:
        self.clip_progress.setVisible(False)
        self.clip_button.setEnabled(True)
        self.refresh_clip_state()

    def values(self) -> tuple[Settings, bool]:
        s = Settings.from_dict(self._base.to_dict())
        s.blur_threshold = self.blur.value()
        s.eyes_threshold = self.eyes.value()
        s.smile_weight = self.smile.value()
        s.picks_per_group = self.picks.value()
        s.embedder = self.embedder.currentText()
        s.similarity = self.similarity.value() or None
        s.max_gap = self.gap.value()
        s.sidecar_style = self.sidecar.currentData()
        s.reject_style = self.reject.currentData()
        s.write_flags = self.flags.isChecked()
        s.overwrite_manual = self.overwrite.isChecked()
        s.pick_label = self.label.text().strip()
        return s, self.embed.isChecked()

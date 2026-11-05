"""Dark theme shared by every window."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

BG = "#0f1115"
PANEL = "#171a21"
CARD = "#1d212a"
LINE = "#2a2f3a"
TEXT = "#e8eaf0"
MUTED = "#9aa3b2"
ACCENT = "#ffb547"
PICK = "#3ecf8e"
ALTERNATE = "#7c8696"
REJECT = "#ff5c69"

STATUS_COLORS = {"pick": PICK, "alternate": ALTERNATE, "reject": REJECT}

STYLESHEET = f"""
QWidget {{ color: {TEXT}; font-size: 13px; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolBar {{ background: {PANEL}; border: none; border-bottom: 1px solid {LINE}; padding: 6px; spacing: 6px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 6px 12px; }}
QToolBar QToolButton:hover {{ background: {CARD}; border-color: {LINE}; }}
QToolBar QToolButton:disabled {{ color: #5b6270; }}
QToolButton#primary {{ background: {ACCENT}; color: #1b1405; font-weight: 600; }}
QToolButton#primary:hover {{ background: #ffc56e; }}
QToolButton#primary:disabled {{ background: #4a3f2a; color: #8b7d63; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; padding: 8px 10px; color: {MUTED}; text-transform: uppercase; }}
QListView, QListWidget, QScrollArea, QTextBrowser {{ background: {BG}; border: none; outline: 0; }}
QListWidget::item {{ border-radius: 8px; padding: 6px; margin: 2px 6px; }}
QListWidget::item:selected {{ background: {CARD}; color: {TEXT}; border: 1px solid {ACCENT}; }}
QListWidget::item:hover {{ background: {PANEL}; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {LINE}; color: {MUTED}; }}
QProgressBar {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 6px; height: 12px; text-align: center; color: {TEXT}; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 6px; padding: 4px 8px; }}
QComboBox QAbstractItemView {{ background: {CARD}; selection-background-color: {LINE}; }}
QPushButton {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:default {{ background: {ACCENT}; color: #1b1405; border-color: {ACCENT}; font-weight: 600; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 10px; margin-top: 14px; padding: 12px 10px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#title {{ font-size: 16px; font-weight: 600; }}
QLabel#hero {{ font-size: 26px; font-weight: 700; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {LINE}; padding: 6px; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(CARD))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#1b1405"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(CARD))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(MUTED))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)

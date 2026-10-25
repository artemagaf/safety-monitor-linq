"""Оформление: графит + сигнальный жёлтый (цвет знаков безопасности), как в макете."""
from PySide6.QtGui import QColor

BG, PANEL, PANEL2, LINE = "#121519", "#1a1e23", "#20252b", "#2d333b"
FG, MUTED = "#e7eaee", "#8e98a4"
ACCENT, ACCENT_INK = "#f2b705", "#1b1600"
CRIT, CRIT_BG, OK, OK_BG, INFO, INFO_BG = "#ff5d63", "#3a1c1f", "#45c08a", "#16302a", "#6aa5ec", "#1a2a3d"
FONT, MONO = "Segoe UI", "Consolas"

STATUS_STYLE = {   # статус тревоги -> (подпись, цвет текста, фон)
    "new": ("Новое", CRIT, CRIT_BG),
    "conf": ("Подтверждено", OK, OK_BG),
    "sent": ("Передано мастеру", INFO, INFO_BG),
    "false": ("Ложное", MUTED, PANEL2),
}


def qc(hex_: str, alpha: int = 255) -> QColor:
    c = QColor(hex_)
    c.setAlpha(alpha)
    return c


QSS = f"""
* {{ font-family: "{FONT}"; font-size: 10pt; color: {FG}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#page {{ background: {PANEL}; }}
QFrame#nav {{ background: {PANEL2}; border-right: 1px solid {LINE}; }}
QLabel#brand {{ font-size: 12pt; font-weight: 600; }}
QLabel#brandSub, QLabel.muted, QLabel#muted {{ color: {MUTED}; font-size: 9pt; }}
QPushButton#navBtn {{ text-align: left; padding: 8px 12px; border: 0; border-radius: 6px; color: {MUTED}; background: transparent; font-weight: 500; }}
QPushButton#navBtn:hover {{ background: {PANEL}; color: {FG}; }}
QPushButton#navBtn:checked {{ background: {PANEL}; color: {FG}; border-left: 3px solid {ACCENT}; }}
QLabel#title {{ font-size: 15pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 11pt; font-weight: 600; }}
QLabel#chip {{ border: 1px solid {LINE}; border-radius: 11px; padding: 3px 10px; color: {MUTED}; }}
QLabel#chipAccent {{ border: 1px solid {ACCENT}; border-radius: 11px; padding: 3px 10px; }}
QFrame#topbar {{ border-bottom: 1px solid {LINE}; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; }}
QFrame#kpi {{ border: 1px solid {LINE}; border-radius: 6px; }}
QLabel#kpiValue {{ font-size: 20pt; font-weight: 600; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {LINE}; border-radius: 4px; padding: 5px 10px; }}
QPushButton:hover {{ border-color: {MUTED}; }}
QPushButton#primary {{ background: {ACCENT}; color: {ACCENT_INK}; border-color: {ACCENT}; font-weight: 600; }}
QPushButton:disabled {{ color: {MUTED}; }}
QComboBox, QLineEdit, QDateEdit {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 4px; padding: 4px 8px; min-height: 20px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE}; selection-background-color: {PANEL2}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {LINE}; border-radius: 3px; background: {PANEL}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QTableWidget {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; gridline-color: {LINE}; selection-background-color: {PANEL2}; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; border: 0; border-bottom: 1px solid {LINE}; padding: 6px 8px; font-size: 8pt; font-weight: 600; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollArea > QWidget > QWidget#feedHost {{ background: transparent; }}
QScrollArea > QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 4px; min-width: 30px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QStatusBar {{ background: {PANEL2}; color: {MUTED}; border-top: 1px solid {LINE}; }}
QListWidget {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; }}
QDialog {{ background: {PANEL}; }}
QToolTip {{ background: {PANEL2}; color: {FG}; border: 1px solid {LINE}; }}
"""

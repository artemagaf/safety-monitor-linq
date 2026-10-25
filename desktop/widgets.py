"""Виджеты приложения: плитка камеры, карточка тревоги, графики, редактор зон, проигрыватель клипов."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from . import theme as T


def to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def fit_rect(img_w: int, img_h: int, box: QRectF) -> QRectF:
    """Прямоугольник картинки внутри box с сохранением пропорций (по центру)."""
    s = min(box.width() / img_w, box.height() / img_h)
    w, h = img_w * s, img_h * s
    return QRectF(box.x() + (box.width() - w) / 2, box.y() + (box.height() - h) / 2, w, h)


def tag(p: QPainter, x: float, y: float, text: str, bg: QColor, fg=QColor("#ffffff"), right=False):
    """Плашка с текстом поверх видео (как на макете)."""
    f = QFont(T.MONO, 8)
    f.setBold(True)
    p.setFont(f)
    w = p.fontMetrics().horizontalAdvance(text) + 12
    h = p.fontMetrics().height() + 4
    r = QRectF(x - w if right else x, y, w, h)
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(r, 3, 3)
    p.setPen(fg)
    p.drawText(r, Qt.AlignCenter, text)
    return r


# ---------- камера ----------

class CameraTile(QWidget):
    """Живой кадр камеры с разметкой нарушений (рисует конвейер) и плашками состояния."""
    clicked = Signal(str)
    power_off = Signal(str)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.setMinimumSize(320, 200)
        # выключить камеру прямо с плитки (включить обратно — кнопка «Камеры» над сеткой)
        self.off_btn = QPushButton("⏻ Выключить", self)
        self.off_btn.setCursor(Qt.PointingHandCursor)
        self.off_btn.setToolTip("Выключить камеру: поток и запись остановятся. Включить — «Камеры» над сеткой")
        self.off_btn.setStyleSheet("QPushButton { background: rgba(10,12,15,190); border: 0; border-radius: 4px;"
                                   " color: #fff; padding: 4px 9px; font-weight: 600; }"
                                   " QPushButton:hover { background: #d4363e; }")
        self.off_btn.clicked.connect(lambda: self.power_off.emit(self.worker.cam["id"]))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img: QImage | None = None
        self._frame_id = None
        self.setCursor(Qt.PointingHandCursor)

        # ручной выбор стоп-кадра: ◀ 12 / 65 ▶  Выбрать…
        self.picker = None
        if worker.cam.get("manual") and len(worker.images) > 1:
            self.picker = QFrame(self)
            self.picker.setStyleSheet("QFrame { background: rgba(10,12,15,190); border-radius: 4px; }"
                                      "QPushButton { background: transparent; border: 0; color: #fff; padding: 3px 9px;"
                                      " font-weight: 600; } QPushButton:hover { background: rgba(255,255,255,30); }"
                                      "QLabel { color: #fff; font-family: Consolas; font-size: 9pt; padding: 0 4px; }")
            row = QHBoxLayout(self.picker)
            row.setContentsMargins(2, 2, 2, 2)
            row.setSpacing(0)
            prev_b, next_b, pick_b = QPushButton("◀"), QPushButton("▶"), QPushButton("Выбрать…")
            self.counter = QLabel()
            prev_b.clicked.connect(lambda: self._step(-1))
            next_b.clicked.connect(lambda: self._step(1))
            pick_b.clicked.connect(self._choose)
            for b in (prev_b, next_b, pick_b):
                b.setCursor(Qt.PointingHandCursor)
            for w in (prev_b, self.counter, next_b, pick_b):
                row.addWidget(w)
            self._update_counter()

    def _update_counter(self):
        self.counter.setText(f"{getattr(self.worker, 'image_index', 0) + 1} / {len(self.worker.images)}")
        self.picker.adjustSize()
        self.picker.move(8, self.height() - self.picker.height() - 8)

    def _step(self, delta: int):
        self.worker.step_image(delta)
        self._update_counter()

    def _choose(self):
        dlg = ImagePicker(self.window(), self.worker.images, getattr(self.worker, "image_index", 0))
        if dlg.exec() and dlg.chosen is not None:
            self.worker.select_image(dlg.chosen)
            self._update_counter()

    def resizeEvent(self, e):
        self.off_btn.adjustSize()
        self.off_btn.move(self.width() - self.off_btn.width() - 8, self.height() - self.off_btn.height() - 8)
        if self.picker:
            self._update_counter()
        super().resizeEvent(e)

    def refresh(self):
        f = self.worker.latest_frame
        if f is not None and id(f) != self._frame_id:
            self._frame_id = id(f)
            self._img = to_qimage(f)
        self.update()

    def mousePressEvent(self, e):
        self.clicked.emit(self.worker.cam["id"])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(r, 6, 6)
        p.setClipPath(path)
        p.fillRect(r, QColor("#07080a"))
        s = self.worker.stats
        online = s.get("online") and self._img is not None
        if online:
            p.drawImage(fit_rect(self._img.width(), self._img.height(), r), self._img)
        elif s.get("online"):   # поток есть, первый кадр ещё обрабатывается
            p.setPen(QColor(T.MUTED))
            p.setFont(QFont(T.FONT, 10))
            p.drawText(r, Qt.AlignCenter, "Подключено · обработка первого кадра…")
        else:   # нет сигнала — штриховка
            p.setPen(QPen(QColor("#1d2227"), 10))
            for x in range(-int(r.height()), int(r.width()), 22):
                p.drawLine(x, int(r.height()), x + int(r.height()), 0)
            p.setPen(QColor("#e7eaee"))
            p.setFont(QFont(T.FONT, 12, QFont.DemiBold))
            p.drawText(r.adjusted(0, -24, 0, 0), Qt.AlignCenter, self.worker.cam.get("name", self.worker.cam["id"]))
            p.setPen(QColor(T.MUTED))
            p.setFont(QFont(T.MONO, 9))
            p.drawText(r.adjusted(0, 24, 0, 0), Qt.AlignCenter,
                       f"Нет сигнала · переподключение\n{self.worker.cam['source']}")
        tag(p, 8, 8, self.worker.cam.get("name", self.worker.cam["id"]), T.qc("#0a0c0f", 190))
        if online:
            rt = tag(p, r.width() - 8, 8, f"{s['fps']:.1f} к/с", T.qc("#0a0c0f", 190), right=True)
            if s.get("violating"):
                tag(p, rt.left() - 6, 8, "НАРУШЕНИЕ", T.qc("#d4363e"), right=True)
        if online and s.get("violating"):
            p.setClipping(False)
            p.setPen(QPen(QColor(T.CRIT), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1.5, 1.5, -1.5, -1.5), 6, 6)
        p.end()


class ImagePicker(QDialog):
    """Сетка миниатюр стоп-кадров: двойной щелчок или «Показать» — выбрать кадр для камеры."""

    def __init__(self, parent, paths, current: int):
        super().__init__(parent)
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QListView, QListWidget, QListWidgetItem
        self.setWindowTitle("Выбор кадра")
        self.resize(980, 640)
        self.chosen = None
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setIconSize(QSize(200, 130))
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setSpacing(6)
        for i, p in enumerate(paths):
            pm = QPixmap(str(p)).scaled(200, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            it = QListWidgetItem(QIcon(pm), f"{i + 1}")
            self.list.addItem(it)
        self.list.setCurrentRow(current)
        self.list.itemDoubleClicked.connect(lambda _: self._accept())
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{len(paths)} кадров · двойной щелчок — показать на камере", objectName="muted"))
        row.addStretch()
        cancel, ok = QPushButton("Отмена"), QPushButton("Показать", objectName="primary")
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self._accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)

    def _accept(self):
        self.chosen = self.list.currentRow()
        self.accept()


# ---------- тревога ----------

class Pill(QLabel):
    def set_status(self, status: str):
        text, fg, bg = T.STATUS_STYLE[status]
        self.setText(text)
        self.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:9px; padding:2px 8px;"
                           f" font-family:{T.MONO}; font-size:8pt;")


class AlertCard(QFrame):
    """Карточка тревоги: кадр-доказательство, что/где/кто и кнопки оператора."""
    status_changed = Signal(int, str)
    open_clip = Signal(dict)

    def __init__(self, alert: dict, clips_dir: Path):
        super().__init__()
        self.alert = alert
        sev = T.CRIT if alert["critical"] else T.ACCENT
        self.setObjectName("alertCard")
        self.setStyleSheet(f"QFrame#alertCard {{ background:{T.PANEL}; border:1px solid {T.LINE};"
                           f" border-left:3px solid {sev}; border-radius:6px; }}")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 10, 10, 10)
        row.setSpacing(10)

        thumb = QLabel()
        thumb.setFixedSize(112, 84)
        pm = QPixmap(str(clips_dir / alert["snapshot"])) if alert.get("snapshot") else QPixmap()
        if not pm.isNull():
            thumb.setPixmap(pm.scaled(112, 84, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        thumb.setStyleSheet("background:#07080a; border-radius:4px;")
        row.addWidget(thumb, 0, Qt.AlignTop)

        col = QVBoxLayout()
        col.setSpacing(3)
        title = QLabel(alert["label"])
        title.setWordWrap(True)
        title.setStyleSheet("font-weight:600; font-size:10.5pt;")
        title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        meta_policy = (QSizePolicy.Ignored, QSizePolicy.Preferred)
        t = time.strftime("%H:%M:%S", time.localtime(alert["ts"]))
        zones = ", ".join(alert["zones"]) or "вне зон"
        meta = QLabel(f"{t} · {alert['place']}\n{zones} · {('человек #' + str(alert['track_id'])) if alert['track_id'] else 'место, без человека'}")
        meta.setStyleSheet(f"color:{T.MUTED}; font-size:9pt;")
        meta.setWordWrap(True)
        meta.setSizePolicy(*meta_policy)
        col.addWidget(title)
        col.addWidget(meta)

        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        self.pill = Pill()
        self.btns = []
        for code, text in (("conf", "Подтвердить"), ("sent", "Мастеру"), ("false", "Ложное")):
            b = QPushButton(text)
            if code == "conf":
                b.setObjectName("primary")
            b.clicked.connect(lambda _=False, c=code: self.status_changed.emit(self.alert["id"], c))
            self.btns.append(b)
            self.actions.addWidget(b)
        self.actions.addStretch()
        col.addLayout(self.actions)
        second = QHBoxLayout()   # статус и клип — отдельной строкой, чтобы карточка не вылезала по ширине
        second.setSpacing(6)
        self.clip_btn = QPushButton("▶ Клип 5 с")
        self.clip_btn.clicked.connect(lambda: self.open_clip.emit(self.alert))
        second.addWidget(self.pill)
        second.addWidget(self.clip_btn)
        second.addStretch()
        col.addLayout(second)
        row.addLayout(col, 1)
        self.set_status(alert.get("status", "new"))

    def set_status(self, status: str):
        self.alert["status"] = status
        new = status == "new"
        for b in self.btns:
            b.setVisible(new)
        self.pill.setVisible(not new)
        self.pill.set_status(status)


# ---------- графики ----------

class BarChart(QWidget):
    """Горизонтальные полосы: категория — значение."""

    def __init__(self):
        super().__init__()
        self.items: list[tuple[str, int, bool]] = []   # (подпись, значение, критичное)
        self.setMinimumHeight(120)

    def set_items(self, items):
        self.items = items
        self.setMinimumHeight(max(120, 30 * len(items) + 10))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.items:
            p.setPen(QColor(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "Нарушений пока нет")
            return
        label_w = min(260, int(self.width() * 0.45))
        mx = max(v for _, v, _ in self.items) or 1
        p.setFont(QFont(T.FONT, 9))
        for i, (name, v, crit) in enumerate(self.items):
            y = 6 + i * 30
            p.setPen(QColor(T.FG))
            p.drawText(QRectF(0, y, label_w - 8, 20), Qt.AlignVCenter | Qt.AlignLeft,
                       p.fontMetrics().elidedText(name, Qt.ElideRight, label_w - 8))
            track = QRectF(label_w, y + 5, self.width() - label_w - 44, 12)
            p.fillRect(track, QColor(T.PANEL2))
            p.fillRect(QRectF(track.x(), track.y(), track.width() * v / mx, 12), QColor(T.CRIT if crit else T.ACCENT))
            p.setPen(QColor(T.FG))
            p.setFont(QFont(T.MONO, 9))
            p.drawText(QRectF(self.width() - 40, y, 40, 20), Qt.AlignVCenter | Qt.AlignRight, str(v))
            p.setFont(QFont(T.FONT, 9))


class HeatMap(QWidget):
    """Тепловая карта: камеры (строки) × часы суток (столбцы)."""

    def __init__(self):
        super().__init__()
        self.rows: list[tuple[str, list[int]]] = []
        self.setMinimumHeight(140)

    def set_rows(self, rows):
        self.rows = rows
        self.setMinimumHeight(28 + 26 * max(1, len(rows)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        lw = 150
        cw = (self.width() - lw) / 24
        p.setFont(QFont(T.MONO, 7))
        p.setPen(QColor(T.MUTED))
        for h in range(0, 24, 2):
            p.drawText(QRectF(lw + h * cw, 0, cw * 2, 14), Qt.AlignLeft, f"{h:02d}")
        mx = max([max(v) for _, v in self.rows] + [1])
        for i, (name, vals) in enumerate(self.rows):
            y = 18 + i * 26
            p.setPen(QColor(T.FG))
            p.setFont(QFont(T.FONT, 9))
            p.drawText(QRectF(0, y, lw - 8, 22), Qt.AlignVCenter,
                       p.fontMetrics().elidedText(name, Qt.ElideRight, lw - 8))
            for h, n in enumerate(vals):
                c = T.qc(T.ACCENT, int(30 + 225 * n / mx) if n else 18)
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawRoundedRect(QRectF(lw + h * cw + 1, y + 2, cw - 2, 18), 2, 2)


# ---------- редактор зон ----------

class ZoneEditor(QWidget):
    """Кадр камеры + зоны. Левый клик — добавить точку новой зоны, правый — убрать последнюю."""
    points_changed = Signal(int)

    ZONE_COLORS = {"restricted": T.CRIT, "barrier": T.CRIT, "machine": T.INFO, "ppe": T.ACCENT, "height": T.ACCENT}

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.img: QImage | None = None
        self.zones: list[dict] = []
        self.points: list[tuple[float, float]] = []
        self.setCursor(Qt.CrossCursor)

    def set_frame(self, bgr, zones):
        self.img = to_qimage(bgr) if bgr is not None else None
        self.zones = zones
        self.update()

    def clear_points(self):
        self.points = []
        self.points_changed.emit(0)
        self.update()

    def _img_rect(self) -> QRectF:
        if self.img is None:
            return QRectF(self.rect())
        return fit_rect(self.img.width(), self.img.height(), QRectF(self.rect()))

    def mousePressEvent(self, e):
        r = self._img_rect()
        if e.button() == Qt.RightButton and self.points:
            self.points.pop()
        elif e.button() == Qt.LeftButton and r.contains(e.position()):
            self.points.append(((e.position().x() - r.x()) / r.width(), (e.position().y() - r.y()) / r.height()))
        self.points_changed.emit(len(self.points))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#07080a"))
        r = self._img_rect()
        if self.img is None:
            p.setPen(QColor(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "Нет кадра с этой камеры")
            return
        p.drawImage(r, self.img)
        to_px = lambda x, y: QPointF(r.x() + x * r.width(), r.y() + y * r.height())
        for z in self.zones:
            c = self.ZONE_COLORS.get(z["type"], T.ACCENT)
            poly = QPolygonF([to_px(x, y) for x, y in z["polygon"]])
            p.setPen(QPen(QColor(c), 2))
            p.setBrush(T.qc(c, 40))
            p.drawPolygon(poly)
            tag(p, poly[0].x() + 4, poly[0].y() + 4, z["name"], T.qc("#0a0c0f", 190))
        if self.points:
            poly = QPolygonF([to_px(x, y) for x, y in self.points])
            p.setPen(QPen(QColor(T.ACCENT), 2, Qt.DashLine))
            p.setBrush(T.qc(T.ACCENT, 50))
            p.drawPolygon(poly) if len(self.points) > 2 else p.drawPolyline(poly)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(T.ACCENT))
            for pt in poly:
                p.drawEllipse(pt, 4, 4)


# ---------- проигрыватель клипа ----------

class ClipPlayer(QDialog):
    """5-секундный клип-доказательство. OpenCV читает mp4, Qt показывает кадры."""

    def __init__(self, parent, clip_path: Path, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        self.cap = cv2.VideoCapture(str(clip_path))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 10
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lay = QVBoxLayout(self)
        self.view = QLabel(alignment=Qt.AlignCenter)
        self.view.setStyleSheet("background:#07080a; border-radius:6px;")
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lay.addWidget(self.view, 1)
        bar = QHBoxLayout()
        self.play = QPushButton("Пауза")
        self.play.clicked.connect(self.toggle)
        self.pos = QLabel()
        self.pos.setStyleSheet(f"color:{T.MUTED}; font-family:{T.MONO};")
        bar.addWidget(self.play)
        bar.addWidget(self.pos)
        bar.addStretch()
        bar.addWidget(QLabel(clip_path.name, objectName="muted"))
        lay.addLayout(bar)
        self.timer = QTimer(self, interval=int(1000 / fps), timeout=self.next_frame)
        self.timer.start()
        self.next_frame()

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("Смотреть")
        else:
            self.timer.start()
            self.play.setText("Пауза")

    def next_frame(self):
        ok, frame = self.cap.read()
        if not ok:   # клип короткий — крутим по кругу
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return
        n = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.pos.setText(f"кадр {n}/{self.total}")
        pm = QPixmap.fromImage(to_qimage(frame))
        self.view.setPixmap(pm.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, e):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(e)

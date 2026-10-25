"""Главное окно «Пост контроля ОТ»: мониторинг, журнал, статистика, зоны и правила, производительность."""
from __future__ import annotations

import csv
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from PySide6.QtCore import QDate, QPointF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDateEdit, QFileDialog,
                               QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from app.config import save_zones
from app.rules import VIOLATION_LABELS

from . import theme as T
from .widgets import AlertCard, BarChart, CameraTile, ClipPlayer, HeatMap, Pill, ZoneEditor

PPE_NAMES = {"helmet": "Каска", "vest": "Жилет", "mask": "Респиратор", "glasses": "Защитные очки", "gloves": "Перчатки"}
ZONE_TYPES = [("restricted", "Опасная зона — вход запрещён"), ("machine", "Рабочее место станка"),
              ("ppe", "Зона обязательных СИЗ"), ("barrier", "Ограждение — перелезать нельзя"),
              ("height", "Работа на высоте — нужна привязь")]


def h2(text: str) -> QLabel:
    return QLabel(text, objectName="h2")


def card(*widgets, layout=None) -> QFrame:
    f = QFrame(objectName="card")
    lay = layout or QVBoxLayout()
    lay.setContentsMargins(14, 14, 14, 14)
    lay.setSpacing(10)
    for w in widgets:
        lay.addWidget(w)
    f.setLayout(lay)
    return f


class Brand(QWidget):
    """Знак поста: жёлто-чёрная сигнальная разметка."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(28, 28)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setClipRect(self.rect())
        p.fillRect(self.rect(), QColor(T.ACCENT))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#111"))
        for x in range(-28, 28, 9):
            p.drawPolygon(QPolygonF([QPointF(x, 28), QPointF(x + 4, 28), QPointF(x + 32, 0), QPointF(x + 28, 0)]))


# ---------------------------------------------------------------- мониторинг

class MonitorPage(QWidget):
    new_alert = Signal(dict)

    def __init__(self, rt, main):
        super().__init__(objectName="page")
        self.rt, self.main = rt, main
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(10)
        bar = QHBoxLayout()
        self.cam_btn = QPushButton("Камеры ▾")
        self.cam_btn.setToolTip("Включить или выключить камеры и видео")
        self.cam_menu = QMenu(self)
        self.cam_actions = {}
        for w in rt.workers:   # меню со всеми камерами из конфига, галочка = включена
            act = QAction(w.cam.get("name", w.cam["id"]), self, checkable=True, checked=w.cam.get("enabled", True))
            act.toggled.connect(lambda on, cid=w.cam["id"]: self.set_camera(cid, on))
            self.cam_menu.addAction(act)
            self.cam_actions[w.cam["id"]] = act
        self.cam_menu.addSeparator()
        self.cam_menu.addAction("Включить все", lambda: [a.setChecked(True) for a in self.cam_actions.values()])
        self.cam_menu.addAction("Выключить все", lambda: [a.setChecked(False) for a in self.cam_actions.values()])
        self.cam_btn.setMenu(self.cam_menu)
        self.cam_count = QLabel(objectName="muted")
        bar.addWidget(self.cam_btn)
        bar.addWidget(self.cam_count)
        bar.addStretch()
        left.addLayout(bar)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        self.no_cams = QLabel("Все камеры выключены.\nВключите нужные кнопкой «Камеры» вверху.",
                              objectName="muted", alignment=Qt.AlignCenter)
        self.tiles = []
        for w in rt.workers:
            t = CameraTile(w)
            t.clicked.connect(self.toggle_focus)
            t.power_off.connect(lambda cid: self.cam_actions[cid].setChecked(False))
            self.tiles.append(t)
        self.focused = None
        self.relayout()
        left.addWidget(self.grid_host, 1)
        lw = QWidget()
        lw.setLayout(left)
        root.addWidget(lw, 3)

        side = QVBoxLayout()
        head = QHBoxLayout()
        head.addWidget(h2("Тревоги смены"))
        head.addStretch()
        head.addWidget(QLabel("одно нарушение — одна тревога", objectName="muted"))
        side.addLayout(head)
        self.feed_host = QWidget(objectName="feedHost")
        self.feed = QVBoxLayout(self.feed_host)
        self.feed.setContentsMargins(0, 0, 4, 0)
        self.feed.setSpacing(8)
        self.empty = QLabel("Нарушений пока нет.\nКак только система подтвердит нарушение,\nоно появится здесь с кадром и клипом.",
                            objectName="muted", alignment=Qt.AlignCenter)
        self.feed.addWidget(self.empty)
        self.feed.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.feed_host)
        side.addWidget(scroll, 1)
        sw = QWidget()
        sw.setLayout(side)
        sw.setMinimumWidth(440)
        root.addWidget(sw, 2)

        self.cards: dict[int, AlertCard] = {}
        self.last_id = 0
        for a in reversed(rt.store.alerts(limit=30)):   # тревоги текущей смены после перезапуска
            self.add_card(a, notify=False)

        QTimer(self, interval=100, timeout=self.refresh_tiles).start()
        QTimer(self, interval=1000, timeout=self.poll_alerts).start()

    def toggle_focus(self, cam_id: str):
        """Щелчок по камере — развернуть её на всю область, повторный — вернуть сетку."""
        self.focused = None if self.focused == cam_id else cam_id
        self.relayout()

    def set_camera(self, cam_id: str, on: bool):
        self.rt.set_enabled(cam_id, on)
        if not on and self.focused == cam_id:
            self.focused = None
        self.relayout()
        name = self.cam_actions[cam_id].text()
        self.main.statusBar().showMessage(f"Камера «{name}» {'включена' if on else 'выключена'}", 5000)

    def relayout(self):
        """Сетка только из включённых камер: 1 — во всю область, до 4 — по 2 в ряд, больше — по 3."""
        for t in self.tiles:
            self.grid.removeWidget(t)
            t.hide()
        self.grid.removeWidget(self.no_cams)
        self.no_cams.hide()
        on = [t for t in self.tiles if t.worker.cam.get("enabled", True)]
        shown = [t for t in on if self.focused is None or t.worker.cam["id"] == self.focused]
        cols = 1 if len(shown) == 1 else 3 if len(shown) > 4 else 2
        for i, t in enumerate(shown):
            self.grid.addWidget(t, i // cols, i % cols)
            t.show()
        if not shown:
            self.grid.addWidget(self.no_cams, 0, 0)
            self.no_cams.show()
        self.cam_count.setText(f"включено {len(on)} из {len(self.tiles)}")

    def refresh_tiles(self):
        for t, w in zip(self.tiles, self.rt.workers):   # включённая заново камера — это новый поток
            t.worker = w
        for t in self.tiles:
            if t.isVisible():
                t.refresh()

    def poll_alerts(self):
        for a in reversed(self.rt.store.alerts(after_id=self.last_id, limit=50)):
            self.add_card(a, notify=True)

    def add_card(self, a: dict, notify: bool):
        self.last_id = max(self.last_id, a["id"])
        self.empty.hide()
        c = AlertCard(a, self.rt.clips_dir)
        c.status_changed.connect(self.main.set_status)
        c.open_clip.connect(self.main.open_clip)
        self.cards[a["id"]] = c
        self.feed.insertWidget(0, c)
        if notify:
            self.new_alert.emit(a)


# ---------------------------------------------------------------- журнал

class JournalPage(QWidget):
    COLS = ["Время", "Нарушение", "Камера / участок", "Зона", "Человек", "Клип", "Статус"]

    def __init__(self, rt, main):
        super().__init__(objectName="page")
        self.rt, self.main = rt, main
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        f = QHBoxLayout()
        self.f_type = QComboBox()
        self.f_type.addItem("Все типы", "")
        for k, v in VIOLATION_LABELS.items():
            self.f_type.addItem(v, k)
        self.f_cam = QComboBox()
        self.f_cam.addItem("Все камеры", "")
        for w in rt.workers:
            self.f_cam.addItem(w.cam.get("name", w.cam["id"]), w.cam["id"])
        self.f_status = QComboBox()
        self.f_status.addItem("Любой статус", "")
        for k, (text, _, _) in T.STATUS_STYLE.items():
            self.f_status.addItem(text, k)
        self.f_all_days = QCheckBox("Все дни")
        self.f_all_days.setChecked(True)
        self.f_date = QDateEdit(QDate.currentDate(), calendarPopup=True)
        self.f_date.setDisplayFormat("dd.MM.yyyy")
        for w in (self.f_type, self.f_cam, self.f_status):
            w.currentIndexChanged.connect(self.reload)
        self.f_date.dateChanged.connect(self.reload)
        self.f_all_days.toggled.connect(self.reload)
        export = QPushButton("Выгрузить CSV")
        export.clicked.connect(self.export_csv)
        for w in (self.f_type, self.f_cam, self.f_status, self.f_date, self.f_all_days):
            f.addWidget(w)
        f.addStretch()
        self.count = QLabel(objectName="muted")
        f.addWidget(self.count)
        f.addWidget(export)
        lay.addLayout(f)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, _: self.main.open_clip(self.rows[r]))
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.menu)
        lay.addWidget(self.table, 1)
        lay.addWidget(QLabel("Двойной щелчок — клип-доказательство. Правый щелчок — изменить статус.", objectName="muted"))
        self.rows: list[dict] = []

    def reload(self):
        day = None if self.f_all_days.isChecked() else self.f_date.date().toString("yyyy-MM-dd")
        self.f_date.setEnabled(not self.f_all_days.isChecked())
        self.rows = self.rt.store.alerts(limit=2000, type_=self.f_type.currentData(), camera=self.f_cam.currentData(),
                                         status=self.f_status.currentData(), day=day)
        self.table.setRowCount(len(self.rows))
        for i, a in enumerate(self.rows):
            vals = [time.strftime("%d.%m %H:%M:%S", time.localtime(a["ts"])), a["label"], a["place"],
                    ", ".join(a["zones"]) or "—", (f"#{a['track_id']}" if a["track_id"] else "место"), "▶ 5 с" if a["clip"] else "пишется…", ""]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 1 and a["critical"]:
                    it.setForeground(QColor(T.CRIT))
                self.table.setItem(i, j, it)
            pill = Pill()
            pill.set_status(a["status"])
            self.table.setCellWidget(i, 6, pill)
        self.count.setText(f"{len(self.rows)} записей")

    def menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0:
            return
        m = QMenu(self)
        for k, (text, _, _) in T.STATUS_STYLE.items():
            act = QAction(text, m)
            act.triggered.connect(lambda _=False, s=k: (self.main.set_status(self.rows[r]["id"], s), self.reload()))
            m.addAction(act)
        m.exec(self.table.viewport().mapToGlobal(pos))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Выгрузить журнал", "журнал_нарушений.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig — чтобы Excel открыл кириллицу
            w = csv.writer(f, delimiter=";")
            w.writerow(["id", "время", "нарушение", "камера", "участок", "зоны", "человек", "статус", "клип", "кадр"])
            for a in self.rows:
                w.writerow([a["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a["ts"])), a["label"],
                            a["camera"], a["place"], ", ".join(a["zones"]), a["track_id"],
                            T.STATUS_STYLE[a["status"]][0], a["clip"] or "", a["snapshot"]])
        self.main.statusBar().showMessage(f"Журнал выгружен: {path}", 5000)


# ---------------------------------------------------------------- статистика

class StatsPage(QWidget):
    def __init__(self, rt):
        super().__init__(objectName="page")
        self.rt = rt
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(16)
        kp = QHBoxLayout()
        self.kpis = {}
        for key, label in (("total", "Нарушений сегодня"), ("conf", "Подтверждено оператором"),
                           ("false", "Ложных тревог"), ("clips", "Клипов-доказательств")):
            f = QFrame(objectName="kpi")
            v = QVBoxLayout(f)
            v.setContentsMargins(14, 12, 14, 12)
            v.addWidget(QLabel(label, objectName="muted"))
            val = QLabel("0", objectName="kpiValue")
            sub = QLabel("", objectName="muted")
            v.addWidget(val)
            v.addWidget(sub)
            self.kpis[key] = (val, sub)
            kp.addWidget(f)
        lay.addLayout(kp)
        row = QHBoxLayout()
        self.by_type = BarChart()
        self.heat = HeatMap()
        row.addWidget(card(h2("По типам нарушений (все дни)"), self.by_type), 3)
        row.addWidget(card(h2("Камеры × часы суток"), self.heat,
                           QLabel("Где и когда нарушают чаще — для решений по сменам и участкам.", objectName="muted")), 4)
        lay.addLayout(row)
        lay.addStretch()

    def reload(self):
        rows = self.rt.store.alerts(limit=100000)
        today = time.strftime("%Y-%m-%d")
        today_rows = [a for a in rows if time.strftime("%Y-%m-%d", time.localtime(a["ts"])) == today]
        st = Counter(a["status"] for a in rows)
        reviewed = st["conf"] + st["false"] + st["sent"]
        self.kpis["total"][0].setText(str(len(today_rows)))
        self.kpis["total"][1].setText(f"всего в журнале: {len(rows)}")
        self.kpis["conf"][0].setText(f"{(st['conf'] + st['sent']) / reviewed * 100:.0f}%" if reviewed else "—")
        self.kpis["conf"][1].setText(f"проверено {reviewed} из {len(rows)}")
        self.kpis["false"][0].setText(str(st["false"]))
        self.kpis["false"][1].setText("кадры ушли в папку дообучения" if st["false"] else "")
        clips = list(self.rt.clips_dir.glob("*.mp4"))
        mb = sum(p.stat().st_size for p in clips) / 1e6
        self.kpis["clips"][0].setText(str(len(clips)))
        self.kpis["clips"][1].setText(f"{mb:.1f} МБ на диске")

        by_type = Counter((a["label"], bool(a["critical"])) for a in rows)
        self.by_type.set_items([(lbl, n, crit) for (lbl, crit), n in by_type.most_common()])
        grid = defaultdict(lambda: [0] * 24)
        for a in rows:
            grid[a["place"]][time.localtime(a["ts"]).tm_hour] += 1
        names = {w.place for w in self.rt.workers}
        self.heat.set_rows([(n, grid[n]) for n in sorted(names | set(grid))])


# ---------------------------------------------------------------- зоны

class ZonesPage(QWidget):
    def __init__(self, rt, main):
        super().__init__(objectName="page")
        self.rt, self.main = rt, main
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(16)

        left = QVBoxLayout()
        top = QHBoxLayout()
        self.cam = QComboBox()
        for w in rt.workers:
            self.cam.addItem(w.cam.get("name", w.cam["id"]), w.cam["id"])
        self.cam.currentIndexChanged.connect(self.load_camera)
        top.addWidget(self.cam)
        snap = QPushButton("Обновить кадр")
        snap.clicked.connect(self.load_camera)
        top.addWidget(snap)
        top.addStretch()
        top.addWidget(QLabel("левый щелчок — точка зоны, правый — убрать последнюю", objectName="muted"))
        left.addLayout(top)
        self.editor = ZoneEditor()
        self.editor.points_changed.connect(lambda n: self.save_btn.setText(f"Сохранить зону ({n} точек)"))
        left.addWidget(self.editor, 1)
        lay.addLayout(left, 3)

        form = QVBoxLayout()
        form.setSpacing(8)
        form.addWidget(h2("Новая зона"))
        self.name = QLineEdit("Ограждение патрона")
        self.ztype = QComboBox()
        for k, v in ZONE_TYPES:
            self.ztype.addItem(v, k)
        form.addWidget(QLabel("Название", objectName="muted"))
        form.addWidget(self.name)
        form.addWidget(QLabel("Тип зоны", objectName="muted"))
        form.addWidget(self.ztype)
        form.addWidget(QLabel("Обязательно в зоне", objectName="muted"))
        self.req = {k: QCheckBox(v) for k, v in PPE_NAMES.items()}
        grid = QGridLayout()
        for i, cb in enumerate(self.req.values()):
            grid.addWidget(cb, i // 2, i % 2)
        form.addLayout(grid)
        form.addWidget(QLabel("Запрещено в зоне", objectName="muted"))
        self.forb = {"gloves": QCheckBox("Перчатки (вращающиеся части)")}
        for cb in self.forb.values():
            form.addWidget(cb)
        self.save_btn = QPushButton("Сохранить зону (0 точек)", objectName="primary")
        self.save_btn.clicked.connect(self.save_zone)
        clear = QPushButton("Сбросить точки")
        clear.clicked.connect(self.editor.clear_points)
        b = QHBoxLayout()
        b.addWidget(self.save_btn)
        b.addWidget(clear)
        form.addLayout(b)
        form.addWidget(h2("Зоны камеры"))
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        form.addWidget(self.list, 1)
        delete = QPushButton("Удалить выбранную зону")
        delete.clicked.connect(self.delete_zone)
        form.addWidget(delete)
        form.addWidget(QLabel("Правила применяются сразу, без перезапуска.\nСохраняются в data/zones.json.",
                              objectName="muted"))
        fw = QWidget()
        fw.setLayout(form)
        fw.setFixedWidth(330)
        lay.addWidget(fw)

    def worker(self):
        return self.rt.workers[self.cam.currentIndex()]

    def load_camera(self):
        w = self.worker()
        self.editor.set_frame(w.latest_raw, w.cam["zones"])
        self.list.clear()
        for z in w.cam["zones"]:
            extra = []
            if z.get("require"):
                extra.append("обязательно: " + ", ".join(PPE_NAMES.get(x, x) for x in z["require"]))
            if z.get("forbid"):
                extra.append("запрещено: " + ", ".join(PPE_NAMES.get(x, x) for x in z["forbid"]))
            kind = dict(ZONE_TYPES).get(z["type"], z["type"])
            QListWidgetItem(f"{z['name']}\n{kind}" + ("\n" + "; ".join(extra) if extra else ""), self.list)

    def save_zone(self):
        pts = self.editor.points
        if len(pts) < 3:
            QMessageBox.information(self, "Зона", "Поставьте на кадре хотя бы 3 точки.")
            return
        z = {"name": self.name.text().strip() or "Зона", "type": self.ztype.currentData(),
             "polygon": [[round(x, 4), round(y, 4)] for x, y in pts]}
        if z["type"] != "restricted":
            z["require"] = [k for k, cb in self.req.items() if cb.isChecked()]
            z["forbid"] = [k for k, cb in self.forb.items() if cb.isChecked()]
        self.worker().cam["zones"].append(z)   # конвейер камеры видит зону со следующего кадра
        save_zones(self.rt.cfg)
        self.editor.clear_points()
        self.load_camera()
        self.main.statusBar().showMessage(f"Зона «{z['name']}» сохранена и уже работает", 5000)

    def delete_zone(self):
        i = self.list.currentRow()
        if i < 0:
            return
        z = self.worker().cam["zones"].pop(i)
        save_zones(self.rt.cfg)
        self.load_camera()
        self.main.statusBar().showMessage(f"Зона «{z['name']}» удалена", 5000)


# ---------------------------------------------------------------- производительность

class BenchThread(QThread):
    done = Signal(dict)

    def __init__(self, rt):
        super().__init__()
        self.rt = rt

    def run(self):
        frame = next((w.latest_raw for w in self.rt.workers if w.latest_raw is not None), None)
        if frame is None:
            self.done.emit({"error": "нет кадров"})
            return
        times = []
        for _ in range(25):
            t0 = time.perf_counter()
            with self.rt.lock:
                self.rt.backend.detect(frame)
            times.append((time.perf_counter() - t0) * 1000)
        times = sorted(times[5:])   # первые прогоны — прогрев
        self.done.emit({"mean": statistics.mean(times), "p95": times[int(len(times) * .95) - 1]})


class PerfPage(QWidget):
    COLS = ["Камера", "Источник", "Состояние", "Кадров/с", "Сеть, мс", "Людей", "Нарушают", "Кусков 5 с", "Сохранено"]

    def __init__(self, rt):
        super().__init__(objectName="page")
        self.rt = rt
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        info = QLabel(f"Инференс: <b>{rt.model_name}</b>. На стенде LinQ — <b>model.backend: linq</b> в config.yaml: "
                      f"сеть выполняется на TPU, подготовка кадра и разбор результата — на CPU.")
        info.setWordWrap(True)
        lay.addWidget(card(h2("Ускоритель и модель"), info))
        self.table = QTableWidget(len(rt.workers), len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        lay.addWidget(self.table)
        b = QHBoxLayout()
        self.bench_btn = QPushButton("Замерить скорость модели", objectName="primary")
        self.bench_btn.clicked.connect(self.bench)
        self.bench_res = QLabel(objectName="muted")
        b.addWidget(self.bench_btn)
        b.addWidget(self.bench_res, 1)
        lay.addLayout(b)
        lay.addStretch()
        QTimer(self, interval=1000, timeout=self.reload).start()

    def reload(self):
        if not self.isVisible():
            return
        for i, w in enumerate(self.rt.workers):
            s = w.stats
            src = str(w.cam["source"])
            vals = [w.cam.get("name", w.cam["id"]), src if "://" in src else Path(src).name,
                    "выключена" if not w.cam.get("enabled", True) else "онлайн" if s["online"] else "нет сигнала",
                    f"{s['fps']:.1f}", f"{s['infer_ms']:.0f}",
                    str(s["people"]), str(s["violating"]), str(s["segments"]), str(s["incidents"])]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 2:
                    it.setForeground(QColor(T.MUTED if not w.cam.get("enabled", True) else T.OK if s["online"] else T.CRIT))
                self.table.setItem(i, j, it)

    def bench(self):
        self.bench_btn.setEnabled(False)
        self.bench_res.setText("Замер: 25 прогонов модели на текущем кадре…")
        self.th = BenchThread(self.rt)
        self.th.done.connect(self.bench_done)
        self.th.start()

    def bench_done(self, r):
        self.bench_btn.setEnabled(True)
        if "error" in r:
            self.bench_res.setText(f"Не удалось: {r['error']}")
            return
        cams = int((1000 / r["mean"]) // 3)
        self.bench_res.setText(f"{self.rt.model_name}: {r['mean']:.1f} мс на кадр (p95 {r['p95']:.1f}), "
                               f"{1000 / r['mean']:.1f} к/с — хватит на {cams} камер(ы) по 3 к/с")


# ---------------------------------------------------------------- отчёты

class ReportThread(QThread):
    done = Signal(str, str)

    def __init__(self, rt, since: float):
        super().__init__()
        self.rt, self.since = rt, since

    def run(self):
        from app.reports import make_report
        try:
            text, source = make_report(self.rt.cfg, self.rt.store.alerts(limit=100000), self.since)
        except Exception as e:   # отчёт не должен ронять приложение
            text, source = f"Не удалось построить отчёт: {e}", "ошибка"
        self.done.emit(text, source)


class ReportsPage(QWidget):
    PERIODS = [("Последние 24 часа", 24 * 3600), ("Сегодня", "today"), ("Последние 7 дней", 7 * 24 * 3600),
               ("Последние 30 дней", 30 * 24 * 3600), ("Всё время", 0)]

    def __init__(self, rt, main):
        super().__init__(objectName="page")
        from PySide6.QtWidgets import QTextBrowser
        from app.reports import credentials_path, load_api_key
        self.rt, self.main = rt, main
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        top = QHBoxLayout()
        self.period = QComboBox()
        for text, _ in self.PERIODS:
            self.period.addItem(text)
        self.btn = QPushButton("Сформировать отчёт", objectName="primary")
        self.btn.clicked.connect(self.generate)
        self.save_btn = QPushButton("Сохранить .md")
        self.save_btn.clicked.connect(self.save)
        self.save_btn.setEnabled(False)
        top.addWidget(QLabel("Период:", objectName="muted"))
        top.addWidget(self.period)
        top.addWidget(self.btn)
        top.addWidget(self.save_btn)
        top.addStretch()
        path = credentials_path(rt.cfg)
        has_key = bool(load_api_key(path))   # только проверка наличия — значение нигде не показывается
        model = rt.cfg.get("reports", {}).get("model", "claude-opus-5-5")
        self.info = QLabel(f"Текст пишет Claude ({model}), ключ: {path.name}" if has_key else
                           f"Ключ Claude не найден ({path}) — отчёт будет построен по шаблону", objectName="muted")
        top.addWidget(self.info)
        lay.addLayout(top)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.setStyleSheet(f"QTextBrowser {{ background: {T.PANEL}; border: 1px solid {T.LINE};"
                                f" border-radius: 6px; padding: 14px; font-size: 10.5pt; }}")
        self.view.setMarkdown("Выберите период и нажмите **«Сформировать отчёт»**.\n\n"
                              "В Claude уходят только сводные цифры и список событий (время, тип, участок, зона, статус). "
                              "Кадры, клипы и данные о людях не отправляются.")
        lay.addWidget(self.view, 1)
        self.text = ""

    def _since(self) -> float:
        val = self.PERIODS[self.period.currentIndex()][1]
        if val == "today":
            t = time.localtime()
            return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
        return time.time() - val if val else 0.0

    def generate(self):
        self.btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.view.setMarkdown("Собираю сводку и жду ответ Claude… обычно это 10–60 секунд.")
        self.th = ReportThread(self.rt, self._since())
        self.th.done.connect(self.done)
        self.th.start()

    def done(self, text: str, source: str):
        self.text = text
        self.view.setMarkdown(text)
        self.btn.setEnabled(True)
        self.save_btn.setEnabled(bool(text))
        self.main.statusBar().showMessage(
            "Отчёт готов — текст составил Claude" if source == "claude" else f"Отчёт готов ({source})", 6000)

    def save(self):
        d = Path(self.rt.cfg.get("reports", {}).get("dir", "data/reports"))
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"отчёт_ОТ_{time.strftime('%Y%m%d_%H%M')}.md"
        p.write_text(self.text, encoding="utf-8")
        self.main.statusBar().showMessage(f"Отчёт сохранён: {p}", 8000)


# ---------------------------------------------------------------- окно

class MainWindow(QMainWindow):
    PAGES = [("monitor", "Мониторинг"), ("journal", "Журнал"), ("stats", "Статистика"),
             ("zones", "Зоны и правила"), ("reports", "Отчёты"), ("perf", "Производительность")]

    def __init__(self, rt):
        super().__init__()
        self.rt = rt
        self.setWindowTitle("Пост контроля ОТ")
        self.resize(1440, 860)
        root = QWidget(objectName="root")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        nav = QFrame(objectName="nav")
        nav.setFixedWidth(210)
        nv = QVBoxLayout(nav)
        nv.setContentsMargins(10, 14, 10, 14)
        nv.setSpacing(4)
        brand = QHBoxLayout()
        brand.addWidget(Brand())
        bt = QVBoxLayout()
        bt.setSpacing(0)
        bt.addWidget(QLabel("Пост контроля ОТ", objectName="brand"))
        bt.addWidget(QLabel("видеоаналитика · LinQ", objectName="brandSub"))
        brand.addLayout(bt)
        nv.addLayout(brand)
        nv.addSpacing(12)
        self.nav_btns = {}
        for key, text in self.PAGES:
            b = QPushButton(text, objectName="navBtn", checkable=True)
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav_btns[key] = b
            nv.addWidget(b)
        nv.addStretch()
        self.nav_info = QLabel(objectName="muted")
        self.nav_info.setWordWrap(True)
        nv.addWidget(self.nav_info)
        h.addWidget(nav)

        main = QVBoxLayout()
        main.setSpacing(0)
        top = QFrame(objectName="topbar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(18, 10, 18, 10)
        self.title = QLabel(objectName="title")
        tl.addWidget(self.title)
        tl.addStretch()
        self.chip_model = QLabel(rt.model_name, objectName="chipAccent")
        self.chip_streams = QLabel(objectName="chip")
        self.clock = QLabel(objectName="chip")
        for w in (self.chip_model, self.chip_streams, self.clock):
            tl.addWidget(w)
        main.addWidget(top)

        self.stack = QStackedWidget()
        self.pages = {"monitor": MonitorPage(rt, self), "journal": JournalPage(rt, self), "stats": StatsPage(rt),
                      "zones": ZonesPage(rt, self), "reports": ReportsPage(rt, self), "perf": PerfPage(rt)}
        for p in self.pages.values():
            self.stack.addWidget(p)
        main.addWidget(self.stack, 1)
        mw = QWidget()
        mw.setLayout(main)
        h.addWidget(mw, 1)
        self.setCentralWidget(root)

        self.pages["monitor"].new_alert.connect(self.on_new_alert)
        if rt.model_note:
            self.statusBar().showMessage(rt.model_note)
        QTimer(self, interval=1000, timeout=self.tick).start()
        self.tick()
        self.show_page("monitor")

    def show_page(self, key: str):
        for k, b in self.nav_btns.items():
            b.setChecked(k == key)
        self.stack.setCurrentWidget(self.pages[key])
        self.title.setText(dict(self.PAGES)[key])
        if key in ("journal", "stats"):
            self.pages[key].reload()
        if key == "zones":
            self.pages[key].load_camera()

    def tick(self):
        online = sum(1 for w in self.rt.active if w.stats["online"])
        self.clock.setText(time.strftime("%d.%m.%Y  %H:%M:%S"))
        self.chip_streams.setText(f"Камер онлайн: {online} из {len(self.rt.active)}")
        new = sum(1 for c in self.pages["monitor"].cards.values() if c.alert["status"] == "new")
        self.nav_btns["monitor"].setText(f"Мониторинг   ● {new}" if new else "Мониторинг")
        self.nav_info.setText(f"Модель: {self.rt.model_name}\n"
                              f"Клипы: {self.rt.clips_dir.parent.name}/{self.rt.clips_dir.name}")

    def on_new_alert(self, a: dict):
        t = time.strftime("%H:%M:%S", time.localtime(a["ts"]))
        self.statusBar().showMessage(f"{t}  Новое нарушение: {a['label']} — {a['place']}", 15000)
        QApplication.alert(self)   # мигнуть кнопкой на панели задач
        if a["critical"]:
            QApplication.beep()

    def set_status(self, alert_id: int, status: str):
        """Решение оператора. «Ложное» — исходный кадр уходит в папку дообучения модели."""
        self.rt.store.set_status(alert_id, status)
        card_ = self.pages["monitor"].cards.get(alert_id)
        if card_:
            card_.set_status(status)
        if status == "false":
            a = self.rt.store.alert(alert_id)
            if a and a.get("raw") and (self.rt.clips_dir / a["raw"]).exists():
                self.rt.retrain_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.rt.clips_dir / a["raw"], self.rt.retrain_dir / Path(a["raw"]).name)
            self.statusBar().showMessage("Отмечено как ложное: кадр сохранён в data/retrain для дообучения", 6000)
        else:
            self.statusBar().showMessage(f"Статус: {T.STATUS_STYLE[status][0]}", 4000)

    def open_clip(self, a: dict):
        fresh = self.rt.store.alert(a["id"]) or a   # клип мог дописаться после того, как карточка появилась
        if fresh.get("clip") and (self.rt.clips_dir / fresh["clip"]).exists():
            ClipPlayer(self, self.rt.clips_dir / fresh["clip"], f"{fresh['label']} — {fresh['place']}").exec()
        elif fresh.get("clip"):
            self.statusBar().showMessage("Клип удалён по лимиту места на диске — кадр остался в журнале", 6000)
        else:
            self.statusBar().showMessage("Клип ещё записывается — будет готов через несколько секунд", 5000)

    def closeEvent(self, e):
        self.statusBar().showMessage("Остановка камер…")
        self.rt.stop()
        super().closeEvent(e)

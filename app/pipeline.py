"""Обработка одной камеры: кадр -> детекция -> трекинг -> правила -> 5-секундные куски -> инциденты.

Видео пишется кусками по segments.seconds секунд. Кусок без подтверждённых нарушений удаляется,
с нарушениями — сохраняется как клип вместе с местом работ. И так по кругу.

Источник камеры: RTSP/веб-камера, видеофайл или набор изображений (файл или папка) —
«имитация стоп-кадров» из требований кейса.
"""
import logging
import threading
import time
from pathlib import Path

import cv2

from .backends import Backend
from .behavior import BehaviorAnalyzer
from .draw import draw
from .events import Debouncer, IncidentStore, Notifier
from .geometry import foot_point, point_in_polygon
from .fire import FireDetector
from .rules import PLACE_ID, VIOLATION_LABELS, evaluate
from .segments import SegmentRecorder
from .tracker import IouTracker

log = logging.getLogger(__name__)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CRITICAL = {"forbidden_gloves", "restricted_zone", "smoking", "fall", "vehicle_proximity", "open_fire",
            "climbing", "no_harness"}   # выделяются в интерфейсе как приоритетные


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, cfg: dict, backend: Backend, backend_lock: threading.Lock,
                 store: IncidentStore, notifier: Notifier, realtime: bool = True, writer=None, pose_model=None):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam, self.cfg = cam, cfg
        self.place = cam.get("place") or cam.get("name") or cam["id"]   # место работ, которое видит камера
        self.backend, self.backend_lock = backend, backend_lock
        self.store, self.notifier = store, notifier
        self.realtime, self.writer = realtime, writer
        self.class_map = cfg["model"]["class_map"]
        self.tracker = IouTracker(**cfg.get("tracker", {}))
        # у камеры могут быть свои обязательные СИЗ: в цеху ЧПУ каска не нужна, на стройплощадке — нужна
        self.rules = {**cfg["rules"], **({"ppe_required": cam["ppe_required"]} if "ppe_required" in cam else {}),
                      **({"forbid": cam["forbid"]} if "forbid" in cam else {})}
        self.behavior = BehaviorAnalyzer(cfg.get("behavior"), pose_model) if cam.get("behavior", True) else None
        fire_cfg = cfg.get("fire") or {}
        self.fire = FireDetector(fire_cfg) if fire_cfg.get("enabled", True) and cam.get("fire", True) else None
        self._media_t = None   # время внутри видеофайла: скорость бега считаем по нему, а не по часам
        r = self.rules
        self.debouncer = Debouncer(r["window_frames"], r["min_ratio"], r["cooldown_sec"])
        seg = cfg.get("segments", {})
        self.recorder = SegmentRecorder(cam["id"], seg.get("dir", "data/clips"), cam["fps"],
                                        seg.get("seconds", 5), seg.get("max_total_mb"),
                                        seg.get("min_violation_sec", 1.0))

        self.stop_event = threading.Event()
        self.latest_jpeg: bytes | None = None
        self.stats = {"fps": 0.0, "infer_ms": 0.0, "frames": 0, "people": 0, "violating": 0,
                      "segments": 0, "incidents": 0, "online": False}
        self._latest_frame = None   # для сетевых камер: самый свежий кадр от потока-читателя
        self.latest_frame = None    # последний размеченный кадр (BGR) — его показывает десктопное приложение
        self.latest_raw = None      # тот же кадр без разметки — для редактора зон
        self._pending_alerts: list[int] = []   # тревоги, ждущие клипа своего 5-секундного куска
        self.snap_dir = Path(seg.get("dir", "data/clips")) / "alerts"
        self.snap_dir.mkdir(parents=True, exist_ok=True)

    # ---------- чтение видео ----------

    def _open(self):
        src = str(self.cam["source"])
        return cv2.VideoCapture(int(src) if src.isdigit() else src)

    def _is_file(self) -> bool:
        return Path(str(self.cam["source"])).is_file()

    def _images(self) -> list[Path]:
        """Источник — картинка или папка картинок (стоп-кадры)."""
        p = Path(str(self.cam["source"]))
        if p.is_dir():
            return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        return [p] if p.is_file() and p.suffix.lower() in IMG_EXT else []

    def _videos(self) -> list[Path]:
        """Плейлист: у камеры задано files: "*.mp4" — все подходящие видео из папки source, по очереди."""
        pattern = self.cam.get("files")
        p = Path(str(self.cam["source"]))
        return sorted(p.glob(pattern)) if pattern and p.is_dir() else []

    def run(self):
        try:
            if self._videos():
                self._run_playlist()
            elif self._images():
                self._run_images()
            elif self._is_file():
                self._run_file()
            else:
                self._run_stream()
        finally:
            self._close_segment()   # последний, неполный кусок тоже проверяем
            self.stats["online"] = False

    def _run_playlist(self):
        """Видео из плейлиста по очереди, по кругу. Каждое видео — новая сцена: трекер начинает заново."""
        n = 0
        while not self.stop_event.is_set():
            for path in self._videos():
                if self.stop_event.is_set():
                    return
                self._close_segment()   # видео бывают разного размера — кусок не может смешивать кадры
                self.now_playing = path.name
                self.tracker = IouTracker(**self.cfg.get("tracker", {}))
                self._run_file(path, loop=False, t_offset=n * 1e6)   # сдвиг времени: скорость не «прыгает»
                n += 1
            if not self.cam["loop"]:
                break

    def _run_file(self, path: Path | None = None, loop: bool | None = None, t_offset: float = 0.0):
        cap = cv2.VideoCapture(str(path)) if path else self._open()
        loop = self.cam["loop"] if loop is None else loop
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(src_fps / self.cam["fps"]))   # анализируем каждый step-й кадр
        self.recorder.set_fps(src_fps / step)   # фактическая частота анализа: 5 с куска = 5 с видео
        idx = loops = 0
        self.stats["online"] = True
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx, loops = 0, loops + 1
                    continue
                break
            idx += 1
            if idx % step:
                continue
            self._media_t = t_offset + loops * 1e6 + idx / src_fps
            t0 = time.monotonic()
            self.process(frame)
            if self.realtime:   # для демо проигрываем файл со скоростью реального времени
                time.sleep(max(0.0, step / src_fps - (time.monotonic() - t0)))
        cap.release()

    # ---------- ручной выбор стоп-кадра (manual: true у камеры) ----------

    @property
    def images(self) -> list[Path]:
        if not hasattr(self, "_img_list"):
            self._img_list = self._images()
        return self._img_list

    def select_image(self, index: int):
        """Показать кадр с номером index (по кругу). Вызывается из интерфейса."""
        if self.images:
            self.image_index = index % len(self.images)

    def step_image(self, delta: int):
        self.select_image(getattr(self, "image_index", 0) + delta)

    def _run_images(self):
        """Стоп-кадры: камера смотрит на неподвижную сцену с частотой fps — так работают трекинг и сглаживание.
        manual: true — кадр держится, пока оператор не выберет другой;
        иначе каждый показывается image_hold_sec секунд, по кругу."""
        imgs, fps = self.images, self.cam["fps"]
        self.recorder.set_fps(fps)
        self.stats["online"] = True
        if self.cam.get("manual"):
            self.image_index, shown, frame = getattr(self, "image_index", 0), None, None
            while not self.stop_event.is_set():
                if self.image_index != shown:
                    shown, frame = self.image_index, cv2.imread(str(imgs[self.image_index]))
                    self.latest_frame = self.latest_raw = frame   # новый кадр виден сразу, до обработки
                t0 = time.monotonic()
                if frame is not None:
                    self.process(frame)
                time.sleep(max(0.0, 1 / fps - (time.monotonic() - t0)))
            return
        hold = max(1, round(self.cam.get("image_hold_sec", 6) * fps))
        while not self.stop_event.is_set():
            for path in imgs:
                frame = cv2.imread(str(path))
                for _ in range(hold if frame is not None else 0):
                    if self.stop_event.is_set():
                        return
                    t0 = time.monotonic()
                    self.process(frame)
                    time.sleep(max(0.0, 1 / fps - (time.monotonic() - t0)))
            if not self.cam["loop"]:
                break

    def _run_stream(self):
        """RTSP/веб-камера: отдельный поток постоянно читает кадры, а мы берём самый свежий.
        Иначе при медленной обработке OpenCV копит буфер и картинка отстаёт от реальности."""
        threading.Thread(target=self._grab_loop, daemon=True).start()
        period = 1.0 / self.cam["fps"]
        while not self.stop_event.is_set():
            t0 = time.monotonic()
            frame, self._latest_frame = self._latest_frame, None
            if frame is not None:
                self.process(frame)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _grab_loop(self):
        cap = self._open()
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            self.stats["online"] = ok
            if ok:
                self._latest_frame = frame
                continue
            log.warning("Камера %s недоступна, переподключение...", self.cam["id"])
            cap.release()
            time.sleep(2)
            cap = self._open()
        cap.release()

    # ---------- обработка кадра ----------

    def process(self, frame):
        if self.latest_frame is None:   # чтобы интерфейс сразу показал картинку, ещё до первой обработки
            self.latest_frame, self.latest_raw = frame, frame
        t0 = time.perf_counter()
        with self.backend_lock:   # одна модель на все камеры
            dets = self.backend.detect(frame)
        for d in dets:
            d.role = self.class_map.get(d.cls.lower())
        if self.fire is not None:   # огонь — отдельная модель (CLIP), раз в секунду
            with self.backend_lock:
                dets += self.fire.detect(frame, self._media_t if self._media_t is not None else time.time())
        infer_ms = (time.perf_counter() - t0) * 1000

        persons = [d for d in dets if d.role == "person"]
        items = [d for d in dets if d.role and d.role != "person"]

        h, w = frame.shape[:2]
        now = time.time()
        tracks = self.tracker.update(persons)
        violations = evaluate(tracks, items, self.cam["zones"], self.rules, (w, h))
        if self.behavior is not None:   # поведение во времени: бег, падение
            with self.backend_lock:
                beh = self.behavior.update(self._media_t if self._media_t is not None else now, tracks, frame, items)
            for tid, vs in beh.items():
                violations.setdefault(tid, []).extend(vs)
        alerts = self.debouncer.update(now, violations)

        s = self.stats
        rec = self.recorder
        header = (f"{self.cam['id']} | {self.backend.name} | {s['fps']:.1f} FPS | infer {infer_ms:.0f} ms"
                  f" | REC {rec.frames + 1}/{rec.max_frames}")
        annotated = draw(frame, tracks, items, self.cam["zones"], violations, header)

        # подтверждённые (сглаженные) нарушения на этом кадре — с местом, где стоит нарушитель
        boxes = {t.id: t.box for t in tracks}
        fires = [d.box for d in items if d.role == "fire"]
        if fires:   # у тревоги места рамка — огонь
            boxes[PLACE_ID] = fires[0]
        findings = [self._describe(tid, v, boxes[tid], w, h) for (tid, _), v in self.debouncer.active.items()
                    if tid in boxes]
        alert_keys = {(tid, v.type) for tid, v in alerts}
        for f in findings:
            if (f["track_id"], f["type"]) in alert_keys:   # одна тревога на нарушение
                self._raise_alert(f, frame, annotated, now)

        if rec.add(annotated, now, findings):   # набрали 5 секунд -> решаем, сохранять или удалить
            self._close_segment()

        self.latest_frame, self.latest_raw = annotated, frame
        self.latest_jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        if self.writer is not None:
            self.writer.write(annotated)

        # скользящие средние для дашборда
        dt = time.perf_counter() - getattr(self, "_last_t", t0)
        self._last_t = time.perf_counter()
        if dt > 0:
            s["fps"] = 0.9 * s["fps"] + 0.1 * (1 / dt) if s["frames"] else 0.0
        s["infer_ms"] = 0.9 * s["infer_ms"] + 0.1 * infer_ms if s["frames"] else infer_ms
        s["frames"] += 1
        s["people"] = len(tracks)
        s["violating"] = sum(1 for vs in violations.values() if vs)

    def _describe(self, tid: int, v, box, w: int, h: int) -> dict:
        """Где именно произошло нарушение: зоны, в которых стоит человек, и его точка в кадре (доли 0..1)."""
        fx, fy = foot_point(box)
        pos = [round(fx / w, 3), round(fy / h, 3)]
        zones = [z["name"] for z in self.cam["zones"] if point_in_polygon(pos, z["polygon"])]
        return {"track_id": tid, "type": v.type, "label": VIOLATION_LABELS.get(v.type, v.type),
                "zones": zones, "pos": pos, "box": [round(x) for x in box]}

    def _raise_alert(self, finding: dict, raw, annotated, now: float):
        """Тревога: кадр-доказательство (размеченный и исходный — исходный пригодится для дообучения,
        если оператор отметит тревогу ложной), запись в журнал, уведомление."""
        stem = f"{self.cam['id']}_{int(now * 1000)}_{finding['track_id']}_{finding['type']}"
        cv2.imwrite(str(self.snap_dir / f"{stem}.jpg"), annotated)
        cv2.imwrite(str(self.snap_dir / f"{stem}_raw.jpg"), raw)
        alert = self.store.add_alert(self.cam["id"], self.place, finding, now, f"alerts/{stem}.jpg",
                                     f"alerts/{stem}_raw.jpg", finding["type"] in CRITICAL)
        self._pending_alerts.append(alert["id"])
        log.info("Нарушение: %s — %s %s (камера %s, #%d)", finding["label"], self.place,
                 finding["zones"], self.cam["id"], finding["track_id"])
        self.notifier.send({**finding, "camera": self.cam["id"], "place": self.place})

    def _close_segment(self):
        self.stats["segments"] += 1 if self.recorder.frames else 0
        seg = self.recorder.close()
        if seg is None:
            return   # нарушений не было — кусок уже удалён
        if self._pending_alerts:   # клип этого куска — видеодоказательство его тревог
            self.store.attach_clip(self._pending_alerts, seg["clip"])
            self._pending_alerts = []
        self.store.remove_clips(seg.get("removed_old", []))
        inc = self.store.add(self.cam["id"], self.place, seg)
        self.stats["incidents"] += 1
        log.info("Сохранён клип %s: %s", seg["clip"], inc["types"])

    def stop(self):
        self.stop_event.set()

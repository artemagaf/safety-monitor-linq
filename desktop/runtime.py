"""Запуск ядра для приложения: модель, журнал, по потоку обработки на каждую камеру."""
import logging
import threading
from pathlib import Path

from app.backends import make_backend
from app.config import save_camera_states
from app.events import IncidentStore, Notifier
from app.pipeline import CameraWorker

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_note = ""
        m = cfg["model"]
        if m["backend"] == "composite":
            missing = [p["path"] for p in m["parts"] if not Path(p["path"]).exists()]
            if missing:
                self.model_note = "Нет весов: " + ", ".join(Path(p).name for p in missing) + " — работает yolo11n.pt"
                log.warning(self.model_note)
                m.update(backend="ultralytics", path=str(Path(missing[0]).with_name("yolo11n.pt")))
        if m["backend"] == "ultralytics" and not Path(m["path"]).exists():
            fallback = Path(m["path"]).with_name("yolo11n.pt")
            self.model_note = f"{Path(m['path']).name} ещё не обучена — работает {fallback.name} (видит только людей)"
            log.warning(self.model_note)
            m["path"] = str(fallback)
        self.backend = make_backend(cfg)
        self.lock = threading.Lock()   # одна модель на все камеры
        self.store = IncidentStore(cfg["events"]["db"])
        self.notifier = Notifier(cfg.get("notify", {}))
        self.pose = None   # модель позы для распознавания падения (вызывается только для «лежачих» рамок)
        pose_path = cfg.get("behavior", {}).get("pose_model")
        if pose_path and Path(pose_path).exists():
            from ultralytics import YOLO
            self.pose = YOLO(pose_path)
        # поток есть у каждой камеры из конфига, но запускаются только включённые
        self.workers = [self._make_worker(cam) for cam in cfg["cameras"]]
        self.clips_dir = Path(cfg.get("segments", {}).get("dir", "data/clips"))
        self.retrain_dir = Path(cfg["events"]["db"]).parent / "retrain"

    @property
    def model_name(self) -> str:
        m = self.cfg["model"]
        if m["backend"] == "composite":
            return " + ".join(Path(p["path"]).stem for p in m["parts"])
        return f"{self.backend.name} · {Path(m['path']).name}"

    def _make_worker(self, cam: dict) -> CameraWorker:
        return CameraWorker(cam, self.cfg, self.backend, self.lock, self.store, self.notifier, realtime=True,
                            pose_model=self.pose)

    @property
    def active(self) -> list[CameraWorker]:
        return [w for w in self.workers if w.cam.get("enabled", True)]

    def start(self):
        for w in self.active:
            w.start()

    def set_enabled(self, cam_id: str, on: bool):
        """Включить или выключить камеру на ходу. Поток нельзя запустить второй раз, поэтому при
        включении создаётся новый — после того как прошлый поток этой камеры дописал последний кусок."""
        i = next(i for i, w in enumerate(self.workers) if w.cam["id"] == cam_id)
        old, cam = self.workers[i], self.workers[i].cam
        if cam.get("enabled", True) == on:
            return
        cam["enabled"] = on
        save_camera_states(self.cfg)
        log.info("Камера %s %s", cam_id, "включена" if on else "выключена")
        if not on:
            old.stop()
            return

        def restart():
            if old.is_alive():
                old.join(timeout=10)
            new = self._make_worker(cam)
            if hasattr(old, "image_index"):   # ручная камера: остаёмся на том же кадре
                new.image_index = old.image_index
            self.workers[i] = new
            if cam.get("enabled", True):      # пока ждали, могли выключить снова
                new.start()
        threading.Thread(target=restart, daemon=True).start()

    def stop(self):
        for w in self.workers:
            w.stop()
        for w in self.workers:
            if w.is_alive():
                w.join(timeout=3)

"""Запись видео кусками по N секунд (по умолчанию 5).

Кадры сразу пишутся во временный файл на диск (а не копятся в памяти). Когда кусок набран:
  - нарушений не было -> файл удаляется;
  - нарушения были    -> файл сохраняется как клип-доказательство, и возвращается описание:
                         что нарушено, кем (номер человека), где (зона, точка в кадре), когда.
Затем начинается следующий кусок — и так по кругу.

Память не забивается: в ОЗУ держится только один кадр-превью, на диске — один временный кусок
на камеру плюс сохранённые клипы, общий объём которых ограничен max_total_mb (старые удаляются).
"""
import logging
import os
import time
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


def enforce_limit(directory: Path, max_total_mb: float) -> list[str]:
    """Удаляет самые старые сохранённые клипы (и их превью), пока папка не влезет в лимит."""
    clips = sorted((p for p in directory.glob("*.mp4") if not p.name.startswith("_tmp_")),
                   key=lambda p: p.stat().st_mtime)
    sizes = {p: p.stat().st_size + _size(p.with_suffix(".jpg")) for p in clips}
    total = sum(sizes.values())
    removed = []
    while clips and total > max_total_mb * 1024 * 1024:
        p = clips.pop(0)
        total -= sizes[p]
        p.unlink(missing_ok=True)
        p.with_suffix(".jpg").unlink(missing_ok=True)
        removed.append(p.name)
    if removed:
        log.info("Лимит %s МБ: удалено старых клипов — %d", max_total_mb, len(removed))
    return removed


def _size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


class SegmentRecorder:
    def __init__(self, cam_id: str, out_dir: str, fps: float, seconds: float, max_total_mb: float | None = None,
                 min_violation_sec: float = 1.0):
        self.cam_id = cam_id
        # кусок сохраняем, только если нарушение видно в нём хотя бы min_violation_sec:
        # хвост в доли секунды на стыке кусков уже есть в соседнем клипе
        self.min_violation_sec = min_violation_sec
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_total_mb = max_total_mb
        for p in self.dir.glob(f"_tmp_{cam_id}_*.mp4"):   # недописанные куски после аварийного выключения
            try:
                p.unlink(missing_ok=True)
            except PermissionError:   # кусок ещё дописывает прошлый поток этой камеры
                pass
        self.seconds = seconds
        self.set_fps(fps)
        self._reset()

    def set_fps(self, fps: float):
        """Частота, с которой в кусок приходят кадры. Кусок меряем числом кадров: seconds × fps."""
        self.fps = fps
        self.max_frames = max(1, round(self.seconds * fps))

    def _reset(self):
        self.writer = None
        self.tmp_path: Path | None = None
        self.frames = 0
        self.t_start = self.t_end = 0.0
        self.findings: dict[tuple[int, str], dict] = {}   # (человек, тип) -> описание нарушения
        self.best_frame, self.best_n = None, 0             # кадр с наибольшим числом нарушений — для превью

    def add(self, frame, now: float, findings: list[dict]) -> bool:
        """Добавляет кадр в текущий кусок. Возвращает True, когда кусок набран и пора вызвать close()."""
        if self.writer is None:
            h, w = frame.shape[:2]
            self.t_start = now
            self.tmp_path = self.dir / f"_tmp_{self.cam_id}_{int(now * 1000)}.mp4"
            self.writer = cv2.VideoWriter(str(self.tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

        self.writer.write(frame)
        self.frames += 1
        self.t_end = now

        for f in findings:
            key = (f["track_id"], f["type"])
            if key in self.findings:
                self.findings[key]["frames"] += 1
            else:   # запоминаем место и время, где нарушение впервые замечено в этом куске
                self.findings[key] = {**f, "first_seen": now, "frames": 1}
        if len(findings) > self.best_n:
            self.best_frame, self.best_n = frame.copy(), len(findings)

        return self.frames >= self.max_frames

    def close(self) -> dict | None:
        """Завершает кусок. None — нарушений не было, файл удалён. Иначе — описание сохранённого клипа."""
        if self.writer is None:
            return None
        self.writer.release()   # на Windows файл нельзя переименовать/удалить, пока он открыт

        min_frames = max(1, round(self.min_violation_sec * self.fps))
        if not any(f["frames"] >= min_frames for f in self.findings.values()):
            os.remove(self.tmp_path)
            self._reset()
            return None

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t_start))
        clip = f"{self.cam_id}_{stamp}_{int(self.t_start * 1000) % 1000:03d}.mp4"
        os.replace(self.tmp_path, self.dir / clip)
        snapshot = clip.replace(".mp4", ".jpg")
        cv2.imwrite(str(self.dir / snapshot), self.best_frame)

        result = {
            "clip": clip, "snapshot": snapshot,
            "t_start": self.t_start, "t_end": self.t_end, "frames": self.frames,
            "violations": list(self.findings.values()),
        }
        self._reset()
        if self.max_total_mb:
            result["removed_old"] = enforce_limit(self.dir, self.max_total_mb)
        return result

    def discard(self):
        """Прервать без сохранения (например, при аварийной остановке)."""
        if self.writer is not None:
            self.writer.release()
            os.remove(self.tmp_path)
        self._reset()

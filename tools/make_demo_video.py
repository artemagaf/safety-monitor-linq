"""Демо-видео для сдачи: все сценарии через настоящий конвейер (модели, трекер, правила, сглаживание).

Каждый сюжет — кадры из data/ прогоняются через CameraWorker.process, на видео идёт размеченный кадр
конвейера и подпись по-русски с подтверждёнными нарушениями. В конце — экраны приложения.

    python tools/make_demo_video.py --out docs/demo.mp4 [--extra data/sh17/видео.mp4]
"""
import argparse
import copy
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends import make_backend  # noqa: E402
from app.config import load_config  # noqa: E402
from app.events import IncidentStore, Notifier  # noqa: E402
from app.pipeline import CameraWorker  # noqa: E402
from app.rules import VIOLATION_LABELS  # noqa: E402

W, H, FPS = 1280, 720, 6
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def fit(img):
    """Кадр любого размера — в 1280×720 с чёрными полями."""
    h, w = img.shape[:2]
    s = min(W / w, (H - 90) / h)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - 90 - img.shape[0]) // 2, (W - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def caption(frame, title, alerts):
    """Нижняя плашка: что показываем и какие нарушения система подтвердила."""
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    d.rectangle((0, H - 90, W, H), fill=(14, 17, 21))
    d.text((24, H - 82), title, font=font(26, True), fill=(245, 190, 30))
    if alerts:
        d.text((24, H - 42), "Тревога: " + ", ".join(alerts), font=font(22), fill=(255, 90, 90))
    else:
        d.text((24, H - 42), "Нарушений нет", font=font(22), fill=(120, 200, 130))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def card(lines, sub=None):
    im = Image.new("RGB", (W, H), (14, 17, 21))
    d = ImageDraw.Draw(im)
    y = 250
    for i, line in enumerate(lines):
        d.text((90, y), line, font=font(54 if i == 0 else 30, i == 0), fill=(245, 190, 30) if i == 0 else (230, 233, 238))
        y += 80 if i == 0 else 46
    if sub:
        d.text((90, H - 90), sub, font=font(22), fill=(150, 156, 165))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


class Demo:
    def __init__(self, cfg_path):
        self.cfg = load_config(cfg_path)
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg["segments"]["dir"] = str(self.tmp / "clips")
        self.backend = make_backend(self.cfg)
        from ultralytics import YOLO
        self.pose = YOLO(self.cfg["behavior"]["pose_model"])
        self.store = IncidentStore(str(self.tmp / "db.sqlite"))
        self.n = 0

    def worker(self, cam_id, **over):
        cam = copy.deepcopy(next(c for c in self.cfg["cameras"] if c["id"] == cam_id))
        cam.update(over)
        self.n += 1
        cfg = {**self.cfg, "segments": {**self.cfg["segments"], "dir": str(self.tmp / f"clips{self.n}")}}
        w = CameraWorker(cam, cfg, self.backend, threading.Lock(), self.store, Notifier({}), realtime=False,
                         pose_model=self.pose)
        w.snap_dir = self.tmp
        return w

    @staticmethod
    def alerts(w):
        return sorted({VIOLATION_LABELS.get(v.type, v.type) for v in w.debouncer.active.values()})

    def still(self, writer, cam_id, path, title, n=12, **over):
        """Стоп-кадр: камера смотрит на неподвижную сцену 3 раза в секунду (как в приложении)."""
        w, frame = self.worker(cam_id, **over), cv2.imread(str(path))
        for i in range(n):
            w._media_t = i / 3
            w.process(frame)
            out = caption(fit(w.latest_frame), title, self.alerts(w))
            for _ in range(FPS // 3):
                writer.write(out)

    def video(self, writer, cam_id, path, title, fps=3.0, titles=None, **over):
        w, cap = self.worker(cam_id, **over), cv2.VideoCapture(str(path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step, idx = max(1, round(src_fps / fps)), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                w._media_t = idx / src_fps
                w.process(frame)
                t = title(w._media_t) if callable(title) else title
                out = caption(fit(w.latest_frame), t, self.alerts(w))
                for _ in range(max(1, round(FPS / fps))):
                    writer.write(out)
            idx += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "docs/demo.mp4"))
    ap.add_argument("--extra", action="append", default=[], help="своё видео (например, курение)")
    a = ap.parse_args()

    demo = Demo(a.config)
    tmp_out = str(Path(a.out).with_suffix(".tmp.mp4"))
    vw = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    hold = lambda img, sec: [vw.write(img) for _ in range(int(sec * FPS))]

    hold(card(["Пост контроля ОТ", "Нарушения техники безопасности по видео", "Инференс на ускорителе LinQ",
               "Команда «МГОКчане» · кейс LinQ · Открой#Моспром 2026"],
              "Все кадры ниже — реальный прогон конвейера: модели → трекер → правила → одна тревога"), 4)

    hold(card(["1. Станки: правила рабочего места", "Перчатки у вращающегося патрона запрещены, очки обязательны"]), 2.5)
    s = ROOT / "data/samples"
    demo.still(vw, "lathe1", s / "lathe.png", "Токарный станок: перчатки у патрона — критично")
    demo.still(vw, "laser1", s / "laser.png", "Лазерная резка: здесь перчатки — норма, правило зависит от станка")
    demo.still(vw, "cnc1", s / "cnc.png", "ЧПУ: участок без нарушений")

    for ex in a.extra:
        hold(card(["2. Курение", "«Второй взгляд»: вырезка головы и рук каждого человека"]), 2.5)
        demo.video(vw, "myvideos", ex, "Курение на площадке")

    hold(card(["3. Поведение во времени", "Бег, падение, телефон у станка, сближение с техникой"]), 2.5)
    demo.video(vw, "demo", ROOT / "data/scenarios.mp4", "Сценарии поведения: трекер + скелет + время")

    hold(card(["4. Открытый огонь и переноска грузов", "Огонь — тревога места, даже без людей в кадре",
               "Подъём груза: правильно — присесть, спина прямая"]), 3)
    sc = ROOT / "data/scenarios2"
    demo.still(vw, "scenes2", sc / "1_fire_campfire.jpg", "Открытый огонь, людей нет", manual=False)
    demo.still(vw, "scenes2", sc / "3_lift_correct_squat.jpg", "Правильный подъём: присед, груз у тела", manual=False)
    demo.still(vw, "scenes2", sc / "4_bent_back_shovel.jpg", "Подъём спиной на прямых ногах", manual=False)
    demo.still(vw, "scenes2", sc / "7_crouch_blocks.jpg", "Присел у груза — нарушения нет", manual=False)

    hold(card(["5. Приложение диспетчера", "Мониторинг, журнал, статистика, зоны и правила, отчёты"]), 2.5)
    for name, t in [("monitor", "Мониторинг: камеры и лента тревог"), ("journal", "Журнал: фильтры, клип 5 с, CSV"),
                    ("stats", "Статистика смены"), ("zones", "Зоны и правила рисуются мышью")]:
        img = cv2.imread(str(ROOT / f"docs/screens/{name}.png"))
        if img is not None:
            hold(caption(fit(img), t, []), 3)

    hold(card(["github.com/artemagaf/safety-monitor-linq", "Код, модели, инструкция по запуску и переносу на LinQ"]), 4)
    vw.release()

    # H.264 — чтобы видео открывалось в браузере (GitHub, Яндекс Диск)
    try:
        import imageio_ffmpeg
        import subprocess
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp_out, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", a.out], check=True)
        Path(tmp_out).unlink()
    except Exception as e:   # нет ffmpeg — оставляем mp4v
        print("ffmpeg недоступен:", e)
        Path(tmp_out).replace(a.out)
    print("готово:", a.out)


if __name__ == "__main__":
    main()

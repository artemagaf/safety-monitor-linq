"""Прогон по видеофайлу: режем на 5-секундные куски, сохраняем только куски с нарушениями.

Пример:
    python tools/demo_video.py --video data/demo.mp4
    python tools/demo_video.py --video data/demo.mp4 --out data/demo_out.mp4   # + полный размеченный ролик
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends import make_backend  # noqa: E402
from app.config import active_cameras, load_config  # noqa: E402
from app.events import IncidentStore, Notifier  # noqa: E402
from app.pipeline import CameraWorker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", help="сохранить ещё и полный размеченный ролик (для отладки)")
    ap.add_argument("--camera", default=None, help="id камеры из конфига, чьи зоны и место работ использовать")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam = next((c for c in cfg["cameras"] if c["id"] == args.camera), active_cameras(cfg)[0])
    cam = {**cam, "source": str(Path(args.video).resolve()), "loop": False}

    writer = None
    if args.out:
        cap = cv2.VideoCapture(cam["source"])
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), cam["fps"], (w, h))

    store = IncidentStore(cfg["events"]["db"])
    last_id = max((i["id"] for i in store.list(limit=1)), default=0)
    worker = CameraWorker(cam, cfg, make_backend(cfg), threading.Lock(), store,
                          Notifier(cfg.get("notify", {})), realtime=False, writer=writer)
    worker.run()   # синхронно, без отдельного потока
    if writer:
        writer.release()

    s = worker.stats
    print(f"Кадров обработано: {s['frames']}, среднее время инференса: {s['infer_ms']:.1f} мс")
    print(f"Кусков по {cfg.get('segments', {}).get('seconds', 5)} с: {s['segments']}, "
          f"сохранено с нарушениями: {s['incidents']}, удалено чистых: {s['segments'] - s['incidents']}")
    for inc in reversed(store.list(after_id=last_id, limit=100)):
        t = time.strftime("%H:%M:%S", time.localtime(inc["t_start"]))
        print(f"\n  Инцидент #{inc['id']} [{t}] {inc['place']} — клип {inc['clip']}")
        for v in inc["violations"]:
            where = ", ".join(v["zones"]) or "вне зон"
            print(f"    • {v['label']}: человек #{v['track_id']}, {where}, точка {v['pos']}, кадров {v['frames']}")


if __name__ == "__main__":
    main()

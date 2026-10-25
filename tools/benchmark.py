"""Замер скорости бэкендов: задержка на кадр и FPS. Таблица в конце — готовый слайд для презентации.

Пример:
    python tools/benchmark.py --image data/sample.jpg --backend ultralytics:models/ppe.pt --backend onnx:models/ppe.onnx
    python tools/benchmark.py --image data/sample.jpg --backend linq     # параметры LinQ берутся из config.yaml
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends import YoloRawBackend, make_backend  # noqa: E402
from app.config import load_config  # noqa: E402


def backend_from_spec(cfg: dict, spec: str):
    """'onnx:models/x.onnx' -> бэкенд onnx с этой моделью; 'linq' -> настройки из конфига."""
    kind, _, path = spec.partition(":")
    m = {**cfg["model"], "backend": kind}
    if path:
        m["path"] = str((ROOT / path) if not Path(path).is_absolute() else path)
    return make_backend({**cfg, "model": m})


def bench(backend, frame, warmup: int, runs: int) -> dict:
    for _ in range(warmup):   # первые прогоны медленнее: выделение памяти, прогрев кэшей/устройства
        backend.detect(frame)

    total, fwd = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        if isinstance(backend, YoloRawBackend):
            blob, r, pad = backend.preprocess(frame)
            t1 = time.perf_counter()
            out = backend._forward(blob)
            fwd.append((time.perf_counter() - t1) * 1000)
            backend.postprocess(out, r, pad, frame.shape)
        else:
            backend.detect(frame)
        total.append((time.perf_counter() - t0) * 1000)

    total.sort()
    return {
        "mean": statistics.mean(total), "p50": total[len(total) // 2], "p95": total[int(len(total) * 0.95) - 1],
        "fwd": statistics.mean(fwd) if fwd else None, "fps": 1000 / statistics.mean(total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--image", required=True)
    ap.add_argument("--backend", action="append", required=True, help="kind[:path], можно несколько")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    frame = cv2.imread(args.image)
    assert frame is not None, f"не удалось прочитать {args.image}"

    rows = []
    for spec in args.backend:
        b = backend_from_spec(cfg, spec)
        r = bench(b, frame, args.warmup, args.runs)
        rows.append((f"{b.name} ({spec})", r))
        print(f"{spec}: {r['mean']:.1f} ms, {r['fps']:.1f} FPS")

    print("\n| Бэкенд | Сеть, мс | Всего, мс (mean) | p50 | p95 | FPS |")
    print("|---|---|---|---|---|---|")
    for name, r in rows:
        fwd = f"{r['fwd']:.1f}" if r["fwd"] is not None else "—"
        print(f"| {name} | {fwd} | {r['mean']:.1f} | {r['p50']:.1f} | {r['p95']:.1f} | {r['fps']:.1f} |")


if __name__ == "__main__":
    main()

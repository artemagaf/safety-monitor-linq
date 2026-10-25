"""Открытый огонь без обучения: CLIP (ViT-B/32) сравнивает кадр с текстовыми описаниями.

Почему не YOLO. В открытых датасетах СИЗ огня нет, а YOLO-World по слову «fire» на костре
не находит ничего. CLIP учили на сотнях миллионов пар «картинка — подпись», и «пламя» он
узнаёт уверенно: на костре 1,00, на кадрах цеха, станков и SH17 — не выше 0,26.

Как работает. Кадр и его 4 четверти (огонь бывает маленьким, в углу) переводятся в вектор и
сравниваются с подписями «огонь» и «не огонь» (цех, лампа, сварочные искры — чтобы не путать).
Вероятность огня = сумма вероятностей «огненных» подписей. Где выше порога — там огонь,
рамка — эта четверть (или весь кадр).

Это нарушение места, а не человека: тревога срабатывает и когда людей в кадре нет.
CLIP медленнее YOLO (~0,25 с на 5 вырезок на CPU), поэтому проверяем раз в interval_sec,
а между проверками отдаём прошлый результат — огонь за секунду не исчезает.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2

from .backends import Detection

log = logging.getLogger(__name__)

FIRE = ["a photo of open fire with flames", "a photo of a burning fire", "a photo of flames and smoke"]
NOT_FIRE = ["a photo of a factory workshop", "a photo of a person at work", "a photo of a glowing lamp",
            "a photo of an office room", "a photo of a machine tool", "a photo of a street",
            "a photo of a warehouse with boxes", "a photo of sparks from welding"]

_model = None
_load_lock = threading.Lock()


def _load(weights_dir: str):
    """Одна модель CLIP на все камеры (≈350 МБ)."""
    global _model
    with _load_lock:
        if _model is None:
            import clip
            import torch
            model, prep = clip.load("ViT-B/32", device="cpu", download_root=weights_dir)
            with torch.no_grad():
                text = model.encode_text(clip.tokenize(FIRE + NOT_FIRE))
                text = text / text.norm(dim=-1, keepdim=True)
            _model = (model, prep, text)
        return _model


_loader: threading.Thread | None = None
_loader_lock = threading.Lock()   # отдельный: _load_lock занят всё время скачивания


def _load_in_background(weights_dir: str) -> bool:
    """True — модель готова. Иначе запускает загрузку в фоне (первый запуск скачивает ~340 МБ)
    и возвращает False: камеры работают без огня, пока веса не скачаются, а не стоят."""
    global _loader
    if _model is not None:
        return True
    with _loader_lock:
        if _loader is None:
            def run():
                try:
                    _load(weights_dir)
                    log.info("CLIP загружен — распознавание огня включено")
                except Exception as e:   # нет интернета и т. п. — остальное работает
                    log.warning("CLIP не загрузился, огонь не распознаётся: %s", e)
            _loader = threading.Thread(target=run, daemon=True, name="clip-load")
            _loader.start()
    return False


class FireDetector:
    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.threshold = c.get("threshold", 0.8)
        self.interval = c.get("interval_sec", 1.0)
        self.weights_dir = str(Path(c.get("weights_dir", "models/weights/clip")))
        self._last_t = -1e9
        self._last: list[Detection] = []

    def probs(self, frame) -> list[tuple[tuple[int, int, int, int], float]]:
        """Вероятность огня для целого кадра и каждой четверти: [(рамка, p), ...]."""
        import torch
        from PIL import Image
        model, prep, text = _load(self.weights_dir)
        h, w = frame.shape[:2]
        boxes = [(0, 0, w, h), (0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                 (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = torch.stack([prep(Image.fromarray(rgb[y1:y2, x1:x2])) for x1, y1, x2, y2 in boxes])
        with torch.no_grad():
            img = model.encode_image(batch)
            img = img / img.norm(dim=-1, keepdim=True)
            p = (100.0 * img @ text.T).softmax(dim=-1)[:, :len(FIRE)].sum(dim=-1)
        return list(zip(boxes, p.tolist()))

    def detect(self, frame, t: float | None = None) -> list[Detection]:
        t = time.time() if t is None else t
        if not _load_in_background(self.weights_dir):
            return []
        if abs(t - self._last_t) < self.interval:
            return [Detection(d.box, d.conf, d.cls, "fire") for d in self._last]
        self._last_t = t
        res = [(b, p) for b, p in self.probs(frame) if p >= self.threshold]
        whole = [r for r in res if r[0][0] == 0 and r[0][1] == 0 and r[0][2] == frame.shape[1]]
        parts = [r for r in res if r not in whole]
        keep = parts or whole   # нашлась четверть — рамка точнее, чем «весь кадр»
        self._last = [Detection(tuple(float(x) for x in b), round(p, 2), "fire", "fire") for b, p in keep]
        return [Detection(d.box, d.conf, d.cls, "fire") for d in self._last]

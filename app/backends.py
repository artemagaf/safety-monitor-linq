"""Бэкенды инференса детектора.

Все бэкенды возвращают одно и то же: список Detection в пикселях исходного кадра.
Благодаря этому трекер, правила и веб-сервис не знают, где выполняется сеть:
на CPU (ultralytics / onnxruntime) или на ускорителе LinQ.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях кадра
    conf: float
    cls: str                                 # имя класса, как его выдаёт модель
    role: str | None = None                  # роль после class_map: person / helmet / no_helmet / ...


class Backend:
    name = "base"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class UltralyticsBackend(Backend):
    """Самый простой путь: модель .pt через библиотеку ultralytics. Удобно для отладки и обучения."""
    name = "ultralytics"

    def __init__(self, m: dict):
        from ultralytics import YOLO

        self.model = YOLO(m["path"])
        self.conf, self.iou, self.imgsz, self.device = m["conf"], m["iou"], m["imgsz"], m.get("device", "cpu")

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        return [
            Detection(tuple(box), conf, r.names[int(k)])
            for box, conf, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
        ]


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Масштабирует кадр в квадрат size×size с сохранением пропорций, добивая поля серым (как при обучении YOLO)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


class YoloRawBackend(Backend):
    """Общая пре- и постобработка для «сырого» выхода YOLOv8/YOLO11: тензор (1, 4 + число_классов, N).

    Наследники реализуют только _forward(blob) — прямой проход сети. Именно эту часть
    переносим на ускоритель LinQ, всё остальное (resize, NMS) остаётся на CPU в numpy/OpenCV.
    """

    def __init__(self, m: dict, names: dict[int, str], imgsz: int | None = None):
        self.size = imgsz or m["imgsz"]
        self.conf, self.iou = m["conf"], m["iou"]
        self.names = names

    def preprocess(self, frame):
        img, r, pad = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)  # BGR->RGB, HWC->NCHW, float32 0..1
        return blob, r, pad

    def postprocess(self, out: np.ndarray, r: float, pad: tuple[int, int], shape) -> list[Detection]:
        pred = np.squeeze(np.asarray(out), 0)
        channels = 4 + len(self.names)
        if pred.shape[0] != channels and pred.shape[1] == channels:   # некоторые рантаймы отдают (N, 4+nc)
            pred = pred.T
        boxes, scores = pred[:4].T, pred[4:].T
        cls_ids, confs = scores.argmax(1), scores.max(1)
        keep = confs >= self.conf
        boxes, cls_ids, confs = boxes[keep], cls_ids[keep], confs[keep]
        if not len(confs):
            return []

        # (cx, cy, w, h) в координатах letterbox -> (x1, y1, x2, y2) в координатах исходного кадра
        pad_xy = np.array(pad, dtype=np.float32)
        x1y1 = (boxes[:, :2] - boxes[:, 2:] / 2 - pad_xy) / r
        x2y2 = (boxes[:, :2] + boxes[:, 2:] / 2 - pad_xy) / r
        h, w = shape[:2]
        x1y1 = x1y1.clip(0, [w, h])
        x2y2 = x2y2.clip(0, [w, h])

        xywh = np.concatenate([x1y1, x2y2 - x1y1], axis=1).tolist()
        idx = cv2.dnn.NMSBoxesBatched(xywh, confs.tolist(), cls_ids.tolist(), self.conf, self.iou)
        return [
            Detection((*x1y1[i].tolist(), *x2y2[i].tolist()), float(confs[i]),
                      self.names.get(int(cls_ids[i]), str(cls_ids[i])))
            for i in np.array(idx).flatten()
        ]

    def detect(self, frame):
        blob, r, pad = self.preprocess(frame)
        return self.postprocess(self._forward(blob), r, pad, frame.shape)

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def read_onnx_names(path: str) -> dict[int, str]:
    """ultralytics при экспорте кладёт имена классов в метаданные ONNX."""
    import onnxruntime as ort

    meta = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    return ast.literal_eval(meta["names"]) if "names" in meta else {}


class OnnxBackend(YoloRawBackend):
    """Модель .onnx через onnxruntime. Это эталон для сверки результатов с LinQ."""
    name = "onnx"

    def __init__(self, m: dict, path: str | None = None):
        import onnxruntime as ort

        path = path or m["path"]
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        fixed = inp.shape[2] if isinstance(inp.shape[2], int) else None  # размер входа, зашитый при экспорте
        super().__init__(m, read_onnx_names(path), imgsz=fixed)

    def _forward(self, blob):
        return self.sess.run(None, {self.input_name: blob})[0]


def read_onnx_io(path: str) -> tuple[str, str, int | None]:
    """Имена входа и выхода ONNX-модели и зашитый размер входа. По этим именам к тензорам
    обращается и программа .tpu — компилятор LinQ сохраняет имена узлов исходного ONNX."""
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    return inp.name, out.name, inp.shape[2] if isinstance(inp.shape[2], int) else None


class LinqBackend(YoloRawBackend):
    """Инференс на российском ИИ-ускорителе LinQ TPU через Python API (модуль pytpu).

    Маршрут из документации LinQ:
        ONNX -> SDK (квантование, компиляция) -> программа .tpu
        кадр -> наш preprocess -> входной тензор -> pytpu: инференс на TPU -> выходной тензор -> наш postprocess

    YOLO компилятор LinQ поддерживает «обрезанной» перед концом головы (см. tools/compile_linq.py):
    на TPU — все свёртки (программа из *_head.onnx), а хвост *_tail.onnx (reshape, DFL, декодирование
    боксов) досчитываем на CPU через onnxruntime — это доли миллисекунды.

    Пре- и постобработка общие с OnnxBackend, поэтому результаты LinQ и эталона на CPU
    можно сверять напрямую (tools/compare.py).
    """
    name = "linq"

    def __init__(self, m: dict, linq: dict):
        module = linq.get("runtime_module") or "pytpu"
        try:
            tpu = importlib.import_module(module)
        except ImportError as e:
            raise RuntimeError(f"Python API LinQ ('{module}') не установлен — "
                               f"запускать на стенде LinQ (Jupyter hackathon.lin-q.ru)") from e

        # имена тензоров и размер входа берём из ONNX, из которого компилировали .tpu
        self.input_name, self.output_name, imgsz = read_onnx_io(linq.get("head_onnx") or linq["names_from"])
        super().__init__(m, read_onnx_names(linq["names_from"]), imgsz=imgsz)

        self.tail = None
        if linq.get("tail_onnx"):
            import onnxruntime as ort
            self.tail = ort.InferenceSession(linq["tail_onnx"], providers=["CPUExecutionProvider"])
            self.tail_inputs = [i.name for i in self.tail.get_inputs()]

        devices = tpu.Device.list_devices()
        if not devices:
            raise RuntimeError("Устройства TPU не найдены (проверь ls /dev/tpu*)")

        # В примере LinQ устройство, программа и сессия открываются через with.
        # Нам они нужны на всё время работы, поэтому держим их в ExitStack и закрываем в close().
        self._stack = contextlib.ExitStack()
        try:
            device = self._stack.enter_context(tpu.Device.open(devices[linq.get("device_id", 0)]))
            program = self._stack.enter_context(device.load(linq["model_path"]))
            self.inference = self._stack.enter_context(program.inference())
        except Exception:
            self._stack.close()
            raise
        log.info("LinQ: устройство %s, программа %s", devices[linq.get("device_id", 0)], linq["model_path"])

    def _forward(self, blob):
        # вход: float32 NCHW, 0..1 — как у ONNX-модели. Масштабирование в INT8 и упаковку
        # делает сам pytpu по метаданным программы .tpu.
        feeds = {self.input_name: blob}
        # в документации SDK вызов называется sync(), в примере API — run(); поддерживаем оба
        out = self.inference.sync(feeds) if hasattr(self.inference, "sync") else self.inference.run(feeds)
        if self.tail is None:
            return np.asarray(out[self.output_name], dtype=np.float32)
        tail_out = self.tail.run(None, {n: np.asarray(out[n], dtype=np.float32) for n in self.tail_inputs})
        return tail_out[0]   # (1, 4 + nc, N) — дальше обычный postprocess

    def close(self):
        self._stack.close()


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class CompositeBackend(Backend):
    """Несколько моделей на одном кадре, детекции объединяются.

    Зачем: обученная модель СИЗ хорошо видит людей, каски, жилеты, очки, но обучена на фото
    со стройплощадок — руки у станка она видит плохо. Вторая модель (YOLO-World с фиксированными
    классами «перчатка / голая рука») закрывает именно этот пробел. У каждой части свой
    фильтр классов (only) и переименование в наши классы (rename), чтобы части не дублировали друг друга.
    """
    name = "composite"

    def __init__(self, cfg: dict):
        self.parts = []
        for p in cfg["model"]["parts"]:
            sub = {**cfg, "model": {**cfg["model"], **p}}
            self.parts.append((make_backend(sub), {k.lower(): v for k, v in (p.get("rename") or {}).items()},
                               {c.lower() for c in p.get("only") or []}, p.get("zoom")))
        self.name = "+".join(b.name for b, _, _, _ in self.parts)

    @staticmethod
    def _keep(d, rename, only) -> bool:
        c = d.cls.lower()
        if only and c not in only:
            return False
        d.cls = rename.get(c, d.cls)
        return True

    def detect(self, frame):
        out = []
        for backend, rename, only, zoom in self.parts:
            if zoom:
                out += self._zoom(frame, backend, rename, only, zoom, out)
                continue
            out += [d for d in backend.detect(frame) if self._keep(d, rename, only)]
        # один и тот же объект от двух моделей (например, человек) — оставляем рамку с большей уверенностью
        out.sort(key=lambda d: -d.conf)
        kept: list[Detection] = []
        for d in out:
            if not any(k.cls == d.cls and _iou(k.box, d.box) > 0.5 for k in kept):
                kept.append(d)
        return kept

    def _zoom(self, frame, backend, rename, only, z: dict, found: list) -> list:
        """«Второй взгляд» на мелкие предметы: вырезаем голову и руки каждого найденного человека
        и прогоняем вырезку отдельно. На кадре 1920×1080, уменьшенном до 640, сигарета занимает
        пару пикселей и теряется, а на вырезке она видна. Работает на людях, которых нашли
        предыдущие части, — поэтому эта часть в списке последняя."""
        h, w = frame.shape[:2]
        persons = sorted((d for d in found if d.cls == "person"),
                         key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]), reverse=True)
        res = []
        for p in persons[:z.get("max_persons", 4)]:
            x1, y1, x2, y2 = p.box
            pw, ph = x2 - x1, y2 - y1
            if ph < z.get("min_height", 120):   # человек далеко — вырезка ничего не даст
                continue
            cx, half = (x1 + x2) / 2, pw * (0.5 + z.get("widen", 0.15))
            cx1, cx2 = int(max(0, cx - half)), int(min(w, cx + half))
            cy1, cy2 = int(max(0, y1 + z.get("top", 0.0) * ph)), int(min(h, y1 + z.get("bottom", 0.45) * ph))
            if cx2 - cx1 < 16 or cy2 - cy1 < 16:
                continue
            for d in backend.detect(frame[cy1:cy2, cx1:cx2]):
                if self._keep(d, rename, only):
                    b = d.box
                    d.box = (b[0] + cx1, b[1] + cy1, b[2] + cx1, b[3] + cy1)   # обратно в координаты кадра
                    res.append(d)
        return res


def make_backend(cfg: dict) -> Backend:
    m = cfg["model"]
    kind = m["backend"]
    if kind == "composite":
        return CompositeBackend(cfg)
    if kind == "ultralytics":
        return UltralyticsBackend(m)
    if kind == "onnx":
        return OnnxBackend(m)
    if kind == "linq":
        linq = cfg.get("linq", {})
        try:
            return LinqBackend(m, linq)
        except Exception as e:
            if not linq.get("fallback"):
                raise
            log.warning("LinQ недоступен (%s), работаю на запасном бэкенде '%s'", e, linq["fallback"])
            if linq["fallback"] == "onnx":
                return OnnxBackend(m, path=linq.get("fallback_path"))
            return UltralyticsBackend({**m, "path": linq.get("fallback_path") or m["path"]})
    raise ValueError(f"Неизвестный backend: {kind}")

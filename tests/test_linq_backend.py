"""LinqBackend против подставного pytpu, повторяющего API из примера LinQ (usage_example_python_api).

Подставной «TPU» внутри гоняет ту же ONNX-модель на CPU, поэтому детекции LinqBackend
должны совпасть с OnnxBackend один в один. Это проверяет всю обвязку без ускорителя.
"""
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends import LinqBackend, OnnxBackend, make_backend  # noqa: E402

ONNX = ROOT / "models" / "yolo11n.onnx"
IMAGE = ROOT / "data" / "sample.jpg"
pytestmark = pytest.mark.skipif(not ONNX.exists() or not IMAGE.exists(), reason="нет models/yolo11n.onnx")
M = {"imgsz": 640, "conf": 0.35, "iou": 0.5, "path": str(ONNX)}


def fake_pytpu(log: list, onnx_for_program=lambda path: ONNX, api="run"):
    """api="run" — как в usage_example_python_api, api="sync" — как в документации SDK."""
    import onnxruntime as ort

    class Inference:
        def __init__(self, sess):
            self.sess = sess

        def _infer(self, feeds):
            [(name, x)] = feeds.items()
            assert x.dtype == np.float32 and x.shape == (1, 3, 640, 640)   # формат входа как в примере LinQ
            out = self.sess.run(None, feeds)
            return {o.name: v for o, v in zip(self.sess.get_outputs(), out)}

    setattr(Inference, api, Inference._infer)

    class Ctx:   # всё в pytpu — контекстные менеджеры (with ...)
        def __init__(self, name, value):
            self.name, self.value = name, value

        def __enter__(self):
            log.append("open " + self.name)
            return self.value

        def __exit__(self, *a):
            log.append("close " + self.name)

    class Program:
        def __init__(self, path):
            self.path = path

        def inference(self):
            sess = ort.InferenceSession(str(onnx_for_program(self.path)), providers=["CPUExecutionProvider"])
            return Ctx("inference", Inference(sess))

    class Dev:
        def load(self, path):
            return Ctx("program", Program(path))

    class Device:
        @staticmethod
        def list_devices():
            return ["/dev/tpu0"]

        @staticmethod
        def open(path):
            return Ctx("device", Dev())

    return types.SimpleNamespace(Device=Device)


def test_linq_backend_matches_onnx(monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, "pytpu", fake_pytpu(log))
    linq = LinqBackend(M, {"runtime_module": "pytpu", "model_path": "models/yolo11n.tpu", "names_from": str(ONNX)})
    frame = cv2.imread(str(IMAGE))

    a, b = OnnxBackend(M).detect(frame), linq.detect(frame)
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert x.cls == y.cls and np.allclose(x.box, y.box) and abs(x.conf - y.conf) < 1e-6

    linq.close()
    assert log == ["open device", "open program", "open inference",
                   "close inference", "close program", "close device"]


def test_linq_falls_back_to_onnx_without_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytpu", None)   # import pytpu -> ImportError
    cfg = {"model": {**M, "backend": "linq"},
           "linq": {"runtime_module": "pytpu", "model_path": "x.tpu", "names_from": str(ONNX),
                    "fallback": "onnx", "fallback_path": str(ONNX)}}
    assert make_backend(cfg).name == "onnx"


def test_split_head_on_tpu_tail_on_cpu(monkeypatch, tmp_path):
    """Как будет на стенде: программа .tpu из головы, хвост на CPU, вызов inference.sync()."""
    import shutil
    from tools.compile_linq import split, verify_split

    src = tmp_path / "m.onnx"
    shutil.copy(ONNX, src)
    head, tail, cuts = split(src)
    assert len(cuts) == 6 and all("/model.23/" in c for c in cuts)   # только свёртки головы Detect
    assert verify_split(src, head, tail, cv2.imread(str(IMAGE))) < 1e-4

    monkeypatch.setitem(sys.modules, "pytpu", fake_pytpu([], onnx_for_program=lambda p: head, api="sync"))
    linq = LinqBackend(M, {"model_path": "m_head.tpu", "head_onnx": str(head), "tail_onnx": str(tail),
                           "names_from": str(src)})
    frame = cv2.imread(str(IMAGE))
    a, b = OnnxBackend(M).detect(frame), linq.detect(frame)
    assert [(d.cls, round(d.conf, 5)) for d in a] == [(d.cls, round(d.conf, 5)) for d in b]
    linq.close()

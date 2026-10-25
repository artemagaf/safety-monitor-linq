"""Тесты пре/постобработки и инструментов. Не требуют обученной модели и ускорителя."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backends import Detection, YoloRawBackend, letterbox  # noqa: E402
from tools.compare import match  # noqa: E402
from tools.prepare_dataset import CLASSES, convert_labels  # noqa: E402

M = {"imgsz": 640, "conf": 0.3, "iou": 0.5}


def raw_output(boxes_cxcywh, class_scores, nc):
    """Собирает тензор (1, 4 + nc, N), как его выдаёт YOLOv8/11."""
    n = len(boxes_cxcywh)
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for k, (b, (cls, score)) in enumerate(zip(boxes_cxcywh, class_scores)):
        out[0, :4, k] = b
        out[0, 4 + cls, k] = score
    return out


def test_letterbox_shape_and_padding():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    out, r, (left, top) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == 0.5 and left == 0 and top == 140


def test_postprocess_maps_boxes_back_to_original_frame():
    b = YoloRawBackend(M, {0: "person", 1: "helmet"})
    # кадр 1280x720 -> r = 0.5, pad сверху 140. Бокс человека (100,200)-(300,600) в исходных координатах
    # в letterbox: x 50..150, y 240..440 -> cx=100, cy=340, w=100, h=200
    out = raw_output([(100, 340, 100, 200)], [(0, 0.9)], nc=2)
    dets = b.postprocess(out, 0.5, (0, 140), (720, 1280, 3))
    assert len(dets) == 1
    assert dets[0].cls == "person"
    np.testing.assert_allclose(dets[0].box, (100, 200, 300, 600), atol=1e-3)


def test_postprocess_nms_and_threshold():
    b = YoloRawBackend(M, {0: "person", 1: "helmet"})
    out = raw_output(
        [(100, 100, 50, 50), (102, 101, 50, 50), (300, 300, 40, 40), (500, 500, 40, 40)],
        [(0, 0.9), (0, 0.8), (1, 0.7), (1, 0.1)],   # два дубля человека, каска, слабая детекция
        nc=2,
    )
    dets = b.postprocess(out, 1.0, (0, 0), (640, 640, 3))
    assert sorted(d.cls for d in dets) == ["helmet", "person"]


def test_postprocess_accepts_transposed_output():
    b = YoloRawBackend(M, {0: "person"})
    out = raw_output([(100, 100, 50, 50)] * 1 + [(400, 400, 50, 50)] * 9, [(0, 0.9)] * 10, nc=1)
    dets_a = b.postprocess(out, 1.0, (0, 0), (640, 640, 3))
    dets_b = b.postprocess(out.transpose(0, 2, 1), 1.0, (0, 0), (640, 640, 3))
    assert len(dets_a) == len(dets_b) == 2


def test_convert_labels_remaps_and_drops():
    names = ["Person", "Hardhat", "machinery"]
    out = convert_labels(["0 0.5 0.5 0.2 0.6", "1 0.5 0.25 0.1 0.1", "2 0.1 0.1 0.1 0.1"], names)
    assert [CLASSES[int(l.split()[0])] for l in out] == ["person", "helmet"]


def test_convert_labels_polygon_to_box():
    [line] = convert_labels(["0 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5"], ["person"])
    assert line == "0 0.200000 0.300000 0.200000 0.400000"


def test_derived_negatives_sh17_style():
    """Рука без перчатки -> bare_hand; рука в перчатке -> ничего (перчатка размечена сама);
    лицо без очков -> no_glasses."""
    names = ["Person", "Head", "Face", "Glasses", "Hands", "Gloves"]
    lines = [
        "4 0.20 0.70 0.05 0.05",   # рука 1 ...
        "5 0.20 0.70 0.05 0.05",   # ... в перчатке
        "4 0.40 0.70 0.05 0.05",   # рука 2 голая
        "2 0.30 0.20 0.06 0.08",   # лицо без очков
    ]
    out = sorted(CLASSES[int(l.split()[0])] for l in convert_labels(lines, names))
    assert out == ["bare_hand", "gloves", "no_glasses"]


def test_compare_match_same_class_only():
    a = [Detection((0, 0, 10, 10), 0.9, "person"), Detection((20, 20, 30, 30), 0.8, "helmet")]
    b = [Detection((1, 1, 10, 10), 0.85, "person"), Detection((20, 20, 30, 30), 0.8, "person")]
    m = match(a, b)
    assert len(m) == 1 and m[0][0].cls == "person"

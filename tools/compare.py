"""Сверка двух бэкендов на одних и тех же картинках: дают ли они одинаковые детекции.

Главная проверка после переноса модели на LinQ: ONNX на CPU — эталон, LinQ — проверяемый.
Квантование в INT8 немного меняет результаты, но совпадение детекций должно быть высоким (> 95%).

Пример:
    python tools/compare.py --images datasets/ppe/val/images --ref onnx:models/ppe.onnx --test linq
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.geometry import iou  # noqa: E402
from tools.benchmark import backend_from_spec  # noqa: E402


def match(ref, test, thr=0.5):
    """Жадно сопоставляет детекции одного класса по IoU. Возвращает пары (ref, test, iou)."""
    pairs = sorted(((iou(a.box, b.box), i, j) for i, a in enumerate(ref) for j, b in enumerate(test)
                    if a.cls == b.cls), reverse=True)
    used_r, used_t, out = set(), set(), []
    for score, i, j in pairs:
        if score < thr:
            break
        if i in used_r or j in used_t:
            continue
        used_r.add(i)
        used_t.add(j)
        out.append((ref[i], test[j], score))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--images", required=True)
    ap.add_argument("--ref", required=True, help="эталонный бэкенд, напр. onnx:models/ppe.onnx")
    ap.add_argument("--test", required=True, help="проверяемый бэкенд, напр. linq")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="детекции без пары с уверенностью < порог + margin считаем пограничными, а не ошибкой")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ref_b, test_b = backend_from_spec(cfg, args.ref), backend_from_spec(cfg, args.test)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    border = cfg["model"]["conf"] + args.margin

    n_ref = n_test = n_match = n_border = 0
    ious, dconf = [], []
    for p in paths[:args.limit]:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        a, b = ref_b.detect(frame), test_b.detect(frame)
        m = match(a, b)
        # Пограничные: объект едва прошёл порог у одного бэкенда и едва не прошёл у другого.
        # Небольшой сдвиг уверенности (квантование, другой resize) — норма, это не ошибка переноса.
        paired = {id(x) for x, _, _ in m} | {id(y) for _, y, _ in m}
        border_a = sum(1 for d in a if id(d) not in paired and d.conf < border)
        border_b = sum(1 for d in b if id(d) not in paired and d.conf < border)
        n_ref, n_test = n_ref + len(a) - border_a, n_test + len(b) - border_b
        n_match, n_border = n_match + len(m), n_border + border_a + border_b
        ious += [s for _, _, s in m]
        dconf += [abs(x.conf - y.conf) for x, y, _ in m]

    recall = min(1.0, n_match / n_ref) if n_ref else 1.0
    precision = min(1.0, n_match / n_test) if n_test else 1.0
    print(f"Картинок: {min(len(paths), args.limit)}; детекций: эталон {n_ref}, проверяемый {n_test}, совпало {n_match}"
          f" (+ пограничных у порога, не учтены: {n_border})")
    print(f"Совпадение с эталоном: recall {recall:.3f}, precision {precision:.3f}")
    if ious:
        print(f"Средний IoU совпавших боксов: {sum(ious) / len(ious):.3f}; "
              f"средняя разница уверенности: {sum(dconf) / len(dconf):.3f}")
    ok = recall >= 0.95 and precision >= 0.95
    print("ИТОГ:", "OK — бэкенды согласованы" if ok else "РАСХОЖДЕНИЕ — проверь пре/постобработку и квантование")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Экспорт обученной модели в ONNX — формат, который принимают компиляторы ИИ-ускорителей (в т.ч. LinQ).

Пример:
    python tools/export.py --weights models/ppe.pt --calib-from datasets/ppe/val/images --calib 200
Результат: models/ppe.onnx (+ models/calib/ — картинки для калибровки INT8, если нужны компилятору LinQ).
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "ppe.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12,
                    help="версия набора операций ONNX; компиляторы ускорителей часто поддерживают не самые новые")
    ap.add_argument("--calib-from", help="папка с картинками (обычно val) для калибровочного набора")
    ap.add_argument("--calib", type=int, default=200, help="сколько картинок взять для калибровки")
    args = ap.parse_args()

    from ultralytics import YOLO

    # Фиксированный размер входа и batch=1: ускорители почти всегда требуют статические формы тензоров.
    # NMS в граф не встраиваем (nms=False) — он выполняется на CPU в YoloRawBackend.postprocess.
    path = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                                     dynamic=False, batch=1, simplify=True, nms=False)
    dst = Path(args.weights).with_suffix(".onnx")
    if Path(path).resolve() != dst.resolve():
        shutil.move(path, dst)
    print(f"ONNX: {dst}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    i, o = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"  вход  {i.name}: {i.shape} {i.type}")
    print(f"  выход {o.name}: {o.shape} {o.type}   (1, 4 + число_классов, число_анкоров)")

    if args.calib_from:
        imgs = sorted(p for p in Path(args.calib_from).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        random.Random(0).shuffle(imgs)
        calib = ROOT / "models" / "calib"
        calib.mkdir(parents=True, exist_ok=True)
        for p in imgs[:args.calib]:
            shutil.copy2(p, calib / p.name)
        print(f"Калибровочный набор: {calib} ({min(len(imgs), args.calib)} шт.)")


if __name__ == "__main__":
    main()

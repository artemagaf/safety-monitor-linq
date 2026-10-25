"""Дообучение YOLO на датасете СИЗ и оценка качества.

Пример (на GPU в Colab/Kaggle — минут 30–60 на 50 эпох для yolo11n):
    python tools/train.py --data datasets/ppe/data.yaml --model yolo11n.pt --epochs 50 --device 0
Результат: models/ppe.pt + метрики mAP по классам на test (или val).
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt", help="стартовые веса: yolo11n/s/m.pt (n — самая быстрая)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="0 — первая GPU, cpu — процессор")
    ap.add_argument("--name", default="ppe")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(ROOT / "runs"), name=args.name, exist_ok=True, resume=args.resume,
        patience=15,          # ранняя остановка, если mAP не растёт 15 эпох
        cos_lr=True,
        close_mosaic=10,      # последние эпохи без mosaic — модель привыкает к «обычным» кадрам
        fliplr=0.5, hsv_v=0.5,  # отражения и разная яркость: в цехах тёмные и пересвеченные зоны
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / "ppe.pt")
    print(f"Лучшие веса: {ROOT / 'models' / 'ppe.pt'}")

    import yaml
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    split = "test" if "test" in data else "val"
    m = YOLO(best).val(data=args.data, split=split, imgsz=args.imgsz, device=args.device,
                       project=str(ROOT / "runs"), name=f"{args.name}_eval", exist_ok=True)

    print(f"\nКачество на {split}: mAP50 = {m.box.map50:.3f}, mAP50-95 = {m.box.map:.3f}")
    print(f"{'класс':<12}{'P':>7}{'R':>7}{'mAP50':>8}")
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, _ = m.box.class_result(i)
        print(f"{m.names[int(c)]:<12}{p:>7.3f}{r:>7.3f}{ap50:>8.3f}")


if __name__ == "__main__":
    main()

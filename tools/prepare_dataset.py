"""Сборка единого датасета СИЗ из нескольких датасетов в формате YOLO.

Поддерживаются три вида папок:
  - Roboflow / Kaggle: <src>/train/images, <src>/train/labels, <src>/valid/..., <src>/data.yaml
  - Ultralytics (Construction-PPE): <src>/images/train, <src>/labels/train, <src>/data.yaml
  - SH17 (производство, станочники): <src>/images, <src>/labels, <src>/train_files.txt, <src>/val_files.txt
    и имена классов в <src>/data.yaml или sh17.yaml

У разных датасетов разные имена классов ("Hardhat", "helmet", "Glasses"...).
Скрипт приводит их к единому списку CLASSES, выбрасывает лишние классы
(техника, обувь, инструмент) и печатает статистику — по ней видно дисбаланс классов.

Негативные классы (no_helmet, no_glasses, bare_hand) в датасетах обычно не размечены — есть просто
«голова», «лицо», «руки». Мы их ВЫВОДИМ: голова, на которой нет каски -> no_helmet; лицо без очков ->
no_glasses; рука без перчатки -> bare_hand. Так модель учится видеть само нарушение (см. EXPLANATION.md).

Пример:
    python tools/prepare_dataset.py --src raw/sh17 --src raw/construction-safety --out datasets/ppe
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

CLASSES = [
    "person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask", "cigarette",
    "glasses", "no_glasses", "gloves", "bare_hand",
]

# имя класса в исходном датасете (нижний регистр) -> имя в CLASSES
SYNONYMS = {
    "person": "person", "worker": "person", "people": "person",
    "helmet": "helmet", "hardhat": "helmet", "hard-hat": "helmet", "hard hat": "helmet", "head_helmet": "helmet",
    "no-hardhat": "no_helmet", "no_helmet": "no_helmet", "no-helmet": "no_helmet",
    "without helmet": "no_helmet", "head_nohelmet": "no_helmet",
    "vest": "vest", "safety vest": "vest", "safety-vest": "vest", "reflective": "vest",
    "no-safety vest": "no_vest", "no_vest": "no_vest", "no-vest": "no_vest",
    "mask": "mask", "respirator": "mask", "with_mask": "mask", "face-mask-medical": "mask",
    "no-mask": "no_mask", "no_mask": "no_mask", "without_mask": "no_mask",
    "cigarette": "cigarette", "smoking": "cigarette", "cigar": "cigarette",
    "glasses": "glasses", "safety glasses": "glasses", "goggles": "glasses", "face-guard": "glasses",
    "no_glasses": "no_glasses", "no-goggles": "no_glasses", "no_goggle": "no_glasses", "no_goggles": "no_glasses",
    "gloves": "gloves", "glove": "gloves",
    "no_gloves": "bare_hand", "no-gloves": "bare_hand", "bare_hand": "bare_hand",
}

# «часть тела» -> (какое СИЗ на ней ищем, какой негативный класс получится, если СИЗ на ней нет)
DERIVED = {
    "head": ("helmet", "no_helmet"),
    "face": ("glasses", "no_glasses"),
    "hands": ("gloves", "bare_hand"), "hand": ("gloves", "bare_hand"),
}

SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_names(src: Path) -> list[str]:
    for name in ("data.yaml", "sh17.yaml", "dataset.yaml"):
        if (src / name).exists():
            names = yaml.safe_load((src / name).read_text(encoding="utf-8"))["names"]
            return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    raise FileNotFoundError(f"в {src} нет data.yaml / sh17.yaml с именами классов")


def parse_line(line: str) -> tuple[int, list[float]] | None:
    """Строка разметки YOLO -> (id, [cx, cy, w, h]). Полигоны (сегментация) превращаем в бокс."""
    parts = line.split()
    if len(parts) < 5:
        return None
    coords = list(map(float, parts[1:]))
    if len(coords) > 4:   # x1 y1 x2 y2 ... — полигон
        xs, ys = coords[0::2], coords[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        coords = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return int(parts[0]), coords


def _inside(center: list[float], box: list[float], pad: float = 0.25) -> bool:
    """Центр СИЗ внутри бокса части тела (с запасом pad: каска выступает над головой, перчатка — за кисть)."""
    cx, cy, w, h = box
    return abs(center[0] - cx) <= w * (0.5 + pad) and abs(center[1] - cy) <= h * (0.5 + pad)


def convert_labels(lines: list[str], src_names: list[str]) -> list[str]:
    """Разметка одной картинки -> строки с нашими классами, включая выведенные негативы."""
    objs = [(src_names[c].lower(), box) for c, box in filter(None, map(parse_line, lines)) if c < len(src_names)]
    out = []
    for name, box in objs:
        if name in DERIVED:
            ppe, neg = DERIVED[name]
            has_ppe = any(SYNONYMS.get(n) == ppe and _inside(b[:2], box) for n, b in objs)
            if not has_ppe:
                out.append((neg, box))
        elif name in SYNONYMS:
            out.append((SYNONYMS[name], box))
    return [f"{CLASSES.index(c)} " + " ".join(f"{v:.6f}" for v in box) for c, box in out]


def iter_split_images(src: Path):
    """Отдаёт (split, путь_к_картинке, путь_к_разметке) для обоих форматов датасетов."""
    found = False
    for split_dir, split in SPLITS.items():
        img_dir, lbl_dir = src / split_dir / "images", src / split_dir / "labels"
        if not img_dir.is_dir():   # формат Ultralytics: images/<split>, labels/<split>
            img_dir, lbl_dir = src / "images" / split_dir, src / "labels" / split_dir
        if img_dir.is_dir():
            found = True
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() in IMG_EXT:
                    yield split, img, lbl_dir / (img.stem + ".txt")
    if found:
        return
    # формат SH17: общие папки images/labels + списки файлов по выборкам
    for split_dir, split in SPLITS.items():
        lst = src / f"{split_dir}_files.txt"
        if not lst.exists():
            continue
        for line in lst.read_text(encoding="utf-8").splitlines():
            name = Path(line.strip()).name
            if name:
                yield split, src / "images" / name, src / "labels" / (Path(name).stem + ".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="папка датасета (можно несколько)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    stats = {s: Counter() for s in ("train", "val", "test")}
    images = Counter()

    for si, src in enumerate(map(Path, args.src)):
        names = read_names(src)
        used = [n for n in names if n.lower() in SYNONYMS or n.lower() in DERIVED]
        dropped = [n for n in names if n not in used]
        print(f"[{src.name}] классы: {names}\n  -> используются: {used}\n  -> отброшены: {dropped or 'нет'}")

        for split, img, lbl in iter_split_images(src):
            if not img.exists():
                continue
            lines = convert_labels(lbl.read_text().splitlines(), names) if lbl.exists() else []
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            stem = f"s{si}_{img.stem}"   # префикс источника, чтобы не было совпадений имён
            shutil.copy2(img, out / split / "images" / (stem + img.suffix))
            (out / split / "labels" / (stem + ".txt")).write_text("\n".join(lines))
            images[split] += 1
            stats[split].update(CLASSES[int(l.split()[0])] for l in lines)

    splits = {s: f"{s}/images" for s in ("train", "val", "test") if images[s]}
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), **splits, "names": dict(enumerate(CLASSES))}, allow_unicode=True))

    print("\nИзображений:", dict(images))
    print(f"{'класс':<12}" + "".join(f"{s:>9}" for s in stats))
    for c in CLASSES:
        print(f"{c:<12}" + "".join(f"{stats[s][c]:>9}" for s in stats))
    print(f"\nГотово: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()

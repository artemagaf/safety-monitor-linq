"""Правила: по людям (трекам) и найденным предметам определяем нарушения на одном кадре.

Модель находит объекты по отдельности: человек, каска, «голова без каски», жилет, сигарета.
Здесь мы решаем, КОМУ принадлежит каждый предмет, и что из этого следует.
"""
from collections import defaultdict
from dataclasses import dataclass

from .backends import Detection
from .geometry import center, contains, foot_point, point_in_polygon, sub_region
from .tracker import Track

VIOLATION_LABELS = {
    "no_helmet": "Нет каски",
    "no_vest": "Нет сигнального жилета",
    "no_mask": "Нет маски / респиратора",
    "no_glasses": "Нет защитных очков",
    "no_gloves": "Нет перчаток",
    "forbidden_gloves": "Работа в перчатках у вращающихся частей станка",
    "restricted_zone": "Нахождение в опасной зоне",
    "smoking": "Курение",
    "phone_use": "Телефон при работе с оборудованием",
    "vehicle_proximity": "Опасное сближение с техникой",
    "running": "Бег по цеху",
    "fall": "Падение, человек лежит неподвижно",
    "open_fire": "Открытый огонь",
    "climbing": "Перелезание через ограждение",
    "no_harness": "Работа на высоте без страховочной привязи",
    "unsafe_carry": "Перенос груза с нарушением правил (закрыт обзор)",
    "bent_lift": "Подъём груза с наклоном спины (поднимать ногами, спина прямая)",
}

# СИЗ: (роль «надето», роль «не надето»)
PPE = {
    "helmet": ("helmet", "no_helmet"),
    "vest": ("vest", "no_vest"),
    "mask": ("mask", "no_mask"),
    "glasses": ("glasses", "no_glasses"),
    "gloves": ("gloves", "bare_hand"),
}

# В какой части тела человека ищем предмет (доли высоты бокса: 0 = макушка, 1 = ноги)
ITEM_REGION = {
    "helmet": (0.0, 0.35), "no_helmet": (0.0, 0.35),
    "mask": (0.0, 0.4), "no_mask": (0.0, 0.4),
    "glasses": (0.0, 0.35), "no_glasses": (0.0, 0.35),
    "vest": (0.1, 0.85), "no_vest": (0.1, 0.85),
    "gloves": (0.1, 1.0), "bare_hand": (0.1, 1.0),
    "cigarette": (0.0, 0.75),
    "phone": (0.0, 0.8),   # у уха или в руках перед собой
    "harness": (0.1, 0.8),  # привязь — на торсе
    "box": (-0.2, 1.0),     # груз в руках — от «над головой» до ног
}
# Руки часто вытянуты к станку и выходят за бокс человека — расширяем поиск по горизонтали сильнее
ITEM_WIDEN = {"gloves": 0.5, "bare_hand": 0.5, "phone": 0.3, "box": 0.4}


@dataclass(frozen=True)
class Violation:
    type: str
    detail: str = ""   # например, название зоны


PLACE_ID = 0   # нарушения места (огонь): не привязаны к человеку


def box_gap(a, b) -> float:
    """Расстояние между краями двух рамок (0 — касаются или пересекаются)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def assign_items(persons: list[Track], items: list[Detection]) -> dict[int, dict[str, float]]:
    """Привязывает предметы к людям. Возвращает {track_id: {роль: макс. уверенность}}.

    Предмет принадлежит человеку, если его центр попал в нужную часть тела
    (каска — в верхнюю треть и т.д.). Если подходят несколько людей — берём того,
    у кого центр этой части тела ближе.
    """
    found: dict[int, dict[str, float]] = defaultdict(dict)
    for item in items:
        region = ITEM_REGION.get(item.role)
        if region is None:
            continue
        c = center(item.box)
        best, best_dist = None, float("inf")
        for p in persons:
            reg = sub_region(p.box, *region, widen=ITEM_WIDEN.get(item.role, 0.15))
            if contains(reg, c):
                rc = center(reg)
                dist = (rc[0] - c[0]) ** 2 + (rc[1] - c[1]) ** 2
                if dist < best_dist:
                    best, best_dist = p, dist
        if best is not None:
            found[best.id][item.role] = max(found[best.id].get(item.role, 0.0), item.conf)
    return found


def evaluate(persons: list[Track], items: list[Detection], zones: list[dict], rules: dict,
             frame_size: tuple[int, int]) -> dict[int, list[Violation]]:
    """Нарушения на текущем кадре: {track_id: [Violation, ...]} (пустой список = всё в порядке)."""
    w, h = frame_size
    found = assign_items(persons, items)
    vehicles = [d for d in items if d.role == "vehicle"]
    boxes = [d for d in items if d.role == "box"]
    result: dict[int, list[Violation]] = {}

    for p in persons:
        out: list[Violation] = []
        fx, fy = foot_point(p.box)
        need = set(rules.get("ppe_required", []))
        forbid: dict[str, str] = {x: "" for x in rules.get("forbid", [])}   # запрещено -> где (зона)

        for z in zones:
            if not point_in_polygon((fx / w, fy / h), z["polygon"]):
                continue
            if z["type"] == "restricted":
                out.append(Violation("restricted_zone", z["name"]))
            elif z["type"] in ("ppe", "machine"):   # machine — рабочее место у станка
                need |= set(z.get("require", []))
                forbid.update({item: z["name"] for item in z.get("forbid", [])})
            elif z["type"] == "height":   # работа на высоте: обязательна страховочная привязь
                if p.box[3] - p.box[1] >= rules.get("min_person_height", 0) and                         "harness" not in found.get(p.id, {}):
                    out.append(Violation("no_harness", z["name"]))

        # ограждение: перелезает — корпус (центр рамки), а не только ноги, оказался в полосе ограждения
        cx, cy = center(p.box)
        for z in zones:
            if z["type"] == "barrier" and point_in_polygon((cx / w, cy / h), z["polygon"]):
                out.append(Violation("climbing", z["name"]))
                break

        # далёких людей на СИЗ не проверяем — на маленьком боксе каску не разглядеть
        if p.box[3] - p.box[1] >= rules.get("min_person_height", 0):
            have = found.get(p.id, {})
            for item in sorted(need - set(forbid)):
                if item not in PPE:
                    continue
                pos, neg = PPE[item]
                pos_conf, neg_conf = have.get(pos, 0.0), have.get(neg, 0.0)
                if neg_conf > pos_conf or (rules.get("ppe_mode") == "absence" and pos_conf == 0.0):
                    out.append(Violation(f"no_{item}"))
            # Запрещённое СИЗ: например, перчатки на токарном станке — их может затянуть во вращающийся патрон
            for item, place in sorted(forbid.items()):
                if item not in PPE:
                    continue
                pos, neg = PPE[item]
                if have.get(pos, 0.0) > have.get(neg, 0.0):
                    out.append(Violation(f"forbidden_{item}", place))

        # телефон: запрещён на всей камере (forbid у камеры) или в зоне станка
        if "phone" in forbid and "phone" in found.get(p.id, {}):
            out.append(Violation("phone_use", forbid["phone"]))

        if "cigarette" in found.get(p.id, {}):
            out.append(Violation("smoking"))

        # груз закрывает голову (несёт перед лицом или над головой) — не видит, куда идёт
        head = sub_region(p.box, 0.0, 0.25, widen=0.0)
        for b in boxes:
            if box_gap(b.box, head) == 0 and center(b.box)[1] < p.box[1] + 0.45 * (p.box[3] - p.box[1]):
                out.append(Violation("unsafe_carry"))
                break

        # техника: зазор между человеком и машиной меньше vehicle_gap ростов человека
        gap_limit = rules.get("vehicle_gap", 0.3) * (p.box[3] - p.box[1])
        for v in vehicles:
            if box_gap(p.box, v.box) <= gap_limit:
                out.append(Violation("vehicle_proximity", v.cls))
                break

        result[p.id] = out

    # открытый огонь — нарушение места, а не человека: тревога и без людей в кадре.
    # Номер 0 = «место» (у людей номера с 1). Ключ 0 есть всегда, чтобы сглаживание видело, что огня уже нет.
    fires = [d for d in items if d.role == "fire"]
    result[PLACE_ID] = [Violation("open_fire", f"{max(f.conf for f in fires):.2f}")] if fires else []
    return result

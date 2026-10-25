"""Простая геометрия для боксов (x1, y1, x2, y2) и полигонов зон."""

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def foot_point(box: Box) -> tuple[float, float]:
    """Точка «ног» человека — середина нижней стороны бокса. По ней определяем, в какой зоне он стоит.
    Чуть выше самого края (на 1% высоты): у станочника ноги часто срезаны нижней границей кадра,
    и точка ровно на краю попадала бы на границу зоны, а не внутрь."""
    return (box[0] + box[2]) / 2, box[3] - 0.01 * (box[3] - box[1])


def sub_region(box: Box, top: float, bottom: float, widen: float = 0.15) -> Box:
    """Вертикальная полоса бокса человека: top/bottom — доли высоты (0 = макушка, 1 = ноги).
    widen расширяет полосу по горизонтали, т.к. каска/жилет часто чуть шире бокса человека."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 - w * widen, y1 + h * top, x2 + w * widen, y1 + h * bottom


def contains(box: Box, pt: tuple[float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def point_in_polygon(pt: tuple[float, float], poly: list[list[float]]) -> bool:
    """Алгоритм трассировки луча (ray casting)."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside

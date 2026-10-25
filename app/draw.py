"""Отрисовка зон, людей и нарушений на кадре.

cv2.putText не умеет кириллицу, поэтому подписи на видео латиницей,
а русские названия показывает веб-интерфейс.
"""
import cv2
import numpy as np

SHORT = {
    "no_helmet": "NO HELMET",
    "no_vest": "NO VEST",
    "no_mask": "NO MASK",
    "no_glasses": "NO GLASSES",
    "no_gloves": "NO GLOVES",
    "forbidden_gloves": "GLOVES AT LATHE",
    "restricted_zone": "DANGER ZONE",
    "smoking": "SMOKING",
    "phone_use": "PHONE",
    "vehicle_proximity": "NEAR VEHICLE",
    "running": "RUNNING",
    "fall": "FALL",
    "open_fire": "FIRE",
    "bent_lift": "BENT LIFT",
    "climbing": "CLIMBING",
    "no_harness": "NO HARNESS",
    "unsafe_carry": "LOAD BLOCKS VIEW",
}
BAD_ITEMS = {"no_helmet", "no_vest", "no_mask", "no_glasses", "bare_hand", "cigarette", "phone", "vehicle", "fire"}
GREEN, RED, YELLOW, BLUE, GRAY = (60, 180, 75), (40, 40, 230), (0, 200, 255), (230, 140, 40), (200, 200, 200)


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw(frame, tracks, items, zones, violations, header: str):
    img = frame.copy()
    h, w = img.shape[:2]

    overlay = img.copy()
    for z in zones:
        pts = (np.array(z["polygon"]) * [w, h]).astype(np.int32)
        color = {"restricted": RED, "barrier": RED, "machine": BLUE}.get(z["type"], YELLOW)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)
    img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)

    for d in items:
        x1, y1, x2, y2 = map(int, d.box)
        color = RED if d.role in BAD_ITEMS else GRAY
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, d.role, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        vs = violations.get(t.id, [])
        color = RED if vs else GREEN
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"#{t.id}" + (": " + ", ".join(SHORT.get(v.type, v.type) for v in vs) if vs else "")
        _label(img, text, x1, y1, color)

    _label(img, header, 0, 22, (40, 40, 40))
    return img

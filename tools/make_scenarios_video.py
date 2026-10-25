"""Демо-ролик со сценариями поведения для проверки и показа: техника рядом с людьми, бег, падение, телефон.

Ролик собирается из наших же кадров (своих видео с производства у команды нет):
  1. data/sample.jpg — люди вплотную к автобусу              -> «опасное сближение с техникой»
  2. человек из sample.jpg быстро пересекает кадр цеха       -> «бег по цеху»
  3. тот же человек лежит на полу цеха                      -> «падение, лежит неподвижно»
  4. фото SH17 с телефоном в руке                           -> «телефон при работе»
Это постановка, а не съёмка реальных происшествий — так и надо говорить на защите.

    python tools/make_scenarios_video.py   ->  data/scenarios.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 10


def fit(img):
    """Вписать картинку в кадр 1280×720 с тёмными полями."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    out = np.full((H, W, 3), 20, np.uint8)
    y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def caption(frame, text):
    f = frame.copy()
    cv2.rectangle(f, (0, H - 34), (W, H), (20, 20, 20), -1)
    cv2.putText(f, text, (14, H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
    return f


def main():
    out_path = ROOT / "data" / "scenarios.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    sample = cv2.imread(str(ROOT / "data" / "sample.jpg"))
    person = sample[399:904, 47:239]                       # человек слева на sample.jpg
    shop = cv2.resize(cv2.imread(str(ROOT / "data" / "samples" / "laser.png")), (W, H))

    def hold(frame, sec, text):
        for _ in range(int(sec * FPS)):
            vw.write(caption(frame, text))

    # 1. техника
    hold(fit(sample), 10, "1/4  people next to a vehicle")

    # 2. бег: рост 150 px, скорость ~285 px/с ≈ 1.9 роста в секунду; проход ~4 с, чтобы трекер
    #    успел подтвердить человека и набрать историю для измерения скорости
    runner = cv2.resize(person, (int(person.shape[1] * 150 / person.shape[0]), 150))
    rh, rw = runner.shape[:2]
    for k in range(2):
        n = int(4.2 * FPS)
        for i in range(n):
            x = int((W - rw) * (i / (n - 1) if k % 2 == 0 else 1 - i / (n - 1)))
            f = shop.copy()
            f[H - 60 - rh:H - 60, x:x + rw] = runner
            vw.write(caption(f, "2/4  running across the shop floor"))
    hold(shop, 2, "2/4  running across the shop floor")

    # 3. падение: человек лежит на полу цеха
    lie = cv2.rotate(person, cv2.ROTATE_90_CLOCKWISE)
    lie = cv2.resize(lie, (500, int(lie.shape[0] * 500 / lie.shape[1])))
    f = shop.copy()
    f[700 - lie.shape[0]:700, 40:40 + lie.shape[1]] = lie
    hold(f, 14, "3/4  person lying on the floor")

    # 4. телефон
    for name in ("pexels-photo-8374249.jpg", "pexels-photo-4963444.jpg"):   # телефон виден и в кадре 1280×720
        phone = ROOT / "data" / "sh17" / name
        if phone.exists():
            hold(fit(cv2.imread(str(phone))), 8, "4/4  phone in hand")
    vw.release()
    print(f"Готово: {out_path}")


if __name__ == "__main__":
    sys.exit(main())

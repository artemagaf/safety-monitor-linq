"""Поведение во времени (задача №1 кейса): падение и неподвижность, бег.

Работает поверх трекера: у каждого человека (track id) хранится короткая история положений.

Бег. Скорость центра человека за последнюю секунду, в ростах человека в секунду. Рост берём
из высоты рамки — так порог не зависит от того, далеко человек от камеры или близко.
Шаг ≈ 0.5–1 роста/с, бег — от ~1.5.

Падение. Человек лежит, если его тело горизонтально. Широкая рамка — только кандидат:
сидящий у пульта или срезанный кадром человек тоже даёт широкую рамку. Поэтому для кандидатов
запускаем модель позы (YOLO11-pose, 17 точек скелета) и проверяем линию «плечи — бёдра»:
лежит, если она ближе к горизонтали, чем к вертикали. Не видно бёдер — не считаем падением
(меньше ложных тревог). Лежит дольше fall_sec — тревога «падение / неподвижность».
Модель позы вызывается только когда есть кандидат, то есть почти никогда — это дёшево.

Подъём груза согнувшись. Правильно: присесть, спина прямая, поднимать ногами, груз у пояса.
Неправильно: ноги прямые, спина наклонена — так срывают поясницу. По скелету меряем наклон
туловища (линия «бёдра — плечи» от вертикали) и сгиб колен (180° минус угол «бедро — колено —
лодыжка»). Наклон ≥ lift_trunk_deg и колени почти прямые (сгиб ≤ lift_knee_deg) дольше
lift_sec — тревога. Колени закрыты грузом — решаем по наклону. Если lift_needs_load, тревога
только когда рядом с человеком найден груз (коробка); вес камера, понятно, не измеряет.
На рисунках: «мёртвая тяга» — наклон 56–63°, правильный присед — наклон 0°, колени 85–89°.
"""
from __future__ import annotations

import logging
import math
from collections import deque

from .geometry import iou
from .rules import box_gap
from .rules import Violation

log = logging.getLogger(__name__)

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12   # индексы точек скелета COCO
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16


def _angle(a, b, c) -> float:
    """Угол abc в градусах."""
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))))


class BehaviorAnalyzer:
    def __init__(self, cfg: dict | None = None, pose_model=None):
        c = cfg or {}
        self.run_speed = c.get("run_speed", 1.5)        # ростов в секунду
        self.run_window = c.get("run_window_sec", 1.0)  # за какой отрезок меряем скорость
        self.fall_ratio = c.get("fall_ratio", 1.15)     # ширина / высота рамки — кандидат в лежащие
        self.fall_sec = c.get("fall_sec", 3.0)          # сколько лежать до тревоги
        self.kp_conf = c.get("pose_kp_conf", 0.3)
        self.lift_trunk = c.get("lift_trunk_deg", 55)  # наклон спины от вертикали
        self.lift_knee = c.get("lift_knee_deg", 40)     # сгиб колен: меньше — ноги прямые
        self.lift_sec = c.get("lift_sec", 1.0)
        self.lift_needs_load = c.get("lift_needs_load", False)
        self.pose_model = pose_model                    # None — проверка позы отключена (только по рамке)
        self.hist: dict[int, deque] = {}
        self.lying_since: dict[int, float] = {}
        self.bent_since: dict[int, float] = {}

    # ---------- бег ----------

    def speed(self, tid: int) -> float:
        """Скорость в ростах человека в секунду за последние run_window секунд."""
        h = self.hist.get(tid)
        if not h or len(h) < 2:
            return 0.0
        t1, x1, y1, hh1 = h[-1]
        t0, x0, y0, _ = next((p for p in h if t1 - p[0] <= self.run_window * 1.5), h[0])
        dt = t1 - t0
        if dt < self.run_window * 0.5:
            return 0.0
        heights = sorted(p[3] for p in h)
        body = heights[len(heights) // 2] or 1.0
        return math.hypot(x1 - x0, y1 - y0) / dt / body

    # ---------- падение ----------

    def _is_lying(self, box, poses) -> bool:
        w, h = box[2] - box[0], box[3] - box[1]
        if w < self.fall_ratio * h:
            return False
        if self.pose_model is None:
            return True
        best = max(poses, key=lambda p: iou(p[0], box), default=None)
        if best is None or iou(best[0], box) < 0.3:
            return False
        kp, conf = best[1], best[2]
        if min(conf[L_SH], conf[R_SH]) < self.kp_conf and max(conf[L_SH], conf[R_SH]) < self.kp_conf:
            return False
        if max(conf[L_HIP], conf[R_HIP]) < self.kp_conf:
            return False   # бёдер не видно (сидит за пультом, срезан кадром) — не падение
        sh = [kp[i] for i in (L_SH, R_SH) if conf[i] >= self.kp_conf]
        hp = [kp[i] for i in (L_HIP, R_HIP) if conf[i] >= self.kp_conf]
        sx, sy = sum(p[0] for p in sh) / len(sh), sum(p[1] for p in sh) / len(sh)
        hx, hy = sum(p[0] for p in hp) / len(hp), sum(p[1] for p in hp) / len(hp)
        if abs(sy - hy) >= abs(sx - hx):
            return False   # туловище ближе к вертикали
        # согнулся стоя (подъём груза) — туловище тоже почти горизонтально, но ноги вертикальны
        an = [kp[i] for i in (L_ANK, R_ANK) if conf[i] >= self.kp_conf]
        if an:
            ax, ay = sum(p[0] for p in an) / len(an), sum(p[1] for p in an) / len(an)
            if ay - hy > abs(ax - hx):
                return False
        return True

    # ---------- подъём груза ----------

    def _pose_of(self, box, poses):
        best = max(poses, key=lambda p: iou(p[0], box), default=None)
        return best if best is not None and iou(best[0], box) >= 0.3 else None

    def lift_angles(self, kp, conf) -> tuple[float, float | None] | None:
        """(наклон туловища от вертикали, сгиб колен) в градусах; None — плечи или бёдра не видны."""
        ok = lambda *ids: [kp[i] for i in ids if conf[i] >= self.kp_conf]
        sh, hp = ok(L_SH, R_SH), ok(L_HIP, R_HIP)
        if not sh or not hp:
            return None
        sx, sy = sum(p[0] for p in sh) / len(sh), sum(p[1] for p in sh) / len(sh)
        hx, hy = sum(p[0] for p in hp) / len(hp), sum(p[1] for p in hp) / len(hp)
        if sy >= hy:   # плечи ниже бёдер — это не наклон, а что-то другое (лежит, кувырок)
            return None
        trunk = math.degrees(math.atan2(abs(sx - hx), hy - sy))
        knees = [180 - _angle(kp[h], kp[k], kp[a]) for h, k, a in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK))
                 if min(conf[h], conf[k], conf[a]) >= self.kp_conf]
        return trunk, (max(knees) if knees else None)

    def _is_bent(self, box, poses) -> bool:
        best = self._pose_of(box, poses)
        a = self.lift_angles(best[1], best[2]) if best else None
        if a is None:
            return False
        trunk, knee = a
        return trunk >= self.lift_trunk and (knee is None or knee <= self.lift_knee)

    def _poses(self, frame):
        r = self.pose_model.predict(frame, conf=0.25, verbose=False)[0]
        if r.keypoints is None or not len(r.boxes):
            return []
        return [(tuple(b), kp, cf) for b, kp, cf in
                zip(r.boxes.xyxy.tolist(), r.keypoints.xy.tolist(), r.keypoints.conf.tolist())]

    # ---------- общий вход ----------

    def update(self, t: float, tracks, frame=None, items=None) -> dict[int, list[Violation]]:
        """t — время кадра в секундах (для видеофайла — время в ролике, для камеры — часы).
        items — предметы с кадра: по коробкам решаем, с грузом ли человек (lift_needs_load)."""
        out: dict[int, list[Violation]] = {}
        seen = set()
        wide = [tr for tr in tracks if (tr.box[2] - tr.box[0]) >= self.fall_ratio * (tr.box[3] - tr.box[1])]
        loads = [d for d in items or [] if d.role == "box"]
        lifters = [tr for tr in tracks if (not self.lift_needs_load or any(
            box_gap(tr.box, d.box) <= 0.2 * (tr.box[3] - tr.box[1]) for d in loads))]
        need_pose = (wide or lifters) and self.pose_model is not None and frame is not None
        poses = self._poses(frame) if need_pose else []

        for tr in tracks:
            seen.add(tr.id)
            x1, y1, x2, y2 = tr.box
            h = self.hist.setdefault(tr.id, deque(maxlen=30))
            if h and t < h[-1][0]:   # видео пошло по кругу — история больше не про этого человека
                h.clear()
                self.lying_since.pop(tr.id, None)
            h.append((t, (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1))
            while h and t - h[0][0] > 5:
                h.popleft()

            vs = []
            if self.speed(tr.id) >= self.run_speed:
                vs.append(Violation("running"))
            if tr in wide and self._is_lying(tr.box, poses):
                since = self.lying_since.setdefault(tr.id, t)
                if t - since >= self.fall_sec:
                    vs.append(Violation("fall", f"{t - since:.0f} с"))
            else:
                self.lying_since.pop(tr.id, None)
            if tr in lifters and poses and self._is_bent(tr.box, poses):
                since = self.bent_since.setdefault(tr.id, t)
                if t - since >= self.lift_sec:
                    vs.append(Violation("bent_lift"))
            else:
                self.bent_since.pop(tr.id, None)
            out[tr.id] = vs

        for tid in [k for k in self.hist if k not in seen and (not self.hist[k] or t - self.hist[k][-1][0] > 10)]:
            del self.hist[tid]
            self.lying_since.pop(tid, None)
            self.bent_since.pop(tid, None)
        return out

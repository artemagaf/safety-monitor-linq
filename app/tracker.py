"""Простой IoU-трекер: даёт каждому человеку стабильный ID между кадрами.

Нужен, чтобы считать нарушение «одного и того же человека» во времени и не слать
одно и то же событие на каждом кадре. Работает с любым бэкендом, т.к. принимает обычные Detection.
"""
from dataclasses import dataclass

from .backends import Detection
from .geometry import Box, center, iou


@dataclass
class Track:
    id: int
    box: Box
    conf: float
    hits: int = 1     # сколько раз трек был найден
    missed: int = 0   # сколько кадров подряд не найден


class IouTracker:
    def __init__(self, iou_thr: float = 0.3, max_missed: int = 10, min_hits: int = 2, max_dist: float = 1.5):
        self.iou_thr, self.max_missed, self.min_hits, self.max_dist = iou_thr, max_missed, min_hits, max_dist
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection]) -> list[Track]:
        # жадное сопоставление: сначала самые похожие пары «трек — детекция»
        pairs = sorted(
            ((iou(t.box, d.box), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)),
            reverse=True,
        )
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.iou_thr:
                break
            if ti in used_t or di in used_d:
                continue
            t = self.tracks[ti]
            t.box, t.conf = dets[di].box, dets[di].conf
            t.hits += 1
            t.missed = 0
            used_t.add(ti)
            used_d.add(di)

        # второй проход — по расстоянию: бегущий человек при 2 кадр/с смещается больше своей ширины,
        # рамки соседних кадров не пересекаются и IoU их не связывает. Ближе max_dist ростов — тот же человек.
        dist_pairs = []
        for ti, t in enumerate(self.tracks):
            if ti in used_t:
                continue
            tc, th = center(t.box), t.box[3] - t.box[1]
            for di, d in enumerate(dets):
                if di in used_d:
                    continue
                dc, dh = center(d.box), d.box[3] - d.box[1]
                if min(th, dh) < 0.6 * max(th, dh):   # слишком разный рост — другой человек
                    continue
                dist = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5 / max(th, 1)
                if dist <= self.max_dist:
                    dist_pairs.append((dist, ti, di))
        for _, ti, di in sorted(dist_pairs):
            if ti in used_t or di in used_d:
                continue
            t = self.tracks[ti]
            t.box, t.conf = dets[di].box, dets[di].conf
            t.hits += 1
            t.missed = 0
            used_t.add(ti)
            used_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self._next_id, d.box, d.conf))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        # наружу отдаём только подтверждённые треки, видимые на текущем кадре
        return [t for t in self.tracks if t.missed == 0 and t.hits >= self.min_hits]

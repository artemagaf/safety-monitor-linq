"""Тесты логики конвейера (трекер, правила, сглаживание) — часть напарника, но проверяет и контракт модели."""
import sys
import time
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.backends import Detection  # noqa: E402
from app.events import Debouncer  # noqa: E402
from app.rules import Violation, evaluate  # noqa: E402
from app.tracker import IouTracker, Track  # noqa: E402

RULES = {"ppe_required": ["helmet"], "ppe_mode": "negative", "min_person_height": 50}
FRAME = (1000, 1000)


def item(box, role, conf=0.8):
    return Detection(box, conf, role, role)


def types(res, tid):
    return sorted(v.type for v in res[tid])


def test_no_helmet_detected_on_head():
    p = Track(1, (100, 100, 200, 400), 0.9)
    res = evaluate([p], [item((130, 100, 170, 140), "no_helmet")], [], RULES, FRAME)
    assert types(res, 1) == ["no_helmet"]


def test_helmet_ok_and_item_goes_to_nearest_person():
    a, b = Track(1, (100, 100, 200, 400), 0.9), Track(2, (180, 100, 280, 400), 0.9)
    res = evaluate([a, b], [item((130, 100, 170, 140), "helmet"), item((215, 100, 255, 140), "no_helmet")],
                   [], RULES, FRAME)
    assert types(res, 1) == [] and types(res, 2) == ["no_helmet"]


def test_absence_mode():
    p = Track(1, (100, 100, 200, 400), 0.9)
    assert types(evaluate([p], [], [], {**RULES, "ppe_mode": "absence"}, FRAME), 1) == ["no_helmet"]
    assert types(evaluate([p], [], [], RULES, FRAME), 1) == []


def test_zones_and_smoking():
    zones = [
        {"name": "Пресс", "type": "restricted", "polygon": [[0, 0.3], [0.5, 0.3], [0.5, 0.5], [0, 0.5]]},
        {"name": "Покраска", "type": "ppe", "require": ["mask"], "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    ]
    p = Track(1, (100, 100, 200, 400), 0.9)   # ноги в (150, 400) -> (0.15, 0.4): внутри обеих зон
    res = evaluate([p], [item((130, 100, 170, 140), "helmet"), item((140, 200, 150, 210), "cigarette")],
                   zones, {**RULES, "ppe_mode": "absence"}, FRAME)
    assert types(res, 1) == ["no_mask", "restricted_zone", "smoking"]


def test_tracker_keeps_id():
    t = IouTracker(min_hits=1)
    first = t.update([Detection((0, 0, 100, 200), 0.9, "person")])
    second = t.update([Detection((5, 0, 105, 200), 0.9, "person")])
    assert first[0].id == second[0].id


def test_debouncer_ignores_flicker_and_respects_cooldown():
    d = Debouncer(window=5, min_ratio=0.6, cooldown_sec=10)
    v = Violation("no_helmet")
    flicker = [d.update(i, {1: [v] if i % 3 == 0 else []}) for i in range(10)]
    assert not any(flicker)
    d = Debouncer(window=5, min_ratio=0.6, cooldown_sec=10)
    alerts = [d.update(i, {1: [v]}) for i in range(8)]
    assert [i for i, a in enumerate(alerts) if a] == [2]   # 3 кадра из 5 -> событие, дальше cooldown


# ---------- 5-секундные куски ----------

import numpy as np  # noqa: E402

from app.segments import SegmentRecorder, enforce_limit  # noqa: E402

FRAME_IMG = np.zeros((120, 160, 3), dtype=np.uint8)
FINDING = {"track_id": 1, "type": "no_helmet", "label": "Нет каски", "zones": ["Зона пресса"], "pos": [0.7, 0.9]}


def fill(rec, findings_at=()):
    """Подаёт кадры, пока кусок не наберётся. findings_at — номера кадров с нарушением."""
    i = 0
    while True:
        full = rec.add(FRAME_IMG, float(i), [FINDING] if i in findings_at else [])
        i += 1
        if full:
            return i


def test_clean_segment_is_deleted(tmp_path):
    rec = SegmentRecorder("cam1", tmp_path, fps=10, seconds=5)
    assert fill(rec) == 50                       # 5 с × 10 кадров/с
    assert rec.close() is None
    assert list(tmp_path.iterdir()) == []        # на диске ничего не осталось


def test_segment_with_violation_is_kept_with_place(tmp_path):
    rec = SegmentRecorder("cam1", tmp_path, fps=10, seconds=5)
    fill(rec, findings_at=range(10, 20))
    seg = rec.close()
    assert (tmp_path / seg["clip"]).stat().st_size > 0
    assert (tmp_path / seg["snapshot"]).exists()
    [v] = seg["violations"]
    assert v["zones"] == ["Зона пресса"] and v["frames"] == 10 and v["first_seen"] == 10.0
    assert not list(tmp_path.glob("_tmp_*"))     # временный файл переименован, а не скопирован


def test_disk_limit_removes_oldest(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.mp4").write_bytes(b"0" * 600_000)
        (tmp_path / f"{name}.jpg").write_bytes(b"0")
        time.sleep(0.02)
    assert enforce_limit(tmp_path, max_total_mb=1.2) == ["a.mp4"]
    assert sorted(p.name for p in tmp_path.glob("*.mp4")) == ["b.mp4", "c.mp4"]


def test_violation_stays_active_during_alert_cooldown():
    """Уведомление одно, но нарушение остаётся подтверждённым — следующий кусок тоже сохранится."""
    d = Debouncer(window=5, min_ratio=0.6, cooldown_sec=30)
    v = Violation("no_helmet")
    alerts = [d.update(i, {1: [v]}) for i in range(10)]
    assert sum(1 for a in alerts if a) == 1
    assert (1, "no_helmet") in d.active


def test_short_tail_on_segment_border_is_not_kept(tmp_path):
    rec = SegmentRecorder("cam1", tmp_path, fps=10, seconds=5, min_violation_sec=1.0)
    fill(rec, findings_at=range(0, 3))           # нарушение видно 0.3 с — это хвост из прошлого куска
    assert rec.close() is None and list(tmp_path.iterdir()) == []


# ---------- станочный цех ----------

LATHE = [{"name": "Токарный 16К20", "type": "machine", "require": ["glasses"], "forbid": ["gloves"],
          "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
MACHINE_RULES = {"ppe_required": [], "ppe_mode": "negative", "min_person_height": 50}


def test_gloves_forbidden_at_lathe():
    p = Track(1, (100, 100, 200, 400), 0.9)
    glove_on_outstretched_hand = item((235, 250, 255, 270), "gloves")   # рука вытянута за бокс человека
    res = evaluate([p], [glove_on_outstretched_hand, item((135, 115, 165, 130), "glasses")],
                   LATHE, MACHINE_RULES, FRAME)
    assert res[1] == [Violation("forbidden_gloves", "Токарный 16К20")]


def test_bare_hands_and_no_glasses_at_lathe():
    p = Track(1, (100, 100, 200, 400), 0.9)
    res = evaluate([p], [item((235, 250, 255, 270), "bare_hand"), item((135, 115, 165, 130), "no_glasses")],
                   LATHE, MACHINE_RULES, FRAME)
    assert types(res, 1) == ["no_glasses"]   # голые руки на токарном — это правильно


# ---------- тревоги: одна на нарушение, статусы оператора, журнал ----------

def test_one_alert_while_violation_lasts_and_new_after_break():
    d = Debouncer(window=3, min_ratio=0.6, cooldown_sec=10)
    v = Violation("forbidden_gloves", "Токарный")
    alerts = [bool(d.update(t, {1: [v]})) for t in range(60)]          # минута непрерывного нарушения
    assert alerts.count(True) == 1                                        # одна тревога, а не по одной на каждые 30 с
    for t in range(60, 80):
        d.update(t, {1: []})                                              # 20 с перерыва (> cooldown)
    again = [bool(d.update(t, {1: [v]})) for t in range(80, 90)]
    assert again.count(True) == 1                                         # нарушение началось заново — новая тревога


def test_alert_journal_status_filters_and_clip(tmp_path):
    from app.events import IncidentStore
    s = IncidentStore(str(tmp_path / "db.sqlite"))
    f = {"track_id": 7, "type": "forbidden_gloves", "label": "Перчатки", "zones": ["Токарный"], "pos": [0.3, 0.9]}
    a = s.add_alert("lathe1", "Мехцех", f, 1_700_000_000.0, "alerts/a.jpg", "alerts/a_raw.jpg", critical=True)
    b = s.add_alert("cnc1", "ЧПУ", {**f, "track_id": 2, "type": "no_glasses"}, 1_700_000_100.0, "b.jpg", "b_raw.jpg", False)
    s.attach_clip([a["id"]], "lathe1_clip.mp4")
    s.set_status(b["id"], "false")
    assert [r["id"] for r in s.alerts()] == [b["id"], a["id"]]           # новые сверху
    assert s.alerts(camera="lathe1")[0]["clip"] == "lathe1_clip.mp4"
    assert [r["id"] for r in s.alerts(status="false")] == [b["id"]]
    assert s.alerts(type_="forbidden_gloves")[0]["zones"] == ["Токарный"]
    assert s.alerts(after_id=a["id"])[0]["id"] == b["id"]


def test_person_cut_by_frame_bottom_is_inside_machine_zone():
    """Ноги срезаны кадром: бокс человека доходит до нижнего края — он всё равно в зоне станка."""
    p = Track(1, (20, 20, 290, 1000), 0.9)
    res = evaluate([p], [item((230, 170, 270, 220), "gloves")], LATHE, MACHINE_RULES, FRAME)
    assert types(res, 1) == ["forbidden_gloves"]


# ---------- поведение: бег, падение, телефон, техника ----------

from app.behavior import BehaviorAnalyzer  # noqa: E402


def run_tracks(beh, boxes, dt=0.5):
    """Подаёт один и тот же трек с разными рамками, возвращает нарушения по кадрам."""
    return [[v.type for v in beh.update(i * dt, [Track(1, b, 0.9)])[1]] for i, b in enumerate(boxes)]


def test_running_vs_walking():
    beh = BehaviorAnalyzer({"run_speed": 1.5})
    walk = [(100 + 50 * i, 100, 150 + 50 * i, 300) for i in range(8)]     # 100 px/с при росте 200 = 0.5 роста/с
    assert not any("running" in v for v in run_tracks(beh, walk))
    beh = BehaviorAnalyzer({"run_speed": 1.5})
    run = [(100 + 200 * i, 100, 150 + 200 * i, 300) for i in range(8)]    # 400 px/с = 2 роста/с
    assert "running" in run_tracks(beh, run)[-1]


class _List(list):
    def tolist(self):
        return list(self)


class FakePose:
    """Имитация YOLO11-pose: один человек, рамка box; плечи и бёдра на одной высоте (лежит)."""

    def __init__(self, box, hips_conf):
        kp = [[0, 0]] * 17
        kp[5] = kp[6] = [box[0] + 50, box[1] + 40]       # плечи
        kp[11] = kp[12] = [box[2] - 70, box[1] + 50]     # бёдра — почти на той же высоте
        conf = [0.9] * 17
        conf[11] = conf[12] = hips_conf
        boxes = _List([list(box)])
        boxes.xyxy = _List([list(box)])
        self.result = SimpleNamespace(boxes=boxes,
                                            keypoints=SimpleNamespace(xy=_List([kp]), conf=_List([conf])))

    def predict(self, frame, **kw):
        return [self.result]


def fall_flags(beh, box, frame=None, n=10):
    return ["fall" in [v.type for v in beh.update(i * 0.5, [Track(1, box, 0.9)], frame)[1]] for i in range(n)]


def test_fall_needs_lying_pose_and_duration():
    lying = (100, 300, 400, 400)                                           # рамка втрое шире высоты
    flags = fall_flags(BehaviorAnalyzer({"fall_sec": 3}), lying)           # без модели позы — по рамке
    assert not flags[4] and flags[7]                                       # 2 с — ещё нет, 3.5 с — тревога

    frame = np.zeros((480, 640, 3), np.uint8)
    seated = BehaviorAnalyzer({"fall_sec": 3}, FakePose(lying, hips_conf=0.05))   # бёдер не видно: сидит
    assert not any(fall_flags(seated, lying, frame))
    down = BehaviorAnalyzer({"fall_sec": 3}, FakePose(lying, hips_conf=0.9))      # туловище горизонтально
    assert fall_flags(down, lying, frame)[7]


def test_phone_forbidden_and_vehicle_proximity():
    p = Track(1, (100, 100, 200, 400), 0.9)
    phone = item((140, 150, 160, 180), "phone")
    assert types(evaluate([p], [phone], [], {"forbid": ["phone"]}, FRAME), 1) == ["phone_use"]
    assert types(evaluate([p], [phone], [], {}, FRAME), 1) == []           # не запрещён — не нарушение
    near = Detection((230, 150, 600, 420), 0.9, "truck", "vehicle")        # зазор 30 px < 0.3 × 300
    far = Detection((500, 150, 900, 420), 0.9, "truck", "vehicle")
    assert types(evaluate([p], [near], [], {}, FRAME), 1) == ["vehicle_proximity"]
    assert types(evaluate([p], [far], [], {}, FRAME), 1) == []


def test_tracker_follows_fast_runner():
    """При 2 кадр/с бегущий смещается больше своей ширины — трекер держит его по расстоянию."""
    t = IouTracker(min_hits=1)
    ids = {t.update([Detection((100 + 120 * i, 100, 160 + 120 * i, 300), 0.9, "person")])[0].id for i in range(5)}
    assert len(ids) == 1


def test_alert_lookup_returns_that_alert_not_newest(tmp_path):
    """Клип открывается для того нарушения, по которому щёлкнули, а не для самого нового."""
    from app.events import IncidentStore
    s = IncidentStore(str(tmp_path / "db.sqlite"))
    f = {"track_id": 1, "type": "fall", "label": "Падение", "zones": [], "pos": [0.5, 0.9]}
    a = s.add_alert("demo", "Демо", f, 1.0, "a.jpg", "a_raw.jpg", True)
    s.add_alert("demo", "Демо", {**f, "type": "running"}, 2.0, "b.jpg", "b_raw.jpg", False)   # новее, без клипа
    s.attach_clip([a["id"]], "a.mp4")
    assert s.alert(a["id"])["clip"] == "a.mp4"
    assert s.alert(999) is None


# ---------- огонь, ограждение, высота, перенос груза ----------

def test_open_fire_climbing_height_and_carry():
    p = Track(1, (100, 100, 200, 400), 0.9)      # центр (150, 250) из 1000×1000
    fire = Detection((600, 600, 650, 650), 0.8, "fire", "fire")
    res = evaluate([p], [fire], [], {}, FRAME)
    assert types(res, 0) == ["open_fire"] and types(res, 1) == []          # огонь — тревога места, не человека
    assert types(evaluate([], [fire], [], {}, FRAME), 0) == ["open_fire"]   # и без людей в кадре
    assert evaluate([p], [], [], {}, FRAME)[0] == []                        # огня нет — ключ места есть, пустой

    barrier = [{"name": "Ограждение", "type": "barrier", "polygon": [[0.1, 0], [0.2, 0], [0.2, 1], [0.1, 1]]}]
    assert types(evaluate([p], [], barrier, {}, FRAME), 1) == ["climbing"]
    away = [{"name": "Ограждение", "type": "barrier", "polygon": [[0.5, 0], [0.6, 0], [0.6, 1], [0.5, 1]]}]
    assert types(evaluate([p], [], away, {}, FRAME), 1) == []

    height = [{"name": "Леса", "type": "height", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    assert types(evaluate([p], [], height, {}, FRAME), 1) == ["no_harness"]
    harness = item((130, 200, 170, 260), "harness")
    assert types(evaluate([p], [harness], height, {}, FRAME), 1) == []

    box_at_face = item((110, 90, 190, 170), "box")      # коробку несут перед лицом
    box_low = item((110, 250, 190, 330), "box")         # на уровне пояса — нормально
    assert types(evaluate([p], [box_at_face], [], {}, FRAME), 1) == ["unsafe_carry"]
    assert types(evaluate([p], [box_low], [], {}, FRAME), 1) == []


def test_playlist_camera_lists_only_matching_videos(tmp_path):
    """Камера-плейлист: files: "*.mp4" берёт видео из папки и не трогает картинки рядом."""
    from app.pipeline import CameraWorker
    for n in ("b.mp4", "a.mp4", "photo.jpg"):
        (tmp_path / n).write_bytes(b"0")
    w = CameraWorker.__new__(CameraWorker)
    w.cam = {"source": str(tmp_path), "files": "*.mp4"}
    assert [p.name for p in w._videos()] == ["a.mp4", "b.mp4"]
    w.cam = {"source": str(tmp_path)}                  # без files — не плейлист
    assert w._videos() == []


# ---------- подъём груза: спиной (нарушение) или ногами (правильно) ----------

class LiftPose(FakePose):
    """Скелет стоящего человека: наклон туловища trunk°, колени согнуты на knee°."""

    def __init__(self, box, trunk, knee):
        import math
        x, y = (box[0] + box[2]) / 2, box[1] + 0.55 * (box[3] - box[1])           # бёдра
        L = 0.3 * (box[3] - box[1])
        sh = [x + L * math.sin(math.radians(trunk)), y - L * math.cos(math.radians(trunk))]
        knee_pt = [x, y + L]
        a = math.radians(knee)                                                   # голень отклоняется на knee°
        ank = [x - L * math.sin(a), y + L + L * math.cos(a)]
        kp = [[0, 0]] * 17
        kp[5] = kp[6] = sh
        kp[11] = kp[12] = [x, y]
        kp[13] = kp[14] = knee_pt
        kp[15] = kp[16] = ank
        boxes = _List([list(box)])
        boxes.xyxy = _List([list(box)])
        self.result = SimpleNamespace(boxes=boxes, keypoints=SimpleNamespace(xy=_List([kp]), conf=_List([[0.9] * 17])))


def lift_flags(beh, box, items=None, n=6):
    frame = np.zeros((480, 640, 3), np.uint8)
    return ["bent_lift" in [v.type for v in beh.update(i * 0.5, [Track(1, box, 0.9)], frame, items)[1]]
            for i in range(n)]


def test_bent_lift_vs_squat():
    box = (200, 50, 300, 450)
    bent = lift_flags(BehaviorAnalyzer({"lift_sec": 1.0}, LiftPose(box, trunk=60, knee=10)), box)
    assert not bent[1] and bent[3]                                       # 0.5 с — ещё нет, 1.5 с — тревога
    squat = BehaviorAnalyzer({}, LiftPose(box, trunk=15, knee=90))       # присел, спина прямая — правильно
    assert not any(lift_flags(squat, box))
    upright = BehaviorAnalyzer({}, LiftPose(box, trunk=5, knee=0))       # просто стоит
    assert not any(lift_flags(upright, box))
    knees_bent = BehaviorAnalyzer({}, LiftPose(box, trunk=50, knee=80))  # наклон, но поднимает ногами
    assert not any(lift_flags(knees_bent, box))


def test_bent_lift_needs_load_when_configured():
    box = (200, 50, 300, 450)
    cfg = {"lift_needs_load": True}
    assert not any(lift_flags(BehaviorAnalyzer(cfg, LiftPose(box, 60, 10)), box))
    load = item((220, 400, 300, 460), "box")
    assert lift_flags(BehaviorAnalyzer(cfg, LiftPose(box, 60, 10)), box, [load])[3]


def test_camera_enabled_state_saved_and_loaded(tmp_path):
    """Включение и выключение камер в приложении переживает перезапуск: data/cameras.json важнее config.yaml."""
    import yaml
    from app.config import active_cameras, load_config, save_camera_states
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"model": {"path": "m.pt"}, "cameras": [
        {"id": "a", "source": 0}, {"id": "b", "source": 1, "enabled": False}]}), encoding="utf-8")
    cfg = load_config(cfg_path)
    assert [c["id"] for c in active_cameras(cfg)] == ["a"] and len(cfg["cameras"]) == 2   # выключенная — в списке
    cfg["cameras"][0]["enabled"], cfg["cameras"][1]["enabled"] = False, True
    save_camera_states(cfg)
    assert [c["id"] for c in active_cameras(load_config(cfg_path))] == ["b"]

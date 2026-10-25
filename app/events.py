"""От «нарушения на кадре» к «инциденту»: сглаживание во времени, хранение и уведомления."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

from .rules import Violation

log = logging.getLogger(__name__)


class Debouncer:
    """Создаёт событие, только если нарушение держится устойчиво.

    Для каждой пары (человек, тип нарушения) храним последние window кадров (было/не было).
    Нарушение «подтверждено», когда доля «было» >= min_ratio — такие попадают в self.active
    (по ним решаем, сохранять ли 5-секундный кусок).

    Тревога (alert) — ровно одна на нарушение: пока оно длится, повторов нет («одно нарушение —
    одно оповещение» из требований кейса). Новая тревога по той же паре — только если нарушение
    прекращалось хотя бы на cooldown_sec и началось снова.
    """

    def __init__(self, window: int, min_ratio: float, cooldown_sec: float, forget_sec: float = 60):
        self.window, self.min_ratio, self.cooldown = window, min_ratio, cooldown_sec
        self.forget_sec = forget_sec
        self.hist: dict[tuple[int, str], deque] = {}
        self.last_active: dict[tuple[int, str], float] = {}   # когда пара последний раз была подтверждена
        self.last_seen: dict[int, float] = {}
        self.active: dict[tuple[int, str], Violation] = {}   # подтверждённые на текущем кадре

    def update(self, now: float, violations: dict[int, list[Violation]]) -> list[tuple[int, Violation]]:
        current = {(tid, v.type): v for tid, vs in violations.items() for v in vs}
        for tid in violations:
            self.last_seen[tid] = now

        alerts = []
        self.active = {}
        for key in set(self.hist) | set(current):
            if key[0] not in violations:   # человека нет на кадре — историю не трогаем
                continue
            dq = self.hist.setdefault(key, deque(maxlen=self.window))
            dq.append(key in current)
            if key in current and sum(dq) >= self.min_ratio * self.window:
                self.active[key] = current[key]
                if now - self.last_active.get(key, float("-inf")) >= self.cooldown:
                    alerts.append((key[0], current[key]))
                self.last_active[key] = now

        self._forget(now)
        return alerts

    def _forget(self, now: float):
        gone = {tid for tid, t in self.last_seen.items() if now - t > self.forget_sec}
        if not gone:
            return
        for d in (self.hist, self.last_active):
            for key in [k for k in d if k[0] in gone]:
                del d[key]
        for tid in gone:
            del self.last_seen[tid]


class IncidentStore:
    """Инциденты в SQLite: один инцидент = один сохранённый 5-секундный клип с нарушениями.

    Для каждого хранится место работ (камера + участок + зоны, где стоял нарушитель),
    время, путь к клипу и превью, и список нарушений внутри клипа.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT, place TEXT, t_start REAL, t_end REAL,
                types TEXT, violations TEXT, clip TEXT, snapshot TEXT)""")
        self.db.commit()

    def add(self, camera: str, place: str, segment: dict) -> dict:
        vs = segment["violations"]
        row = dict(camera=camera, place=place, t_start=segment["t_start"], t_end=segment["t_end"],
                   types=",".join(sorted({v["type"] for v in vs})),
                   violations=json.dumps(vs, ensure_ascii=False),
                   clip=segment["clip"], snapshot=segment["snapshot"])
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO incidents (camera, place, t_start, t_end, types, violations, clip, snapshot) "
                "VALUES (:camera, :place, :t_start, :t_end, :types, :violations, :clip, :snapshot)", row)
            self.db.commit()
        return {"id": cur.lastrowid, **row, "violations": vs}

    def list(self, after_id: int = 0, limit: int = 50, camera: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM incidents WHERE id > ?", [after_id]
        if camera:
            sql += " AND camera = ?"
            args.append(camera)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [{**dict(r), "violations": json.loads(r["violations"])} for r in self.db.execute(sql, args)]

    # ---------- тревоги: одна на нарушение, со статусом от оператора ----------

    STATUSES = {"new": "Новое", "conf": "Подтверждено", "sent": "Передано мастеру", "false": "Ложное"}

    def _ensure_alerts(self):
        if getattr(self, "_alerts_ready", False):
            return
        with self.lock:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL, camera TEXT, place TEXT, track_id INTEGER, type TEXT, label TEXT,
                    critical INTEGER, zones TEXT, pos TEXT, snapshot TEXT, raw TEXT, clip TEXT,
                    status TEXT DEFAULT 'new', status_ts REAL)""")
            self.db.commit()
        self._alerts_ready = True

    def add_alert(self, camera: str, place: str, finding: dict, ts: float, snapshot: str, raw: str,
                  critical: bool) -> dict:
        self._ensure_alerts()
        row = dict(ts=ts, camera=camera, place=place, track_id=finding["track_id"], type=finding["type"],
                   label=finding["label"], critical=int(critical), zones=json.dumps(finding["zones"], ensure_ascii=False),
                   pos=json.dumps(finding["pos"]), snapshot=snapshot, raw=raw)
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO alerts (ts, camera, place, track_id, type, label, critical, zones, pos, snapshot, raw) "
                "VALUES (:ts, :camera, :place, :track_id, :type, :label, :critical, :zones, :pos, :snapshot, :raw)", row)
            self.db.commit()
        return {"id": cur.lastrowid, **row, "status": "new"}

    def attach_clip(self, alert_ids: list[int], clip: str):
        """Клип 5-секундного куска, в котором случилась тревога, становится её видеодоказательством."""
        self._ensure_alerts()
        with self.lock:
            self.db.executemany("UPDATE alerts SET clip = ? WHERE id = ? AND clip IS NULL",
                                [(clip, i) for i in alert_ids])
            self.db.commit()

    def set_status(self, alert_id: int, status: str):
        assert status in self.STATUSES
        self._ensure_alerts()
        with self.lock:
            self.db.execute("UPDATE alerts SET status = ?, status_ts = strftime('%s','now') WHERE id = ?",
                            (status, alert_id))
            self.db.commit()

    def alerts(self, after_id: int = 0, limit: int = 500, camera: str | None = None, type_: str | None = None,
               status: str | None = None, day: str | None = None) -> list[dict]:
        """Журнал тревог с фильтрами (требование кейса: тип, камера, время). day — 'YYYY-MM-DD' (местное время)."""
        self._ensure_alerts()
        sql, args = "SELECT * FROM alerts WHERE id > ?", [after_id]
        for col, val in (("camera", camera), ("type", type_), ("status", status)):
            if val:
                sql += f" AND {col} = ?"
                args.append(val)
        if day:
            sql += " AND date(ts, 'unixepoch', 'localtime') = ?"
            args.append(day)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = [dict(r) for r in self.db.execute(sql, args)]
        for r in rows:
            r["zones"], r["pos"] = json.loads(r["zones"] or "[]"), json.loads(r["pos"] or "[]")
        return rows

    def alert(self, alert_id: int) -> dict | None:
        """Одна тревога по номеру (свежая — клип мог прикрепиться позже)."""
        self._ensure_alerts()
        with self.lock:
            r = self.db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        if r is None:
            return None
        r = dict(r)
        r["zones"], r["pos"] = json.loads(r["zones"] or "[]"), json.loads(r["pos"] or "[]")
        return r

    def remove_clips(self, clips: list[str]):
        """Клипы, удалённые по лимиту диска, помечаем в базе: запись о нарушении остаётся, видео — нет."""
        if not clips:
            return
        with self.lock:
            self.db.executemany("UPDATE incidents SET clip = NULL, snapshot = NULL WHERE clip = ?",
                                [(c,) for c in clips])
            self.db.commit()


class Notifier:
    """Отправка события во внешние системы. Работает в фоне, чтобы не тормозить обработку видео."""

    def __init__(self, cfg: dict):
        self.webhook = cfg.get("webhook_url")
        self.tg_token = os.environ.get(cfg.get("telegram_token_env") or "", "")
        self.tg_chat = cfg.get("telegram_chat_id")

    def send(self, event: dict):
        if self.webhook or (self.tg_token and self.tg_chat):
            threading.Thread(target=self._send, args=(event,), daemon=True).start()

    def _send(self, event: dict):
        try:
            if self.webhook:
                req = urllib.request.Request(self.webhook, data=json.dumps(event, ensure_ascii=False).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            if self.tg_token and self.tg_chat:
                text = (f"⚠️ {event['label']}\nМесто: {event['place']}"
                        f"{' / ' + ', '.join(event['zones']) if event['zones'] else ''}\n"
                        f"Камера: {event['camera']}, человек #{event['track_id']}")
                data = urllib.parse.urlencode({"chat_id": self.tg_chat, "text": text}).encode()
                urllib.request.urlopen(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data, timeout=5)
        except Exception as e:
            log.warning("Не удалось отправить уведомление: %s", e)

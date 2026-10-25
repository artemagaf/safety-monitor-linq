"""Автоматические отчёты по охране труда: сводка нарушений за период, текст пишет Claude.

Что уходит в Claude: только сводные цифры и список событий (время, тип, участок, зона, статус).
Кадры, клипы и любые персональные данные не отправляются — люди в журнале обезличены (#номер трека).

Ключ приложение читает само из файла (reports.credentials_file в config.yaml), при каждом отчёте.
В код, логи и архив проекта ключ не попадает. Нет ключа или нет сети — строится отчёт по шаблону,
без языковой модели, чтобы служба ОТ всё равно получила сводку.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

KEY_FIELDS = ("api_key", "apikey", "key", "anthropic_api_key", "claude_api_key", "token", "secret")
STATUS_RU = {"new": "не разобрано", "conf": "подтверждено", "sent": "передано мастеру", "false": "ложное"}


# ---------- ключ ----------

def desktop_dir() -> Path:
    """Рабочий стол пользователя. На Windows он может быть перенесён (например, в OneDrive),
    поэтому берём настоящий путь из реестра, а не просто ~/Desktop."""
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as k:
                return Path(os.path.expandvars(winreg.QueryValueEx(k, "Desktop")[0]))
        except OSError:
            pass
    return Path.home() / "Desktop"


def credentials_path(cfg: dict) -> Path:
    raw = cfg.get("reports", {}).get("credentials_file", "{desktop}/iishka_credentials.json")
    return Path(os.path.expandvars(raw.replace("{desktop}", str(desktop_dir())))).expanduser()


def load_api_key(path: Path) -> str | None:
    """Ключ из файла: JSON с полем api_key / key / token … или просто строка ключа. Значение не логируется."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8-sig").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text or None
    if isinstance(data, str):
        return data.strip() or None
    if isinstance(data, dict):
        low = {str(k).lower(): v for k, v in data.items()}
        for f in KEY_FIELDS:
            if isinstance(low.get(f), str) and low[f].strip():
                return low[f].strip()
        strings = [v for v in data.values() if isinstance(v, str) and v.strip()]
        if len(strings) == 1:   # единственное строковое поле — это и есть ключ
            return strings[0].strip()
    return None


# ---------- данные за период ----------

def collect(alerts: list[dict], since: float, until: float | None = None) -> dict:
    """Сводка по тревогам из журнала за [since, until)."""
    until = until or time.time()
    rows = [a for a in alerts if since <= a["ts"] < until]
    if rows and not since:   # «всё время» — период от первого события, а не от 1970 года
        since = min(a["ts"] for a in rows)
    by_type, by_place = Counter(a["label"] for a in rows), Counter(a["place"] for a in rows)
    by_hour = Counter(time.localtime(a["ts"]).tm_hour for a in rows)
    by_status = Counter(a.get("status", "new") for a in rows)
    zones = Counter(z for a in rows for z in a.get("zones", []))
    per_place_type: dict[str, Counter] = defaultdict(Counter)
    for a in rows:
        per_place_type[a["place"]][a["label"]] += 1
    events = [{
        "время": time.strftime("%d.%m %H:%M", time.localtime(a["ts"])), "нарушение": a["label"],
        "участок": a["place"], "зона": ", ".join(a.get("zones", [])) or "—",
        "критичное": bool(a.get("critical")), "статус": STATUS_RU.get(a.get("status", "new"), a.get("status")),
    } for a in sorted(rows, key=lambda a: a["ts"])]
    return {
        "период": f"{time.strftime('%d.%m.%Y %H:%M', time.localtime(since))} — "
                  f"{time.strftime('%d.%m.%Y %H:%M', time.localtime(until))}",
        "всего": len(rows), "критичных": sum(1 for a in rows if a.get("critical")),
        "по_типам": dict(by_type.most_common()), "по_участкам": dict(by_place.most_common()),
        "по_часам": {f"{h:02d}:00": n for h, n in sorted(by_hour.items())},
        "по_статусам": {STATUS_RU.get(k, k): v for k, v in by_status.items()},
        "по_зонам": dict(zones.most_common()),
        "участок_и_тип": {p: dict(c.most_common()) for p, c in per_place_type.items()},
        "события": events[-200:],   # последние 200 — для длинных периодов хватит сводок выше
    }


# ---------- отчёт по шаблону (без языковой модели) ----------

def template_report(stats: dict, note: str = "") -> str:
    lines = [f"# Отчёт по нарушениям охраны труда", f"Период: {stats['период']}", ""]
    if note:
        lines += [f"> {note}", ""]
    if not stats["всего"]:
        return "\n".join(lines + ["Нарушений за период не зафиксировано."])
    fp = stats["по_статусам"].get("ложное", 0)
    lines += ["## Итоги",
              f"- Всего нарушений: **{stats['всего']}**, из них критичных: **{stats['критичных']}**",
              f"- Отмечено оператором как ложные: {fp}", ""]
    for title, key in (("По типам", "по_типам"), ("По участкам", "по_участкам"), ("По часам", "по_часам")):
        lines.append(f"## {title}")
        lines += [f"- {k}: {v}" for k, v in stats[key].items()]
        lines.append("")
    top_place = next(iter(stats["по_участкам"]), None)
    top_type = next(iter(stats["по_типам"]), None)
    if top_place and top_type:
        lines += ["## На что обратить внимание",
                  f"- Больше всего нарушений на участке «{top_place}».",
                  f"- Самое частое нарушение — «{top_type}»."]
    return "\n".join(lines)


# ---------- отчёт от Claude ----------

SYSTEM_PROMPT = """Ты — помощник инженера по охране труда на производственном предприятии.
По сводке нарушений, которую зафиксировала система видеоаналитики, пиши отчёт на русском языке для
службы охраны труда и руководства. Опирайся только на данные из сводки, ничего не додумывай:
если данных для вывода мало, так и скажи. Пиши деловым языком, без воды.

Структура отчёта (Markdown):
# Отчёт по нарушениям охраны труда
Период: …
## Итоги — 3–5 пунктов с цифрами
## Критичные инциденты — каждый с временем и участком; если их нет, так и напиши
## Проблемные участки и время — где и когда нарушают чаще всего
## Качество срабатываний — доля подтверждённых и ложных тревог, что из этого следует
## Рекомендации — 3–6 конкретных мер (инструктаж, ограждения, контроль в определённые часы и т.п.),
   каждая привязана к цифре из сводки"""


class ReportError(Exception):
    """Отчёт от Claude получить не удалось — сообщение можно показать пользователю как есть."""


def claude_report(stats: dict, api_key: str, rcfg: dict) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=rcfg.get("timeout_sec", 180), max_retries=2)
    data = json.dumps(stats, ensure_ascii=False, indent=1)
    try:
        # Claude Opus 5.5 + серверный запасной режим: если модель откажется отвечать по соображениям
        # безопасности, API сам повторит запрос на запасной модели (fallbacks="default").
        resp = client.beta.messages.create(
            model=rcfg.get("model", "claude-opus-5-5"),
            max_tokens=rcfg.get("max_tokens", 16000),
            system=SYSTEM_PROMPT,
            output_config={"effort": rcfg.get("effort", "medium")},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Сводка нарушений за период (JSON):\n\n{data}"}],
        )
    except anthropic.AuthenticationError:
        raise ReportError("Ключ Claude не принят (неверный или отозван). Проверьте файл с ключом.")
    except anthropic.PermissionDeniedError:
        raise ReportError("У ключа нет доступа к модели. Проверьте права ключа в консоли Anthropic.")
    except anthropic.RateLimitError:
        raise ReportError("Превышен лимит запросов к Claude. Повторите через минуту.")
    except anthropic.BadRequestError as e:
        raise ReportError(f"Claude отклонил запрос: {e.message}")
    except anthropic.APIStatusError as e:
        raise ReportError(f"Ошибка сервиса Claude ({e.status_code}). Повторите позже.")
    except anthropic.APIConnectionError:
        raise ReportError("Нет связи с сервисом Claude. Проверьте интернет.")

    if resp.stop_reason == "refusal":
        raise ReportError("Claude отказался составлять отчёт по этим данным.")
    text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise ReportError("Claude вернул пустой ответ.")
    if resp.stop_reason == "max_tokens":
        text += "\n\n_Отчёт обрезан по длине — увеличьте reports.max_tokens в config.yaml._"
    return text


def make_report(cfg: dict, alerts: list[dict], since: float, until: float | None = None) -> tuple[str, str]:
    """Возвращает (текст отчёта в Markdown, откуда он: 'claude' | 'шаблон')."""
    stats = collect(alerts, since, until)
    rcfg = cfg.get("reports", {})
    if not stats["всего"]:
        return template_report(stats), "шаблон"
    path = credentials_path(cfg)
    key = load_api_key(path)
    if not key:
        return template_report(stats, f"Отчёт построен по шаблону: файл с ключом Claude не найден ({path.name})."), "шаблон"
    try:
        return claude_report(stats, key, rcfg), "claude"
    except ReportError as e:
        log.warning("Отчёт Claude не получен: %s", e)
        return template_report(stats, f"Отчёт построен по шаблону: {e}"), "шаблон"

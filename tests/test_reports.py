"""Отчёты: чтение ключа из файла, сводка, вызов Claude (имитация клиента — без сети и без настоящего ключа)."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import reports  # noqa: E402

T0 = time.mktime((2026, 10, 24, 9, 0, 0, 0, 0, -1))
ALERTS = [
    {"ts": T0, "label": "Перчатки у станка", "place": "Токарный", "zones": ["Рабочее место"], "critical": 1, "status": "conf"},
    {"ts": T0 + 60, "label": "Перчатки у станка", "place": "Токарный", "zones": [], "critical": 1, "status": "new"},
    {"ts": T0 + 3600, "label": "Телефон", "place": "ЧПУ", "zones": [], "critical": 0, "status": "false"},
    {"ts": T0 - 86400 * 3, "label": "Бег", "place": "Склад", "zones": [], "critical": 0, "status": "new"},   # вне периода
]


@pytest.mark.parametrize("content, expected", [
    ('{"api_key": "sk-ant-test-123"}', "sk-ant-test-123"),
    ('{"ANTHROPIC_API_KEY": " sk-ant-x "}', "sk-ant-x"),
    ('{"name": "iishka", "value": "sk-ant-only-string"}', None),   # два строковых поля, ни одно не «ключ»
    ('{"something": "sk-ant-single"}', "sk-ant-single"),           # единственная строка — это ключ
    ('sk-ant-plain-text', "sk-ant-plain-text"),
    ('"sk-ant-json-string"', "sk-ant-json-string"),
])
def test_load_api_key_formats(tmp_path, content, expected):
    p = tmp_path / "creds.json"
    p.write_text(content, encoding="utf-8")
    assert reports.load_api_key(p) == expected


def test_missing_key_file_gives_template_report(tmp_path):
    cfg = {"reports": {"credentials_file": str(tmp_path / "nope.json")}}
    text, source = reports.make_report(cfg, ALERTS, since=T0 - 3600, until=T0 + 7200)
    assert source == "шаблон" and "не найден" in text
    assert "Всего нарушений: **3**" in text and "критичных: **2**" in text


def test_collect_counts_only_period():
    s = reports.collect(ALERTS, since=T0 - 3600, until=T0 + 7200)
    assert s["всего"] == 3 and s["по_типам"] == {"Перчатки у станка": 2, "Телефон": 1}
    assert s["по_статусам"]["ложное"] == 1 and len(s["события"]) == 3


class FakeAnthropic:
    """Имитация anthropic.Anthropic: запоминает параметры запроса, отвечает заготовленным текстом."""
    last = {}

    def __init__(self, api_key, **kw):
        FakeAnthropic.last = {"api_key": api_key, **kw}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        FakeAnthropic.last["request"] = kw
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text="# Отчёт\nВсё хорошо")])


def test_claude_report_request(tmp_path, monkeypatch):
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"api_key": "sk-ant-test"}), encoding="utf-8")
    text, source = reports.make_report({"reports": {"credentials_file": str(p)}}, ALERTS, T0 - 3600, T0 + 7200)
    assert source == "claude" and text == "# Отчёт\nВсё хорошо"
    req = FakeAnthropic.last["request"]
    assert FakeAnthropic.last["api_key"] == "sk-ant-test"
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]
    assert "Перчатки у станка" in req["messages"][0]["content"]           # сводка ушла в запрос
    assert "jpg" not in req["messages"][0]["content"]                    # кадры — нет


def test_claude_refusal_falls_back_to_template(tmp_path, monkeypatch):
    import anthropic

    class Refusing(FakeAnthropic):
        def _create(self, **kw):
            return SimpleNamespace(stop_reason="refusal", content=[])
    monkeypatch.setattr(anthropic, "Anthropic", Refusing)
    p = tmp_path / "c.json"
    p.write_text('{"api_key": "sk-ant-test"}', encoding="utf-8")
    text, source = reports.make_report({"reports": {"credentials_file": str(p)}}, ALERTS, T0 - 3600, T0 + 7200)
    assert source == "шаблон" and "отказался" in text

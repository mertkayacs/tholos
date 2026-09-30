import importlib
import json
import sys
from types import ModuleType

import pytest

from tholos import cli
from tholos import workspace as w


def test_teams(db):
    for name in ("research-desk", "price-watch"):
        w.load_team(db, name)
        w.load_team(db, name)
    assert len(w.list_agents(db)) == 4 and len(w.list_tables(db)) == 2
    assert len(w.list_schedules(db)) == 3 and len(w.list_rules(db)) == 3
    assert w.get_table(db, "leads")["columns"] == [
        "title",
        "url",
        "source",
        "status",
        "score",
        "summary",
    ]
    assert w.get_table(db, "prices")["columns"] == ["product", "url", "price", "target", "checked"]
    assert "https://arxiv.org/list/cs.AI/recent" in w.get_note(db, "Sources")["body"]


def test_cli_export(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("THOLOS_HOME", str(tmp_path))
    aid = w.save_agent(db, None, "Scout", "You check.", None, ["finish"])
    rid = w.queue_run(db, aid, "Check", "message")
    cli.main(["export", "--run", str(rid)])
    result = json.loads(capsys.readouterr().out)
    assert result["id"] == rid and result["steps"] == []
    with pytest.raises(SystemExit) as exc:
        cli.main(["export", "--run", "999"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv,host,port",
    [
        ([], "127.0.0.1", 7070),
        (["--host", "0.0.0.0", "--port", "7000"], "0.0.0.0", 7000),
        (["serve", "--host", "localhost", "--port", "8080"], "localhost", 8080),
    ],
)
def test_cli_serve(argv, host, port, monkeypatch):
    module = ModuleType("tholos.web")
    module.app = object()
    monkeypatch.setitem(sys.modules, "tholos.web", module)
    calls = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    cli.main(argv)
    assert calls == [(module.app, {"host": host, "port": port})]


def test_cli_unavailable_web(monkeypatch, capsys):
    original = importlib.import_module

    def missing(name):
        if name == "tholos.web":
            raise ModuleNotFoundError("Missing", name=name)
        return original(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    cli.main([])
    assert "not installed yet" in capsys.readouterr().out


def test_cli_bench(monkeypatch):
    from tholos.bench import runner as b

    calls = []
    monkeypatch.setenv("MODEL_KEY", "secret")
    monkeypatch.setattr(b, "bench", lambda *args: calls.append(args))
    cli.main(
        [
            "bench",
            "--base-url",
            "http://local.test/v1",
            "--model",
            "small",
            "--api-key-env",
            "MODEL_KEY",
            "--json-mode",
            "object",
            "--only",
            "notes",
            "--limit",
            "2",
            "--out",
            "result.jsonl",
        ]
    )
    assert calls == [
        ("http://local.test/v1", "small", "secret", "object", None, "notes", 2, "result.jsonl")
    ]
    with pytest.raises(SystemExit):
        cli.main(["bench", "--base-url", "url", "--model", "small", "--api-key-env", "MISSING_KEY"])

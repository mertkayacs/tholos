import importlib
import json
import sys
from types import ModuleType

import pytest

from tholos import cli, web
from tholos import workspace as w


@pytest.fixture(autouse=True)
def cli_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOLOS_HOME", str(tmp_path))
    monkeypatch.setenv("THOLOS_HOST", "127.0.0.1")


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
    module.is_loopback = web.is_loopback
    monkeypatch.setitem(sys.modules, "tholos.web", module)
    calls = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    cli.main(argv)
    assert calls == [(module.app, {"host": host, "port": port})]


@pytest.mark.parametrize("module", ["tholos.web", "jinja2"])
def test_cli_import_error_is_not_hidden(monkeypatch, capsys, module):
    original = importlib.import_module

    def missing(name):
        if name == "tholos.web":
            raise ModuleNotFoundError("Missing", name=module)
        return original(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError) as error:
        cli.main([])
    assert error.value.name == module
    assert capsys.readouterr().out == ""


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


def test_cli_first_run_output(db, monkeypatch, capsys):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    cli.main([])
    assert capsys.readouterr().out == (
        "Tholos: http://127.0.0.1:7070\n"
        "No model yet. Run: ollama pull hf.co/mertkayacs/Tholos-2B:Q4_K_M, "
        "then open Settings > Detect.\n"
    )
    assert w.get_setting(db, "access_token") is None


@pytest.mark.parametrize(
    "host,url",
    [
        ("127.0.0.2", "http://127.0.0.2:7070"),
        ("localhost", "http://localhost:7070"),
        ("::1", "http://[::1]:7070"),
    ],
)
def test_cli_loopback_hides_token_and_configured_model_hint(db, monkeypatch, capsys, host, url):
    import uvicorn

    w.set_setting(db, "access_token", "saved-owner-token")
    w.save_model(
        db, None, "Local", "http://localhost/v1", "small", "stored-model-key", "schema", 0, 512
    )
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    cli.main(["--host", host])
    assert capsys.readouterr().out == f"Tholos: {url}\n"


def test_cli_environment_host_enables_auth_and_reuses_token(db, monkeypatch, capsys):
    import uvicorn
    from starlette.testclient import TestClient

    monkeypatch.setenv("THOLOS_HOST", "0.0.0.0")

    def serve(app, **kwargs):
        assert kwargs["host"] == "0.0.0.0"
        with TestClient(app) as client:
            assert not app.state.loopback
            assert client.get("/", follow_redirects=False).status_code == 303

    monkeypatch.setattr(uvicorn, "run", serve)
    cli.main([])
    token = w.get_setting(db, "access_token")
    assert token and f"Access token: {token}\n" in capsys.readouterr().out
    cli.main([])
    assert w.get_setting(db, "access_token") == token
    assert f"Access token: {token}\n" in capsys.readouterr().out


def test_cli_host_flag_updates_web_auth_host(db, monkeypatch, capsys):
    import uvicorn
    from starlette.testclient import TestClient

    def serve(app, **kwargs):
        assert kwargs["host"] == "0.0.0.0"
        with TestClient(app) as client:
            assert not app.state.loopback
            assert client.get("/", follow_redirects=False).status_code == 303

    monkeypatch.setattr(uvicorn, "run", serve)
    cli.main(["--host", "0.0.0.0"])
    assert f"Access token: {w.get_setting(db, 'access_token')}\n" in capsys.readouterr().out

import inspect
import socket
import threading
import time
import types
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest
import uvicorn

from tholos import db as db_module
from tholos import web
from tholos import workspace as w

runner_stub = types.SimpleNamespace(calls=[])


def _decide(db, approval_id, approve, always=False):
    runner_stub.calls.append(("decide", approval_id, approve, always))


def _answer(db, approval_id, text):
    runner_stub.calls.append(("answer", approval_id, text))


runner_stub.decide = _decide
runner_stub.answer = _answer


class Worker:
    instances = []

    def __init__(self, db):
        self.db = db
        self.wakes = 0
        self.started = False
        Worker.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    def wake(self):
        self.wakes += 1


worker_stub = types.SimpleNamespace(Worker=Worker)

BASE_URL = "http://127.0.0.1:7070"
ORIGIN = {"Origin": BASE_URL}


class TemplateHTML(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.elements = []
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        assert "style" not in attrs, f"Inline style on {tag}"
        assert tag != "style", "Inline stylesheet"
        assert tag != "script" or attrs.get("src"), "Inline script"
        self.elements.append((tag, attrs))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOLOS_HOME", str(tmp_path))
    monkeypatch.setenv("THOLOS_HOST", "127.0.0.1")
    return tmp_path


@pytest.fixture
def stubs(monkeypatch):
    # web.py calls runner.decide/answer and worker.Worker as module globals;
    # patch them so approval tests do not execute real tools and no real
    # worker loop runs.
    monkeypatch.setattr(web, "runner", runner_stub)
    monkeypatch.setattr(web, "worker", worker_stub)


@pytest.fixture
def client(home, stubs):
    from starlette.testclient import TestClient

    with TestClient(web.app, base_url=BASE_URL) as client:
        client.get("/")  # establishes the session cookie
        yield client


@pytest.fixture
def live(home, stubs):
    # TestClient wedges on infinite streaming responses (starlette 1.7), so the
    # SSE tests run against a real uvicorn in a thread. Every read is bounded by
    # a hard httpx timeout and stops after one event.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(web.app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(10)


@pytest.fixture
def side(home):
    # A second connection to the same file, for seeding from the test thread.
    connection = db_module.connect(str(home / "tholos.db"))
    db_module.init(connection)
    yield connection
    connection.close()


def csrf(client) -> str:
    return client.cookies["tholos_session"].split(".")[0]


def post(client, url, data=None, headers=None):
    return client.post(
        url,
        data={"csrf_token": csrf(client), **(data or {})},
        headers={"X-CSRF-Token": csrf(client), **ORIGIN, **(headers or {})},
    )


def put(client, url, data=None):
    return client.put(
        url,
        data={"csrf_token": csrf(client), **(data or {})},
        headers={"X-CSRF-Token": csrf(client), **ORIGIN},
    )


def seed(side_db):
    model_id = w.save_model(
        side_db, None, "local", "http://127.0.0.1:11434/v1", "m", None, "schema", 0.2, 512
    )
    agent_id = w.save_agent(
        side_db, None, "Scout", "You check sources.", model_id, ["table_read", "finish"], 12, False
    )
    w.save_schedule(side_db, None, agent_id, "30m", "Check the sources")
    w.create_table(side_db, "leads", ["title", "score"], "you")
    w.add_rows(side_db, "leads", [{"title": "one", "score": "3"}], "you")
    w.write_note(side_db, "Focus", "# Focus\nSmall models.", "you")
    w.add_memory(side_db, agent_id, "The owner likes short briefs.", "you")
    w.add_task(side_db, "Old task", "done details", to=None)
    w.set_task(side_db, 1, "done", "finished")
    return agent_id


def test_pages_empty_db(client):
    for path in ("/", "/agents", "/agents/new", "/tables", "/notes", "/activity", "/settings"):
        response = client.get(path)
        assert response.status_code == 200, path
        TemplateHTML(response.text)


def test_pages_seeded_db(client, side):
    agent_id = seed(side)
    run_id = w.queue_run(side, agent_id, "Scheduled: check", "schedule")
    assert run_id
    paths = [
        "/", "/agents", "/agents/new", "/agents/Scout", "/tables", "/tables/leads",
        "/notes", "/notes/Focus", "/activity", "/settings", f"/runs/{run_id}",
    ]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text[:500])
        TemplateHTML(response.text)


def test_template_routes_follow_csp(client, side):
    agent_id = seed(side)
    run_id = w.queue_run(side, agent_id, "Scheduled: check", "schedule")
    _approval(side)
    _approval(side, kind="question")
    paths = [
        "/", "/agents", "/agents/new", "/agents/Scout", f"/runs/{run_id}",
        "/tables", "/tables/leads", "/notes", "/notes/Focus", "/activity", "/settings",
        "/partials/activity", "/partials/tables/leads/grid", "/partials/notes/Focus/editor",
        *[f"/partials/board/{lane}" for lane in ("scheduled", "working", "waiting", "done")],
    ]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path
        TemplateHTML(response.text)
    # Check conditional template branches too, including test/error responses.
    for path in (Path(web.__file__).parent / "templates").rglob("*"):
        if path.suffix in {".html", ".svg"}:
            source = path.read_text()
            assert "style=" not in source, path
            assert "<style" not in source, path


def test_agent_prompts_and_primary_action(client, side):
    seed(side)
    page = TemplateHTML(client.get("/agents/Scout").text)
    prompts = [(tag, attrs) for tag, attrs in page.elements if attrs.get("name") == "prompt"]
    assert len(prompts) == 2
    assert all(tag == "textarea" and 2 <= int(attrs["rows"]) <= 4 for tag, attrs in prompts)
    primary = [attrs for _, attrs in page.elements if "primary" in attrs.get("class", "").split()]
    assert len(primary) == 1


def test_empty_table_has_grid_empty_state(client, side):
    w.create_table(side, "empty", ["title", "score"], "you")
    response = client.get("/tables/empty")
    page = TemplateHTML(response.text)
    empty_states = [attrs for tag, attrs in page.elements if tag == "p"
                    and "grid-empty" in attrs.get("class", "").split()]
    assert len(empty_states) == 1
    assert "No rows yet. Agents add rows here, or add one yourself." in response.text


def test_security_headers(client):
    response = client.get("/")
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "same-origin"


def test_telegram_settings_private_chat_help(client):
    response = client.get("/settings")
    assert response.status_code == 200
    assert "Use a private chat with your bot; groups are not supported." in response.text
    TemplateHTML(response.text)


def test_rebinding_origin_cannot_create_task(client, side):
    response = post(client, "/tasks", {"title": "blocked"}, headers={
        "Host": "attacker.test:7070", "Origin": "http://attacker.test:7070",
    })
    assert response.status_code == 403
    assert "This address is not allowed. Open http://127.0.0.1:7070 instead." in response.text
    assert not w.list_tasks(side)


@pytest.mark.parametrize("path", ["/", "/events", "/static/app.css", "/login"])
def test_loopback_blocks_attacker_host_on_every_path(client, path):
    response = client.get(path, headers={"Host": "attacker.test"}, follow_redirects=False)
    assert response.status_code == 403
    assert "This address is not allowed." in response.text
    assert "set-cookie" not in response.headers
    assert "default-src 'self'" in response.headers["content-security-policy"]


@pytest.mark.parametrize("host", [
    "127.0.0.1:7070", "localhost:7070", "[::1]:7070",
    "127.0.0.1", "LOCALHOST", "[::1]", "::1",
])
def test_loopback_allows_local_hosts(client, side, host):
    assert client.get("/", headers={"Host": host}).status_code == 200
    assert client.get("/static/app.css", headers={"Host": host}).status_code == 200
    response = post(client, "/tasks", {"title": "local task"}, headers={
        "Host": host, "Origin": f"http://{host}",
    })
    assert response.status_code == 200
    assert w.list_tasks(side)[0]["title"] == "local task"


@pytest.mark.parametrize("host", [
    "localhost.attacker.test", "127.0.0.1.attacker.test", "attacker.test@127.0.0.1",
    "localhost/attacker.test", "[::1", "[::1]:7070.attacker.test", "",
])
def test_loopback_rejects_malformed_and_lookalike_hosts(client, host):
    assert client.get("/", headers={"Host": host}).status_code == 403


def test_loopback_rejects_duplicate_host_headers(client):
    response = client.get("/", headers=[("Host", "127.0.0.1"), ("Host", "attacker.test")])
    assert response.status_code == 403


def test_csrf_missing_and_wrong(client):
    assert client.post("/tasks", data={"title": "x"}, headers=ORIGIN).status_code == 403
    wrong = client.post(
        "/tasks",
        data={"title": "x", "csrf_token": "nope"},
        headers={"X-CSRF-Token": "nope", **ORIGIN},
    )
    assert wrong.status_code == 403


def test_csrf_wrong_origin(client):
    response = post(client, "/tasks", {"title": "x"}, headers={"Origin": "http://evil.example"})
    assert response.status_code == 403
    assert "origin" in response.text.lower()


def test_new_task_calls_worker_wake(client, side):
    seed(side)
    response = post(client, "/tasks", {"to": "Scout", "title": "Look up", "details": "d"})
    assert response.status_code == 200
    assert Worker.instances[-1].wakes >= 1
    tasks = w.list_tasks(side, limit=5)
    assert tasks[0]["title"] == "Look up"


def _approval(side_db, kind="approve"):
    agent_id = side_db.execute("SELECT id FROM agents LIMIT 1").fetchone()[0]
    run_id = w.queue_run(side_db, agent_id, "trigger", "task")
    step_id = side_db.execute(
        "INSERT INTO steps(run_id,n,thought,tool,args,result,status,ms,created_at) "
        "VALUES(?,1,'t','web_fetch','{}','{}','waiting',0,?)",
        (run_id, db_module.now()),
    ).lastrowid
    return side_db.execute(
        "INSERT INTO approvals(run_id,step_id,agent_id,kind,tool,args,args_hash,preview,"
        "status,created_at) VALUES(?,?,?,?,?,?,'h','fetch a page','pending',?)",
        (run_id, step_id, agent_id, kind, "web_fetch", '{"url": "https://example.com"}',
         db_module.now()),
    ).lastrowid


def test_approve_deny_always_answer(client, side):
    seed(side)
    runner_stub.calls.clear()
    approval_id = _approval(side)
    response = post(client, f"/approvals/{approval_id}/decide", {"action": "approve"})
    assert response.status_code == 200
    assert ("decide", approval_id, True, False) in runner_stub.calls

    approval_id = _approval(side)
    post(client, f"/approvals/{approval_id}/decide", {"action": "deny"})
    assert ("decide", approval_id, False, False) in runner_stub.calls

    approval_id = _approval(side)
    post(client, f"/approvals/{approval_id}/decide", {"action": "always"})
    assert ("decide", approval_id, True, True) in runner_stub.calls

    question_id = _approval(side, kind="question")
    response = post(client, f"/approvals/{question_id}/answer", {"text": "yes, do it"})
    assert response.status_code == 200
    assert ("answer", question_id, "yes, do it") in runner_stub.calls


def read_one_event(url, timeout=5):
    lines = []
    with httpx.stream("GET", url, timeout=timeout) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            lines.append(line)
            if line.startswith("data:"):
                break
    return lines


def test_sse_replay(live, side):
    w.add_event(side, "you", "note", "1", "note changed")
    lines = read_one_event(live + "/events?since=0")
    assert any("note changed" in line for line in lines)


def test_sse_live_event(live, side):
    lines = []

    def read():
        lines.extend(read_one_event(live + "/events"))

    thread = threading.Thread(target=read)
    thread.start()
    deadline = time.monotonic() + 5
    while web.app.state.sse_active == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert web.app.state.sse_active == 1
    w.add_event(side, "you", "task", "9", "task added")
    thread.join(5)
    assert not thread.is_alive(), "SSE stream did not deliver the event in time"
    assert any("task added" in line for line in lines)


def test_sse_stops_on_disconnect(live):
    with httpx.stream("GET", live + "/events?since=0", timeout=5) as response:
        assert response.status_code == 200
        deadline = time.monotonic() + 5
        while web.app.state.sse_active == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert web.app.state.sse_active == 1
    deadline = time.monotonic() + 5
    while web.app.state.sse_active != 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert web.app.state.sse_active == 0


def test_cell_edit_and_conflict(client, side):
    seed(side)
    row = w.get_table(side, "leads")["rows"][0]
    ok = put(
        client,
        f"/tables/leads/rows/{row['id']}",
        {"column": "title", "value": "two", "expected_version": str(row["version"])},
    )
    assert ok.status_code == 200
    assert "two" in ok.text
    assert w.get_table(side, "leads")["rows"][0]["data"]["title"] == "two"

    stale = put(
        client,
        f"/tables/leads/rows/{row['id']}",
        {"column": "title", "value": "three", "expected_version": "999"},
    )
    assert stale.status_code == 200
    assert "changed while you were editing" in stale.text
    assert "two" in stale.text and "Reload" in stale.text


def test_script_escaped_in_table_and_note(client, side):
    w.create_table(side, "t", ["c"], "you")
    w.add_rows(side, "t", [{"c": "<script>alert(1)</script>"}], "you")
    w.write_note(side, "n", "Text with <script>alert(1)</script> inside.", "you")
    for path in ("/tables/t", "/notes/n"):
        body = client.get(path).text
        assert "<script>alert(1)" not in body, path
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body, path


def test_csv_export_neutralizes_formula(client, side):
    w.create_table(side, "t", ["c"], "you")
    w.add_rows(side, "t", [{"c": "=cmd|' /C calc'!A1"}], "you")
    response = client.get("/tables/t.csv")
    assert response.status_code == 200
    assert "'=cmd" in response.text
    assert "\n=cmd" not in response.text


def test_non_loopback_requires_login(home, side, stubs, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setenv("THOLOS_HOST", "0.0.0.0")
    with TestClient(web.app, base_url="http://attacker.test:7070") as client:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"
        login = client.get("/login")
        assert login.status_code == 200
        assert "Access token" in login.text
        TemplateHTML(login.text)

        token = w.get_setting(side, "access_token")
        assert token
        bad = client.post(
            "/login", data={"token": "wrong"},
            headers={"Origin": "http://attacker.test:7070"}, follow_redirects=False
        )
        assert bad.status_code == 403
        TemplateHTML(bad.text)
        good = client.post(
            "/login", data={"token": token},
            headers={"Origin": "http://attacker.test:7070"}, follow_redirects=False
        )
        assert good.status_code == 303
        assert client.get("/").status_code == 200


def test_all_endpoints_are_coroutine_functions():
    # Lead review item 6: the DB connection is only touched on the event-loop
    # thread, so every endpoint must stay an async def.
    for route in web.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            assert inspect.iscoroutinefunction(endpoint), f"{route.path} is not async"


def test_markdown_renderer():
    html_out = web.markdown("# Title\n- a\n- b\n**bold** *it* `code`\n```\n<x>\n```\n[l](https://x.y)")
    assert "<h1>Title</h1>" in html_out
    assert "<ul>" in html_out and "<li>a</li>" in html_out
    assert "<strong>bold</strong>" in html_out
    assert "<em>it</em>" in html_out and "<code>code</code>" in html_out
    assert "<pre><code>" in html_out and "&lt;x&gt;" in html_out
    assert '<a href="https://x.y" rel="noopener noreferrer">l</a>' in html_out
    assert "<script>" not in web.markdown("<script>alert(1)</script>")
    assert "http://evil" not in web.markdown("[x](javascript:alert(1))")

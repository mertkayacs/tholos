import asyncio
import json
import threading
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from tholos import fetch, runner, tools, worker
from tholos import workspace as w


def reply(tool, **args):
    return {"thought": "Next step.", "tool": tool, "args": args}


def scripted(db, replies):
    iterator = iter(replies)

    def handler(request):
        assert not db.in_transaction
        body = json.loads(request.content)
        value = next(iterator)
        if callable(value):
            value = value(body)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": w.dumps(value)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def setup(db):
    mid = w.save_model(db, None, "Local", "http://local.test/v1", "small", None, "schema", 0, 512)
    aid = w.save_agent(db, None, "Scout", "You check sources.", mid, list(tools.SPECS))
    w.save_agent(db, None, "Writer", "You write briefs.", mid, ["note_write", "finish"])
    return aid, mid


def drive(db, transport, after_step=None):
    run = worker.claim(db)
    assert run is not None
    asyncio.run(runner.run_one(db, run, transport, after_step))
    return w.get_run(db, run["id"])


def test_table_work_and_handoff(db, setup):
    aid, _ = setup
    rid = w.queue_run(db, aid, "Check", "message")
    model = scripted(
        db,
        [
            reply("table_create", table="leads", columns=["title", "score"]),
            reply("table_add", table="leads", rows=[{"title": "Small model", "score": 1}]),
            reply("table_read", table="leads", query=None, limit=None),
            reply("table_update", table="leads", row=1, values={"title": None, "score": 5}),
            reply("task_add", to="Writer", title="Write brief", details="Small model scored 5"),
            reply("finish", summary="Scored and handed off"),
        ],
    )
    run = drive(db, model)
    assert run["id"] == rid and run["status"] == "done" and len(run["steps"]) == 6
    assert run["tokens_in"] == 60 and run["tokens_out"] == 24
    assert len(run["messages"]) == 14
    assert w.get_table(db, "leads")["rows"][0]["data"]["score"] == 5
    assert w.recent_changes(db, "row")[0]["run_id"] == rid
    writer = w.get_agent(db, "Writer")
    writer_run = drive(
        db,
        scripted(
            db,
            [
                reply("note_write", title="Brief", text="Small model", mode="replace"),
                reply("finish", summary="Brief ready"),
            ],
        ),
    )
    assert writer_run["agent_id"] == writer["id"] and w.list_tasks(db)[0]["status"] == "done"


def test_ask_answer_resume(db, setup):
    rid = w.queue_run(db, setup[0], "Choose", "message")
    run = drive(db, scripted(db, [reply("ask", question="Which month?")]))
    assert run["status"] == "waiting" and len(run["messages"]) == 3
    approval = w.list_waiting(db)[0]
    assert approval["kind"] == "question" and approval["preview"] == "Which month?"
    runner.answer(db, approval["id"], "March")
    assert w.get_run(db, rid)["messages"][-1]["content"].endswith(
        '{"answer":"March"}\n</tool_response>'
    )
    with pytest.raises(ValueError):
        runner.answer(db, approval["id"], "April")

    def finish(body):
        assert '"answer":"March"' in body["messages"][-1]["content"]
        return reply("finish", summary="March selected")

    assert drive(db, scripted(db, [finish]))["status"] == "done"


@pytest.mark.parametrize("approve,always", [(True, False), (False, False), (True, True)])
def test_approval_decisions(db, setup, monkeypatch, approve, always):
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "Story"})
    rid = w.queue_run(db, setup[0], "Read web", "message")
    run = drive(db, scripted(db, [reply("web_fetch", url="https://news.test/")]))
    assert run["status"] == "waiting"
    approval = w.list_waiting(db)[0]
    assert approval["args_hash"] == runner.args_hash({"url": "https://news.test/"})
    assert "web_fetch" in approval["preview"]
    runner.decide(db, approval["id"], approve, always)
    result = w.get_run(db, rid)["steps"][0]["result"]
    assert ("text" in result) == approve
    if not approve:
        assert result["error"] == "the owner denied this"
    if always:
        assert w.list_rules(db)[0]["match"] == "news.test"
    with pytest.raises(ValueError):
        runner.decide(db, approval["id"], approve)
    assert drive(db, scripted(db, [reply("finish", summary="Done")]))["status"] == "done"


def test_approval_hash_and_new_denial(db, setup, monkeypatch):
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "Story"})
    rid = w.queue_run(db, setup[0], "Read", "message")
    drive(db, scripted(db, [reply("web_fetch", url="https://news.test/")]))
    approval = w.list_waiting(db)[0]
    db.execute(
        "UPDATE approvals SET args=? WHERE id=?", ('{"url":"https://other.test/"}', approval["id"])
    )
    with pytest.raises(ValueError, match="changed"):
        runner.decide(db, approval["id"], True)
    db.execute(
        "UPDATE approvals SET args=? WHERE id=?", (w.dumps(approval["args"]), approval["id"])
    )
    w.add_rule(db, "web_fetch", "deny", "Scout", "news.test")
    monkeypatch.setattr(fetch, "get", lambda *args: pytest.fail("Denied fetch executed"))
    runner.decide(db, approval["id"], True)
    assert w.get_run(db, rid)["steps"][0]["result"]["error"] == "denied by a rule"


def test_deny_and_conflict_paths(db, setup):
    w.create_table(db, "leads", ["title"], "you")
    row = w.add_rows(db, "leads", [{"title": "Original"}], "you")[0]
    w.queue_run(db, setup[0], "Update", "message")
    edited = False

    def interfere(db, rid, tool):
        nonlocal edited
        if tool == "table_read" and not edited:
            w.update_row(db, "leads", row, {"title": "Owner edit"}, "you")
            edited = True

    run = drive(
        db,
        scripted(
            db,
            [
                reply("table_read", table="leads", query=None, limit=None),
                reply("table_update", table="leads", row=row, values={"title": "Agent edit"}),
                reply("table_read", table="leads", query=None, limit=None),
                reply("table_update", table="leads", row=row, values={"title": "Agent edit"}),
                reply("finish", summary="Updated"),
            ],
        ),
        interfere,
    )
    assert run["status"] == "done" and run["steps"][1]["result"]["error"] == "conflict"
    w.add_rule(db, "table_add", "deny")
    w.queue_run(db, setup[0], "Denied", "message")
    run = drive(
        db,
        scripted(
            db,
            [
                reply("table_add", table="leads", rows=[{"title": "Forbidden"}]),
                reply("finish", summary="Denied"),
            ],
        ),
    )
    assert run["steps"][0]["result"]["error"] == "denied by a rule"
    assert len(w.get_table(db, "leads")["rows"]) == 1


def test_read_versions_survive_wait(db, setup):
    w.create_table(db, "leads", ["title"], "you")
    row = w.add_rows(db, "leads", [{"title": "Original"}], "you")[0]
    rid = w.queue_run(db, setup[0], "Update", "message")
    drive(
        db,
        scripted(
            db,
            [
                reply("table_read", table="leads", query=None, limit=None),
                reply("ask", question="Update it?"),
            ],
        ),
    )
    runner.answer(db, w.list_waiting(db)[0]["id"], "Yes")
    run = drive(
        db,
        scripted(
            db,
            [
                reply("table_update", table="leads", row=row, values={"title": "Updated"}),
                reply("finish", summary="Done"),
            ],
        ),
    )
    assert run["id"] == rid and "error" not in run["steps"][2]["result"]


def test_stuck_and_step_limit(db, setup):
    rid = w.queue_run(db, setup[0], "Search", "message")
    run = drive(db, scripted(db, [reply("search", query="missing")] * 3))
    assert run["error"] == "stuck" and run["id"] == rid and len(run["steps"]) == 3
    aid, mid = setup
    w.save_agent(db, aid, "Scout", "You check.", mid, list(tools.SPECS), max_steps=2)
    w.queue_run(db, aid, "Search", "message")
    run = drive(db, scripted(db, [reply("search", query="one"), reply("search", query="two")]))
    assert run["error"] == "step limit" and len(run["steps"]) == 2


def test_invalid_output_continues(db, setup):
    w.queue_run(db, setup[0], "Check", "message")
    run = drive(db, scripted(db, [{}, {}, reply("finish", summary="Done")]))
    assert run["status"] == "done" and run["steps"][0]["status"] == "invalid"
    assert run["steps"][0]["result"]["invalid_json_count"] == 2
    assert run["tokens_in"] == 30


def test_schedule_followup_pause_recovery(db, setup):
    aid, mid = setup
    sid = w.save_schedule(db, None, aid, "5m", "Check")
    due = (datetime.now(UTC) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.execute("UPDATE schedules SET next_at=? WHERE id=?", (due, sid))
    worker.enqueue(db)
    worker.enqueue(db)
    assert len(w.list_runs(db)) == 1
    assert w.list_runs(db)[0]["dedupe_key"] == f"sched:{sid}:{due[:16]}"
    assert w.list_schedules(db)[0]["next_at"] > due
    drive(
        db,
        scripted(
            db, [reply("follow_up", minutes=5, note="Check again"), reply("finish", summary="Done")]
        ),
    )
    assert worker.claim(db) is None
    follow = w.list_runs(db, status="queued")[0]
    w.save_agent(db, aid, "Scout", "You check.", mid, list(tools.SPECS), paused=True)
    db.execute("UPDATE runs SET due_at=? WHERE id=?", (now_string(), follow["id"]))
    assert worker.claim(db) is None
    w.save_agent(db, aid, "Scout", "You check.", mid, list(tools.SPECS))
    run = worker.claim(db)
    assert run["id"] == follow["id"] and run["fence"] == 1
    worker.recover(db)
    assert w.get_run(db, run["id"])["status"] == "running"
    for _ in range(4):
        db.execute("UPDATE runs SET status='running',lease_until=? WHERE id=?", (due, run["id"]))
        worker.recover(db)
    assert w.get_run(db, run["id"])["status"] == "failed"
    assert w.get_run(db, run["id"])["attempts"] == 4


def now_string():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_stale_worker_cannot_commit(db, setup):
    w.create_table(db, "leads", ["title"], "you")
    rid = w.queue_run(db, setup[0], "Write", "message")
    entered, release = threading.Event(), threading.Event()

    def delayed(body):
        entered.set()
        assert release.wait(5)
        return reply("table_add", table="leads", rows=[{"title": "Stale write"}])

    async def exercise():
        job = asyncio.create_task(runner.run_one(db, worker.claim(db), scripted(db, [delayed])))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.01)
            w.stop_run(db, rid)
        finally:
            release.set()
        await job

    asyncio.run(exercise())
    assert w.get_run(db, rid)["status"] == "stopped"
    assert not w.get_table(db, "leads")["rows"] and not w.get_run(db, rid)["steps"]


def test_worker_concurrency_and_stop(db, setup):
    w.set_setting(db, "workers", 2)
    for _ in range(2):
        w.queue_run(db, setup[0], "Check", "message")
    lock = threading.Lock()
    active = peak = 0

    def handler(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": w.dumps(reply("finish", summary="Done"))}}]},
        )

    async def exercise():
        service = worker.Worker(db, httpx.MockTransport(handler))
        service.start()
        service.start()
        try:
            for _ in range(200):
                if len(w.list_runs(db, status="done")) == 2:
                    break
                await asyncio.sleep(0.01)
            assert len(w.list_runs(db, status="done")) == 2
        finally:
            await service.stop()

    asyncio.run(exercise())
    assert peak == 2


def test_effect_and_trace_rollback_together(db, setup, monkeypatch):
    w.create_table(db, "items", ["title"], "you")
    rid = w.queue_run(db, setup[0], "Write", "message")
    add_event = w.add_event

    def fail_trace(db, actor, kind, ref, text):
        if kind == "step":
            raise RuntimeError("trace failed")
        return add_event(db, actor, kind, ref, text)

    monkeypatch.setattr(w, "add_event", fail_trace)
    run = drive(db, scripted(db, [reply("table_add", table="items", rows=[{"title": "Rollback"}])]))
    assert run["id"] == rid and run["status"] == "failed" and run["error"] == "trace failed"
    assert not w.get_table(db, "items")["rows"]
    assert not w.recent_changes(db, "row") and not run["steps"]
    assert not w.search(db, "Rollback")


def test_approved_local_write_exactly_once(db, setup):
    w.create_table(db, "items", ["title"], "you")
    w.add_rule(db, "table_add", "ask")
    rid = w.queue_run(db, setup[0], "Write", "message")
    drive(db, scripted(db, [reply("table_add", table="items", rows=[{"title": "Approved"}])]))
    assert not w.get_table(db, "items")["rows"]
    approval = w.list_waiting(db)[0]
    runner.decide(db, approval["id"], True)
    assert w.get_table(db, "items")["rows"][0]["data"]["title"] == "Approved"
    assert w.recent_changes(db, "row")[0]["run_id"] == rid
    with pytest.raises(ValueError):
        runner.decide(db, approval["id"], True)
    assert len(w.get_table(db, "items")["rows"]) == 1


def test_always_allow_next_run(db, setup, monkeypatch):
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "Story"})
    w.queue_run(db, setup[0], "Read", "message")
    drive(db, scripted(db, [reply("web_fetch", url="https://news.test/")]))
    runner.decide(db, w.list_waiting(db)[0]["id"], True, always=True)
    drive(db, scripted(db, [reply("finish", summary="Done")]))
    w.queue_run(db, setup[0], "Read again", "message")
    run = drive(
        db,
        scripted(
            db, [reply("web_fetch", url="https://news.test/"), reply("finish", summary="Done")]
        ),
    )
    assert run["status"] == "done" and not w.list_waiting(db)


def test_worker_stop_requeues_inflight(db, setup):
    rid = w.queue_run(db, setup[0], "Check", "message")
    entered, release = threading.Event(), threading.Event()

    def delayed(body):
        entered.set()
        release.wait(5)
        return reply("finish", summary="Done")

    async def exercise():
        service = worker.Worker(db, scripted(db, [delayed]))
        service.start()
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            await service.stop()
            assert w.get_run(db, rid)["status"] == "queued"
        finally:
            release.set()
            await service.stop()

    asyncio.run(exercise())
    assert w.get_run(db, rid)["fence"] == 2 and not w.get_run(db, rid)["steps"]

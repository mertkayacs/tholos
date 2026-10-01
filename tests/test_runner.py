import asyncio
import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from tholos import fetch, runner, tools, worker
from tholos import workspace as w
from tholos.db import clock as runtime_clock


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
    assert len(run["messages"]) == 13 and run["messages"][-1]["role"] == "assistant"
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

    completed = drive(db, scripted(db, [finish]))
    assert completed["status"] == "done" and len(completed["messages"]) == 5
    assert completed["messages"][-1]["role"] == "assistant"


@pytest.mark.parametrize("approve,always", [(True, False), (False, False), (True, True)])
def test_approval_decisions(db, setup, monkeypatch, approve, always):
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "Story"})
    rid = w.queue_run(db, setup[0], "Read web", "message")
    run = drive(db, scripted(db, [reply("web_fetch", url="https://news.test/")]))
    assert run["status"] == "waiting"
    approval = w.list_waiting(db)[0]
    assert approval["args_hash"] == runner.args_hash({"url": "https://news.test/"})
    assert "web_fetch" in approval["preview"]
    asyncio.run(runner.decide(db, approval["id"], approve, always))
    result = w.get_run(db, rid)["steps"][0]["result"]
    assert ("text" in result) == approve
    if not approve:
        assert result["error"] == "the owner denied this"
    if always:
        assert w.list_rules(db)[0]["match"] == "news.test"
    with pytest.raises(ValueError):
        asyncio.run(runner.decide(db, approval["id"], approve))
    assert drive(db, scripted(db, [reply("finish", summary="Done")]))["status"] == "done"


def test_decide_fetch_runs_off_the_event_loop(db, setup, monkeypatch):
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "Story"})
    w.queue_run(db, setup[0], "Read web", "message")
    drive(db, scripted(db, [reply("web_fetch", url="https://news.test/")]))
    approval = w.list_waiting(db)[0]
    started, release = threading.Event(), threading.Event()
    real_get = fetch.get

    def slow_get(url, transport=None):
        started.set()
        assert release.wait(5)
        return real_get(url, transport)

    monkeypatch.setattr(fetch, "get", slow_get)
    ticks = []

    async def main():
        decision = asyncio.create_task(runner.decide(db, approval["id"], True))
        while not started.is_set():
            await asyncio.sleep(0.01)
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks.append(True)
        release.set()
        await decision

    asyncio.run(main())
    assert len(ticks) == 3
    assert w.get_run(db, approval["run_id"])["steps"][0]["result"]["text"] == "Story"


def test_approval_hash_and_new_denial(db, setup, monkeypatch):
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "Story"})
    rid = w.queue_run(db, setup[0], "Read", "message")
    drive(db, scripted(db, [reply("web_fetch", url="https://news.test/")]))
    approval = w.list_waiting(db)[0]
    db.execute(
        "UPDATE approvals SET args=? WHERE id=?", ('{"url":"https://other.test/"}', approval["id"])
    )
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(runner.decide(db, approval["id"], True))
    db.execute(
        "UPDATE approvals SET args=? WHERE id=?", (w.dumps(approval["args"]), approval["id"])
    )
    w.add_rule(db, "web_fetch", "deny", "Scout", "news.test")
    monkeypatch.setattr(fetch, "get", lambda *args: pytest.fail("Denied fetch executed"))
    asyncio.run(runner.decide(db, approval["id"], True))
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


@pytest.mark.parametrize("repeat_third", [False, True])
def test_second_identical_call_nudge(db, setup, repeat_third):
    w.write_note(db, "Focus", "Use the sources.", "you")
    rid = w.queue_run(db, setup[0], "Read Focus", "message")
    read = reply("note_read", title="Focus")
    result = {"title": "Focus", "text": "Use the sources.", "version": 1}
    nudge = "You already have this result from your previous step. Use it or take the next step."

    def second(body):
        assert body["messages"][-1]["content"] == (
            f"<tool_response>\n{w.dumps(result)}\n</tool_response>"
        )
        return read

    def third(body):
        assert body["messages"][-1]["content"] == (
            f"<tool_response>\n{w.dumps(result | {'note': nudge})}\n</tool_response>"
        )
        return read if repeat_third else reply("finish", summary="Used the sources.")

    run = drive(db, scripted(db, [read, second, third]))
    assert run["id"] == rid and len(run["steps"]) == 3
    assert run["status"] == ("failed" if repeat_third else "done")
    assert run["error"] == ("stuck" if repeat_third else None)
    assert run["steps"][0]["result"] == result
    assert run["steps"][1]["result"] == result | {"note": nudge}
    if repeat_third:
        assert run["steps"][2]["result"] == result


def test_repeated_call_executes_with_reordered_args(db, setup):
    w.write_note(db, "Focus", "Start.", "you")
    w.queue_run(db, setup[0], "Append twice", "message")
    run = drive(
        db,
        scripted(
            db,
            [
                reply("note_write", title="Focus", text="Next.", mode="append"),
                reply("note_write", mode="append", text="Next.", title="Focus"),
                reply("finish", summary="Appended twice."),
            ],
        ),
    )
    assert run["status"] == "done"
    assert w.get_note(db, "Focus")["body"].count("Next.") == 2
    assert run["steps"][0]["result"] == {"title": "Focus", "version": 2}
    assert run["steps"][1]["result"] == {
        "title": "Focus",
        "version": 3,
        "note": (
            "You already have this result from your previous step. Use it or take the next step."
        ),
    }


def test_changed_args_and_intervening_call_reset_nudge(db, setup):
    w.queue_run(db, setup[0], "Search", "message")
    run = drive(
        db,
        scripted(
            db,
            [
                reply("search", query="one"),
                reply("search", query="two"),
                reply("search", query="one"),
                reply("search", query="one"),
                reply("finish", summary="Done"),
            ],
        ),
    )
    assert run["status"] == "done"
    assert all("note" not in step["result"] for step in run["steps"][:3])
    assert run["steps"][3]["result"]["note"] == (
        "You already have this result from your previous step. Use it or take the next step."
    )


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


@pytest.fixture
def model_clock(monkeypatch):
    clock = [datetime.now(UTC).replace(microsecond=0)]
    monkeypatch.setattr(runtime_clock, "now", lambda: clock[0])
    return clock


def test_runtime_stamps_follow_a_pinned_clock(db, setup, pin_clock):
    aid, _ = setup
    w.set_setting(db, "timezone", "UTC")
    pin_clock(datetime(2026, 6, 1, 9, tzinfo=UTC))
    w.save_schedule(db, None, aid, "5m", "Check")
    w.queue_run(db, aid, "Check", "message")
    leases = []

    def after_step(conn, run_id, tool):
        row = conn.execute("SELECT lease_until FROM runs WHERE id=?", (run_id,)).fetchone()
        leases.append(row[0])

    model = scripted(
        db, [reply("follow_up", minutes=5, note="Again"), reply("finish", summary="Done")]
    )
    run = drive(db, model, after_step)
    follow = w.list_runs(db, status="queued")[0]
    stamps = [
        w.list_schedules(db)[0]["next_at"],
        leases[0],
        run["created_at"],
        run["started_at"],
        run["ended_at"],
        follow["due_at"],
    ]
    assert all(stamp.startswith("2026-06-01T09:") for stamp in stamps), stamps
    assert run["messages"][1]["content"].startswith("Now: 2026-06-01T09:")
    due, created = (datetime.fromisoformat(follow[key]) for key in ("due_at", "created_at"))
    assert due - created == timedelta(minutes=5)


@pytest.mark.parametrize("timeout,elapsed", [(600, 121), (600, 600), (120, 120), (1, 1)])
def test_model_call_inside_timeout_keeps_lease(db, setup, model_clock, timeout, elapsed):
    aid, mid = setup
    w.save_model(db, mid, "Local", "http://local.test/v1", "small", None,
                 "schema", 0, 512, timeout)
    w.create_table(db, "items", ["title"], "you")
    rid = w.queue_run(db, aid, "Add an item", "message")

    def delayed_add(body):
        model_clock[0] += timedelta(seconds=elapsed)
        return reply("table_add", table="items", rows=[{"title": "Landed"}])

    def delayed_finish(body):
        model_clock[0] += timedelta(seconds=elapsed)
        return reply("finish", summary="Done")

    result = drive(db, scripted(db, [delayed_add, delayed_finish]))
    assert result["id"] == rid and result["status"] == "done"
    assert len(result["steps"]) == 2 and result["error"] is None
    assert w.get_table(db, "items")["rows"][0]["data"] == {"title": "Landed"}


@pytest.mark.parametrize("timeout", [120, 600])
def test_claim_lease_covers_model_timeout(db, setup, model_clock, timeout):
    aid, mid = setup
    w.save_model(db, mid, "Local", "http://local.test/v1", "small", None,
                 "schema", 0, 512, timeout)
    rid = w.queue_run(db, aid, "Check", "message")
    claimed = worker.claim(db)
    lease = datetime.fromisoformat(claimed["lease_until"])
    assert lease > model_clock[0] + timedelta(seconds=timeout)
    model_clock[0] += timedelta(seconds=timeout)
    worker.recover(db)
    assert w.get_run(db, rid)["status"] == "running"


def test_json_correction_keeps_lease_for_both_requests(db, setup, model_clock):
    rid = w.queue_run(db, setup[0], "Check", "message")

    def invalid(body):
        model_clock[0] += timedelta(seconds=80)
        return {}

    def corrected(body):
        model_clock[0] += timedelta(seconds=80)
        return reply("finish", summary="Corrected")

    result = drive(db, scripted(db, [invalid, corrected]))
    assert result["id"] == rid and result["status"] == "done"
    assert len(result["steps"]) == 1 and result["steps"][0]["status"] == "retried"
    assert result["steps"][0]["result"]["summary"] == "Corrected"
    assert result["steps"][0]["result"]["invalid_json_count"] == 1


def test_stolen_long_lease_cannot_commit(db, setup):
    aid, mid = setup
    w.save_model(db, mid, "Local", "http://local.test/v1", "small", None,
                 "schema", 0, 512, 600)
    w.create_table(db, "items", ["title"], "you")
    rid = w.queue_run(db, aid, "Write", "message")
    entered, release = threading.Event(), threading.Event()

    def delayed(body):
        entered.set()
        assert release.wait(5)
        return reply("table_add", table="items", rows=[{"title": "Stolen"}])

    async def exercise():
        original = worker.claim(db)
        job = asyncio.create_task(runner.run_one(db, original, scripted(db, [delayed])))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.01)
            expired = (datetime.now(UTC) - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            db.execute("UPDATE runs SET lease_until=? WHERE id=?", (expired, rid))
            worker.recover(db)
            replacement = worker.claim(db, run_id=rid)
            assert replacement["fence"] > original["fence"]
        finally:
            release.set()
        await job
        assert w.get_run(db, rid)["fence"] == replacement["fence"]

    asyncio.run(exercise())
    result = w.get_run(db, rid)
    assert result["status"] == "running" and not result["steps"]
    assert not w.get_table(db, "items")["rows"]


def test_worker_loop_survives_unexpected_errors(db, setup, monkeypatch, caplog):
    claims = []
    real_claim, real_recover = worker.claim, worker.recover
    recover_calls = 0

    def broken_recover(db_arg, at=None):
        nonlocal recover_calls
        recover_calls += 1
        if recover_calls == 2:  # the first call comes from start(), the second from the loop
            raise RuntimeError("database is locked")
        return real_recover(db_arg, at)

    def counting_claim(db_arg, at=None, run_id=None):
        claims.append(True)
        return real_claim(db_arg, at, run_id)

    real_sleep = asyncio.sleep

    async def quick_sleep(delay, *args):
        await real_sleep(0.01 if delay == 5 else delay)

    monkeypatch.setattr(worker, "recover", broken_recover)
    monkeypatch.setattr(worker, "claim", counting_claim)
    monkeypatch.setattr(asyncio, "sleep", quick_sleep)

    async def exercise():
        service = worker.Worker(db)
        service.start()
        deadline = asyncio.get_running_loop().time() + 5
        while len(claims) < 2 and asyncio.get_running_loop().time() < deadline:
            service.wake()
            await real_sleep(0.01)
        await service.stop()

    with caplog.at_level(logging.ERROR, logger="tholos.worker"):
        asyncio.run(exercise())
    assert len(claims) >= 2
    assert "database is locked" in caplog.text


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
    asyncio.run(runner.decide(db, approval["id"], True))
    assert w.get_table(db, "items")["rows"][0]["data"]["title"] == "Approved"
    assert w.recent_changes(db, "row")[0]["run_id"] == rid
    with pytest.raises(ValueError):
        asyncio.run(runner.decide(db, approval["id"], True))
    assert len(w.get_table(db, "items")["rows"]) == 1


def test_always_allow_next_run(db, setup, monkeypatch):
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "Story"})
    w.queue_run(db, setup[0], "Read", "message")
    drive(db, scripted(db, [reply("web_fetch", url="https://news.test/")]))
    asyncio.run(runner.decide(db, w.list_waiting(db)[0]["id"], True, always=True))
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


def test_finish_ends_with_assistant_and_retains_result(db, setup):
    tid = w.add_task(db, "Check sources", to="Scout")
    run = drive(db, scripted(db, [reply("finish", summary="Ready")]))
    assert run["status"] == "done" and run["task_id"] == tid
    assert len(run["messages"]) == 3 and run["messages"][-1]["role"] == "assistant"
    assert json.loads(run["messages"][-1]["content"])["tool"] == "finish"
    assert run["steps"][-1]["result"] == {"summary": "Ready"}
    assert w.list_tasks(db, "done")[0]["result"] == "Ready"


def test_approved_finish_has_no_tool_response(db, setup):
    w.add_rule(db, "finish", "ask")
    rid = w.queue_run(db, setup[0], "Check", "message")
    waiting = drive(db, scripted(db, [reply("finish", summary="Ready")]))
    assert waiting["status"] == "waiting" and len(waiting["messages"]) == 3
    asyncio.run(runner.decide(db, w.list_waiting(db)[0]["id"], True))
    completed = w.get_run(db, rid)
    assert completed["status"] == "done" and completed["messages"] == waiting["messages"]
    assert completed["steps"][-1]["result"] == {"summary": "Ready"}


def test_denied_finish_keeps_feedback_before_successful_finish(db, setup):
    rule_id = w.add_rule(db, "finish", "ask")
    rid = w.queue_run(db, setup[0], "Check", "message")
    drive(db, scripted(db, [reply("finish", summary="Ready")]))
    asyncio.run(runner.decide(db, w.list_waiting(db)[0]["id"], False))
    queued = w.get_run(db, rid)
    assert queued["status"] == "queued" and queued["messages"][-1]["role"] == "user"
    assert "the owner denied this" in queued["messages"][-1]["content"]
    w.delete_rule(db, rule_id)
    completed = drive(db, scripted(db, [reply("finish", summary="Revised")]))
    assert completed["status"] == "done" and len(completed["messages"]) == 5
    assert completed["messages"][-1]["role"] == "assistant"

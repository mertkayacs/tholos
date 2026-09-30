import asyncio
import json
import types

import httpx
import pytest

from tholos import telegram
from tholos import workspace as w
from tholos.db import now

runner_stub = types.SimpleNamespace(calls=[])


def _decide(db, approval_id, approve, always=False):
    runner_stub.calls.append(("decide", approval_id, approve, always))


def _answer(db, approval_id, text):
    runner_stub.calls.append(("answer", approval_id, text))


runner_stub.decide = _decide
runner_stub.answer = _answer


@pytest.fixture(autouse=True)
def stub_runner(monkeypatch):
    monkeypatch.setattr(telegram, "runner", runner_stub)
    runner_stub.calls.clear()


def make_transport(calls, updates=None):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        method = request.url.path.rsplit("/", 1)[-1]
        if method == "getUpdates":
            return httpx.Response(200, json={"ok": True, "result": updates or []})
        if method == "sendMessage":
            result = {"message_id": 1000 + len(calls)}
            return httpx.Response(200, json={"ok": True, "result": result})
        if method == "answerCallbackQuery":
            return httpx.Response(200, json={"ok": True, "result": True})
        return httpx.Response(200, json={"ok": True, "result": None})

    return httpx.MockTransport(handler)


def make_poller(db, calls):
    return telegram.Poller(db, "token", "42", transport=make_transport(calls))


def payloads(calls, method):
    return [
        json.loads(request.content)
        for request in calls
        if request.url.path.endswith(f"/{method}")
    ]


def seed_approval(db, kind="approve"):
    agent = w.get_agent(db, "Scout")
    agent_id = agent["id"] if agent else w.save_agent(
        db, None, "Scout", "You check sources.", None, ["finish"], 12, False
    )
    run_id = w.queue_run(db, agent_id, "trigger", "task")
    step_id = db.execute(
        "INSERT INTO steps(run_id,n,thought,tool,args,result,status,ms,created_at) "
        "VALUES(?,1,'t','web_fetch','{}','{}','waiting',0,?)",
        (run_id, now()),
    ).lastrowid
    return db.execute(
        "INSERT INTO approvals(run_id,step_id,agent_id,kind,tool,args,args_hash,preview,"
        "status,created_at) VALUES(?,?,?,?,?,'{}','h','fetch example.com','pending',?)",
        (run_id, step_id, agent_id, kind, "web_fetch", now()),
    ).lastrowid


def test_owner_filter(db):
    calls = []
    poller = make_poller(db, calls)

    async def main():
        for _ in range(2):
            await poller.handle_update(
                {"update_id": 1, "message": {"message_id": 1, "chat": {"id": 999},
                                             "from": {"id": 42}, "text": "hello"}}
            )

    asyncio.run(main())
    assert payloads(calls, "sendMessage") == []
    assert poller.ignored_chats == {"999"}


@pytest.mark.parametrize("chat_id,sender_id", [(42, 999), (999, 42), (42, None)])
def test_callback_requires_owner_sender_and_chat(db, chat_id, sender_id):
    calls, wakes = [], []
    poller = make_poller(db, calls)
    poller.wake = lambda: wakes.append(True)
    approval_id = seed_approval(db)
    callback = {"id": "cb", "data": poller.callback("a", approval_id),
                "message": {"chat": {"id": chat_id}}}
    if sender_id is not None:
        callback["from"] = {"id": sender_id}

    async def main():
        await poller.handle_update({"update_id": 1, "callback_query": callback})
        await poller.stop()

    asyncio.run(main())
    assert not runner_stub.calls
    assert not calls and not wakes
    assert w.list_waiting(db)[0]["id"] == approval_id


@pytest.mark.parametrize("chat_id,sender_id", [(42, 999), (999, 42), (42, None)])
@pytest.mark.parametrize("reply", [False, True])
def test_message_requires_owner_sender_and_chat(db, chat_id, sender_id, reply):
    calls, wakes = [], []
    poller = make_poller(db, calls)
    poller.wake = lambda: wakes.append(True)
    question_id = seed_approval(db, kind="question")
    poller.question_messages[5] = question_id
    message = {"message_id": 7, "chat": {"id": chat_id}, "text": "Scout: check papers"}
    if sender_id is not None:
        message["from"] = {"id": sender_id}
    if reply:
        message["reply_to_message"] = {"message_id": 5}

    async def main():
        await poller.handle_update({"update_id": 1, "message": message})
        await poller.stop()

    asyncio.run(main())
    assert not runner_stub.calls
    assert not calls and not wakes
    assert not w.list_tasks(db)
    assert w.list_waiting(db)[0]["id"] == question_id


def test_callback_hmac_check(db):
    calls = []
    poller = make_poller(db, calls)
    approval_id = seed_approval(db)
    runner_stub.calls.clear()

    async def main():
        await poller.handle_update(
            {"update_id": 1, "callback_query": {
                "id": "cb1", "data": f"a:{approval_id}:000000000000", "from": {"id": 42},
                "message": {"chat": {"id": 42}}}})

    asyncio.run(main())
    assert runner_stub.calls == []
    answers = payloads(calls, "answerCallbackQuery")
    assert answers and "not valid" in answers[0]["text"]


def test_approve_flow_calls_decide(db):
    calls = []
    poller = make_poller(db, calls)
    seed_approval(db)
    runner_stub.calls.clear()

    async def main():
        for action, approve, always in (("a", True, False), ("d", False, False), ("w", True, True)):
            fresh_id = seed_approval(db)
            await poller.handle_update(
                {"update_id": 1, "callback_query": {
                    "id": "cb", "data": poller.callback(action, fresh_id), "from": {"id": 42},
                    "message": {"chat": {"id": 42}}}})
            assert ("decide", fresh_id, approve, always) in runner_stub.calls

    asyncio.run(main())
    answers = payloads(calls, "answerCallbackQuery")
    assert {a["text"] for a in answers} == {"Approved.", "Denied.", "Always allowed."}


def test_message_creates_task_for_named_agent(db):
    calls = []
    poller = make_poller(db, calls)
    w.save_agent(db, None, "Scout", "You check sources.", None, ["finish"], 12, False)
    w.save_agent(db, None, "Analyst", "You score leads.", None, ["finish"], 12, False)

    async def main():
        await poller.handle_update(
            {"update_id": 1, "message": {"message_id": 5, "chat": {"id": 42}, "from": {"id": 42},
                                         "text": "Scout: check the new papers"}})
        await poller.handle_update(
            {"update_id": 2, "message": {"message_id": 6, "chat": {"id": 42}, "from": {"id": 42},
                                         "text": "plain note for the team"}})

    asyncio.run(main())
    scout = w.get_agent(db, "Scout")
    analyst = w.get_agent(db, "Analyst")
    tasks = w.list_tasks(db)
    assert tasks[1]["title"] == "check the new papers"
    assert tasks[1]["agent_id"] == scout["id"]
    assert tasks[1]["created_by"] == "telegram"
    assert tasks[0]["title"] == "plain note for the team"
    assert tasks[0]["agent_id"] == analyst["id"]  # first agent by name order
    confirms = payloads(calls, "sendMessage")
    assert any("Task added for Scout" in m["text"] for m in confirms)


def test_question_notify_and_reply(db):
    calls = []
    poller = make_poller(db, calls)
    approval_id = seed_approval(db, kind="question")
    runner_stub.calls.clear()

    async def main():
        await poller.notify()
        messages = payloads(calls, "sendMessage")
        assert len(messages) == 1
        assert "asks:" in messages[0]["text"]
        message_id = next(iter(poller.question_messages))
        assert poller.question_messages[message_id] == approval_id
        await poller.handle_update(
            {"update_id": 1, "message": {
                "message_id": 7, "chat": {"id": 42}, "from": {"id": 42}, "text": "yes, go ahead",
                "reply_to_message": {"message_id": message_id}}})

    asyncio.run(main())
    assert ("answer", approval_id, "yes, go ahead") in runner_stub.calls


def test_approval_notify_sends_buttons(db):
    calls = []
    poller = make_poller(db, calls)
    approval_id = seed_approval(db)

    async def main():
        await poller.notify()

    asyncio.run(main())
    messages = payloads(calls, "sendMessage")
    assert len(messages) == 1
    buttons = messages[0]["reply_markup"]["inline_keyboard"][0]
    assert [b["text"] for b in buttons] == ["Approve", "Deny", "Always allow"]
    assert buttons[0]["callback_data"] == poller.callback("a", approval_id)


def test_failed_run_and_owner_task_notify(db):
    calls = []
    poller = make_poller(db, calls)
    agent_id = w.save_agent(db, None, "Scout", "You check sources.", None, ["finish"], 12, False)
    run_id = w.queue_run(db, agent_id, "trigger", "task")
    db.execute(
        "UPDATE runs SET status='failed', error='step limit', ended_at=? WHERE id=?",
        (now(), run_id),
    )
    w.add_task(db, "look at this", to=None)

    async def main():
        await poller.notify()

    asyncio.run(main())
    texts = [m["text"] for m in payloads(calls, "sendMessage")]
    assert any(f"Run #{run_id}" in t and "failed" in t for t in texts)
    assert any("look at this" in t for t in texts)

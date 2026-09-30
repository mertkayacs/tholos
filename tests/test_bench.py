import json

import httpx
import pytest

from tholos import fetch, tools
from tholos import workspace as w
from tholos.bench import runner as b

PROFILE = {
    "base_url": "http://local.test/v1",
    "model": "small",
    "api_key": None,
    "json_mode": "schema",
    "temperature": 0,
    "max_tokens": 512,
}


def transport(reference):
    steps = iter(reference)

    def handler(request):
        value = {"thought": "Next step.", **next(steps)}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": w.dumps(value)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    return httpx.MockTransport(handler)


def scenario(reference, expect):
    return {
        "id": "local-001",
        "category": "notes",
        "template": "b-local-1",
        "agent": "Scout",
        "workspace": {
            "agents": [
                {"name": "Scout", "role": "You check sources.", "tools": list(tools.SPECS)},
                {"name": "Writer", "role": "You write.", "tools": ["finish"]},
            ],
            "tables": [],
            "notes": [],
            "rules": [],
            "memories": [],
            "tasks": [],
        },
        "trigger": {"kind": "message", "text": "Check"},
        "fixtures": {},
        "respond": {},
        "reference": reference,
        "expect": expect,
        "max_steps": 14,
        "about": "A local harness regression test.",
    }


def test_every_assertion_and_isolation(monkeypatch):
    reference = [
        {"tool": "table_create", "args": {"table": "items", "columns": ["title", "score"]}},
        {"tool": "table_add", "args": {"table": "items", "rows": [{"title": "Bird", "score": 1}]}},
        {"tool": "table_read", "args": {"table": "items", "query": None, "limit": None}},
        {
            "tool": "table_update",
            "args": {"table": "items", "row": 2, "values": {"title": None, "score": 4}},
        },
        {
            "tool": "note_write",
            "args": {"title": "Brief", "text": "Bird scored", "mode": "replace"},
        },
        {
            "tool": "task_add",
            "args": {"to": "Writer", "title": "Write report", "details": "Bird scored 4"},
        },
        {"tool": "web_fetch", "args": {"url": "https://news.test/"}},
        {"tool": "ask", "args": {"question": "Use this score?"}},
        {"tool": "remember", "args": {"fact": "Prefer short reports."}},
        {"tool": "follow_up", "args": {"minutes": 10, "note": "Check again"}},
        {"tool": "finish", "args": {"summary": "Bird scored 4"}},
    ]
    expect = [
        {"type": "finished"},
        {"type": "status", "is": "done"},
        {"type": "row", "table": "items", "where": {"title": " bird "}, "has": {"score": "4"}},
        {"type": "rows", "table": "items", "count": 1, "min": 1, "max": 2},
        {"type": "table_exists", "table": "items", "columns": ["score", "title"]},
        {"type": "note", "title": "Brief", "contains": ["SCORED"], "not_contains": ["wrong"]},
        {"type": "unchanged", "table": "guard"},
        {"type": "unchanged", "note": "Guard"},
        {"type": "task", "to": "Writer", "title_contains": "report"},
        {"type": "no_task", "to": "you"},
        {"type": "called", "tool": "table_read", "args": {"table": "items"}},
        {"type": "not_called", "tool": "table_add", "args": {"table": "guard"}},
        {"type": "asked"},
        {"type": "approval", "tool": "web_fetch"},
        {"type": "memory", "agent": "Scout", "contains": "short reports"},
        {"type": "follow_up", "min_minutes": 10, "max_minutes": 10},
        {"type": "finish_contains", "any": ["scored 4", "other"]},
        {"type": "max_steps", "n": 11},
    ]
    item = scenario(reference, expect)
    item["workspace"]["tables"] = [
        {"name": "guard", "columns": ["title"], "rows": [{"title": "Keep"}]}
    ]
    item["workspace"]["notes"] = [{"title": "Guard", "body": "Keep"}]
    item["workspace"]["memories"] = [{"agent": "Scout", "text": "Initial memory"}]
    item["workspace"]["tasks"] = [{"title": "Old task", "to": "Writer", "status": "todo"}]
    item["fixtures"] = {"https://news.test/": "<title>News</title>Bird"}
    item["respond"] = {"approve": True, "answer": "Yes"}
    previous = {"https://previous.test/": "Previous"}
    monkeypatch.setattr(fetch, "FIXTURES", previous)
    for _ in range(2):
        result = b.run_scenario(item, PROFILE, transport(reference))
        assert result["passed"], result["failed_assertions"]
        assert result["status"] == "done" and result["error"] is None
        assert result["steps"] == 11 and result["tokens"] == {"in": 110, "out": 44}
        assert result["invalid_json_count"] == 0
        assert len(result["messages"]) == 23 and result["messages"][-1]["role"] == "assistant"
        assert fetch.FIXTURES is previous


@pytest.mark.parametrize(
    "value,match,expected",
    [
        ("  Bird ", "bird", True),
        ("4", 4, True),
        (4, "4.0", True),
        ("bird", {"contains": "IR"}, True),
        ("new", {"in": ["old", "NEW"]}, True),
        ("4", {"gte": 3, "lte": 5}, True),
        ("bad", {"gte": 3}, False),
        ("", {"nonempty": True}, False),
        (0, {"nonempty": True}, True),
        (None, {"nonempty": True}, False),
        (True, 1, False),
        ("nan", 1, False),
        ("Bird", "finch", False),
        (2, {"lte": 1}, False),
    ],
)
def test_matches(value, match, expected):
    assert b.matches(value, match) is expected


def test_failed_assertions_are_recorded():
    ref = [{"tool": "finish", "args": {"summary": "Nothing changed"}}]
    item = scenario(
        ref,
        [
            {"type": "note", "title": "Missing"},
            {"type": "task", "to": "you"},
            {"type": "asked"},
            {"type": "finish_contains", "any": ["different"]},
        ],
    )
    result = b.run_scenario(item, PROFILE, transport(ref))
    assert not result["passed"] and len(result["failed_assertions"]) == 4


def test_mid_run_transport_failure_is_recorded():
    read = {"tool": "note_read", "args": {"title": "Brief"}}
    item = scenario([read], [{"type": "finished"}])
    item["workspace"]["notes"] = [{"title": "Brief", "body": "Original"}]
    first = transport([read])
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) > 1:
            raise httpx.ReadTimeout("teacher timed out", request=request)
        return first.handle_request(request)

    result = b.run_scenario(item, PROFILE, httpx.MockTransport(handler))
    assert not result["passed"]
    assert result["status"] == "failed" and result["error"] == "teacher timed out"
    assert result["steps"] == 1


@pytest.mark.parametrize("error,max_steps", [("stuck", 14), ("step limit", 2)])
def test_model_behavior_failure_is_recorded(error, max_steps):
    read = {"tool": "note_read", "args": {"title": "Brief"}}
    item = scenario([read], [{"type": "finished"}])
    item["max_steps"] = max_steps
    result = b.run_scenario(item, PROFILE, transport([read] * 3))
    assert not result["passed"]
    assert result["status"] == "failed" and result["error"] == error
    assert result["steps"] == min(3, max_steps)


@pytest.mark.parametrize(
    "trigger,text",
    [
        (
            {"kind": "task", "from": "you", "title": "Check", "details": "Details"},
            "Task #1 from you: Check\nDetails",
        ),
        ({"kind": "schedule", "prompt": "Check"}, "Scheduled: Check"),
        ({"kind": "follow_up", "note": "Check"}, "Follow-up: Check"),
        ({"kind": "message", "text": "Check"}, "Message from the owner: Check"),
    ],
)
def test_trigger_forms(trigger, text):
    ref = [{"tool": "finish", "args": {"summary": "Done"}}]
    item = scenario(ref, [{"type": "finished"}, {"type": "no_task"}])
    item["trigger"] = trigger
    result = b.run_scenario(item, PROFILE, transport(ref))
    assert result["passed"] and result["messages"][1]["content"].endswith(text)


def test_interference_row_and_note():
    read = {"tool": "table_read", "args": {"table": "items", "query": None, "limit": None}}
    update = {
        "tool": "table_update",
        "args": {"table": "items", "row": 1, "values": {"title": "Updated"}},
    }
    finish = {"tool": "finish", "args": {"summary": "Updated"}}
    reference = [read, update, read, update, finish]
    item = scenario(
        reference,
        [{"type": "finished"}, {"type": "row", "table": "items", "where": {"title": "Updated"}}],
    )
    item["workspace"]["tables"] = [
        {"name": "items", "columns": ["title"], "rows": [{"title": "Original"}]}
    ]
    item["interfere"] = {
        "after_tool": "table_read",
        "table": "items",
        "row_match": {"title": "Original"},
        "set": {"title": "Owner edit"},
    }
    result = b.run_scenario(item, PROFILE, transport(reference))
    assert result["passed"] and '"error":"conflict"' in result["messages"][5]["content"]
    read = {"tool": "note_read", "args": {"title": "Brief"}}
    write = {
        "tool": "note_write",
        "args": {"title": "Brief", "text": "Original\nOwner\nUpdated", "mode": "replace"},
    }
    reference = [read, write, read, write, finish]
    item = scenario(
        reference,
        [
            {"type": "finished"},
            {"type": "note", "title": "Brief", "contains": ["Owner", "Updated"]},
        ],
    )
    item["workspace"]["notes"] = [{"title": "Brief", "body": "Original"}]
    item["interfere"] = {"after_tool": "note_read", "note": "Brief", "append": "\nOwner"}
    result = b.run_scenario(item, PROFILE, transport(reference))
    assert result["passed"] and '"error":"conflict"' in result["messages"][5]["content"]


def test_waiting_and_denied_approval():
    ref = [{"tool": "web_fetch", "args": {"url": "https://news.test/"}}]
    item = scenario(
        ref, [{"type": "status", "is": "waiting"}, {"type": "approval", "tool": "web_fetch"}]
    )
    assert b.run_scenario(item, PROFILE, transport(ref))["passed"]
    item["respond"] = {"approve": False}
    ref += [{"tool": "finish", "args": {"summary": "Owner denied the page"}}]
    item["expect"] = [{"type": "finished"}]
    result = b.run_scenario(item, PROFILE, transport(ref))
    assert result["passed"] and "the owner denied this" in result["messages"][3]["content"]


def test_bench_filter_output(tmp_path, monkeypatch, capsys):
    ref = [{"tool": "finish", "args": {"summary": "Done"}}]
    first = scenario(ref, [{"type": "finished"}])
    second = first | {"id": "local-002", "category": "table_add"}
    (tmp_path / "one.json").write_text(json.dumps(first))
    (tmp_path / "two.json").write_text(json.dumps(second))
    run = b.run_scenario
    monkeypatch.setattr(b, "run_scenario", lambda item, profile: run(item, profile, transport(ref)))
    out = tmp_path / "results.jsonl"
    results = b.bench(
        "http://local.test/v1",
        "small",
        scenarios=str(tmp_path),
        only="notes",
        limit=1,
        out=str(out),
    )
    assert len(results) == 1 and json.loads(out.read_text())["id"] == "local-001"
    assert "notes" in capsys.readouterr().out
    with pytest.raises(ValueError, match="No benchmark"):
        b.bench("url", "model", scenarios=str(tmp_path), only="missing")


@pytest.mark.parametrize(
    "created,mentions,expected",
    [
        (
            [("Writer", "New leads", "Willowfield supply at willowfield.example")],
            ["WILLOWFIELD SUPPLY", "willowfield.example"],
            True,
        ),
        (
            [
                ("Writer", "Willowfield supply", "First lead"),
                ("Writer", "Second lead", "Harbor logistics at harbor.example"),
            ],
            ["Willowfield supply", "Harbor logistics", "harbor.example"],
            True,
        ),
        (
            [("Writer", "New leads", "Willowfield supply")],
            ["Willowfield supply", "Harbor logistics"],
            False,
        ),
        ([("you", "New leads", "Willowfield supply")], ["Willowfield supply"], False),
        ([], ["Willowfield supply"], False),
        (
            [
                ("Writer", "New lead", "Willowfield supply"),
                ("you", "Other lead", "Harbor logistics"),
            ],
            ["Willowfield supply", "Harbor logistics"],
            False,
        ),
        ([], [], False),
        ([("Writer", "Unrestricted task", "Details")], [], True),
    ],
)
def test_task_mentions_use_new_tasks_for_assignee(created, mentions, expected):
    reference = [
        {"tool": "task_add", "args": {"to": to, "title": title, "details": details}}
        for to, title, details in created
    ] + [{"tool": "finish", "args": {"summary": "Handed off the requested work."}}]
    assertion = {"type": "task", "to": "Writer"}
    if mentions:
        assertion["mentions"] = mentions
    item = scenario(reference, [assertion])
    item["workspace"]["tasks"] = [
        {
            "title": "Existing Willowfield supply",
            "to": "Writer",
            "status": "todo",
            "details": "Harbor logistics at harbor.example",
        }
    ]
    result = b.run_scenario(item, PROFILE, transport(reference))
    assert result["passed"] is expected, result["failed_assertions"]


@pytest.mark.parametrize(
    "summary,facts,expected",
    [
        ("There are 3 leads with status new.", {"all": ["3"]}, True),
        ("Found 3, all ready.", {"all": ["3"]}, True),
        ("Found 30 leads.", {"all": ["3"]}, False),
        ("Count: 2033.", {"any": ["3"]}, False),
        ("Mitteco quoted 289.", {"all": ["MITTECO", "289"]}, True),
        ("Mitteco quoted 299.", {"all": ["Mitteco", "289"]}, False),
        ("Another vendor quoted 289.", {"all": ["Mitteco", "289"]}, False),
        ("Mittecompany quoted 2890.", {"any": ["Mitteco", "289"]}, False),
        ("Confirmed O-1002 and O-1004.", {"all": ["O-1002", "O-1004"]}, True),
        ("Confirmed O-10020.", {"all": ["O-1002"]}, False),
        ("Version 1.2+ is ready.", {"all": ["1.2+"]}, True),
        ("[ready] yes.", {"all": ["[ready]"]}, True),
        ("{ready} yes.", {"all": ["[ready]"]}, False),
        ("Price is +3.", {"all": ["+3"]}, True),
        ("Price is +30.", {"all": ["+3"]}, False),
        ("3 items are ready.", {"all": ["3"], "any": ["ready", "pending"]}, True),
        ("3 items are blocked.", {"all": ["3"], "any": ["ready", "pending"]}, False),
        ("30 items are ready.", {"all": ["3"], "any": ["ready", "pending"]}, False),
    ],
)
def test_finish_facts_match_whole_tokens(summary, facts, expected):
    reference = [{"tool": "finish", "args": {"summary": summary}}]
    item = scenario(reference, [{"type": "finish_contains", **facts}])
    result = b.run_scenario(item, PROFILE, transport(reference))
    assert result["passed"] is expected, result["failed_assertions"]

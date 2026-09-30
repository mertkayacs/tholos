import json

import pytest

from tholos import fetch, model, rules, tools
from tholos import workspace as w


@pytest.fixture
def context(db):
    aid = w.save_agent(db, None, "Scout", "You check sources.", None, list(tools.SPECS))
    w.save_agent(db, None, "Writer", "You write briefs.", None, ["finish"])
    rid = w.queue_run(db, aid, "Check", "message")
    return w.get_run(db, rid), w.get_agent(db, aid)


def test_every_tool(db, context, monkeypatch):
    run, agent = context

    def call(name, **args):
        result = tools.run_tool(db, run, agent, name, args)
        assert "error" not in result, result
        return result

    call("table_create", table="leads", columns=["title", "score"])
    ids = call("table_add", table="leads", rows=[{"title": "Small model", "score": 1}])["rows"]
    result = call("table_read", table="leads", query="TITLE:small score:1", limit=None)
    assert result["total"] == 1 and result["rows"][0]["row"] == ids[0]
    call("table_update", table="leads", row=ids[0], values={"score": 5})
    call("table_update", table="leads", row=ids[0], values={"score": 4})
    assert w.get_table(db, "leads")["rows"][0]["data"]["score"] == 4
    call("note_write", title="Brief", text="Small models", mode="replace")
    assert call("note_read", title="Brief")["text"] == "Small models"
    call("note_write", title="Brief", text=" and tools", mode="append")
    assert call("search", query="tools")["results"]
    task = call("task_add", to="Writer", title="Write brief", details="Top item")["task"]
    assert w.list_runs(db, w.get_agent(db, "Writer")["id"])[0]["task_id"] == task
    monkeypatch.setattr(fetch, "FIXTURES", {"https://news.test/": "<title>News</title>Story"})
    assert call("web_fetch", url="https://news.test/")["notice"].startswith("Untrusted")
    assert call("ask", question="Which source?")["question"] == "Which source?"
    call("remember", fact="Prefer short briefs.")
    assert w.list_memories(db, agent["id"])[0]["source"] == f"run:{run['id']}"
    follow = call("follow_up", minutes=5, note="Check later")["run"]
    assert w.get_run(db, follow)["trigger"] == "Follow-up: Check later"
    assert call("finish", summary="Ready")["summary"] == "Ready"


def test_tool_conflicts(db, context):
    run, agent = context
    w.create_table(db, "leads", ["title"], "you")
    row = w.add_rows(db, "leads", [{"title": "Original"}], "you")[0]
    args = {"table": "leads", "row": row, "values": {"title": "Updated"}}
    assert tools.run_tool(db, run, agent, "table_update", args)["error"] == "conflict"
    tools.run_tool(db, run, agent, "table_read", {"table": "leads", "query": None, "limit": None})
    w.update_row(db, "leads", row, {"title": "Owner edit"}, "you")
    assert (
        tools.run_tool(db, run, agent, "table_update", args)["row"]["data"]["title"] == "Owner edit"
    )
    w.write_note(db, "Brief", "Owner text", "you")
    args = {"title": "Brief", "text": "Replacement", "mode": "replace"}
    assert tools.run_tool(db, run, agent, "note_write", args)["error"] == "conflict"
    tools.run_tool(db, run, agent, "note_read", {"title": "Brief"})
    w.write_note(db, "Brief", "Changed", "you")
    assert tools.run_tool(db, run, agent, "note_write", args)["error"] == "conflict"
    assert w.get_note(db, "Brief")["body"] == "Changed"


@pytest.mark.parametrize(
    "name,args",
    [
        ("table_read", {"table": "missing", "query": None, "limit": None}),
        ("table_create", {"table": "bad", "columns": []}),
        ("table_add", {"table": "missing", "rows": [{}]}),
        ("table_update", {"table": "missing", "row": 99, "values": {}}),
        ("note_read", {"title": "missing"}),
        ("note_write", {"title": "x", "text": "x", "mode": "bad"}),
        ("search", {"query": '"'}),
        ("task_add", {"to": "missing", "title": "x", "details": ""}),
        ("web_fetch", {"url": "file:///etc/passwd"}),
        ("ask", {}),
        ("remember", {"fact": "x" * 201}),
        ("follow_up", {"minutes": 4, "note": "x"}),
        ("finish", {"summary": "x", "extra": "bad"}),
    ],
)
def test_tool_errors(db, context, name, args):
    run, agent = context
    assert "error" in tools.run_tool(db, run, agent, name, args)


def test_limits(db, context):
    run, agent = context
    w.create_table(db, "t", ["text"], "you")
    assert "error" in tools.run_tool(
        db, run, agent, "table_add", {"table": "t", "rows": [{"unknown": None}]}
    )
    assert "error" in tools.run_tool(
        db, run, agent, "table_read", {"table": "t", "query": None, "limit": 51}
    )
    assert "error" in tools.run_tool(db, run, agent, "table_add", {"table": "t", "rows": []})
    assert "error" in tools.run_tool(db, run, agent, "missing", {})
    assert "error" in tools.run_tool(
        db, run, agent | {"tools": ["finish"]}, "ask", {"question": "x"}
    )
    for value in [
        {"text": "x" * 5000},
        {"rows": [{"text": "x" * 1000}] * 10},
        {"nested": {"text": "x" * 5000}},
        {"text": "\n" * 5000},
    ]:
        result = tools.compact(value)
        assert len(w.dumps(result)) <= 2400 and result["truncated"] is True
    w.add_rows(db, "t", [{"text": "x" * 2300}] * 3, "you")
    result = tools.run_tool(
        db, run, agent, "table_read", {"table": "t", "query": None, "limit": 50}
    )
    assert len(run["_reads"]["rows"]) == len(result["rows"])


def test_rules(db):
    assert rules.check(db, "Scout", "web_fetch", "news.test") == "ask"
    assert rules.check(db, "Scout", "table_read", "leads") == "allow"
    allow = w.add_rule(db, "web_fetch", "allow", "SCOUT", "*.TEST")
    assert rules.check(db, "Scout", "web_fetch", "NEWS.test") == "allow"
    ask = w.add_rule(db, "web_fetch", "ask", "*", "news.test")
    assert rules.check(db, "Scout", "web_fetch", "news.test") == "ask"
    deny = w.add_rule(db, "web_fetch", "deny", "Scout", "news.*")
    assert rules.check(db, "Scout", "web_fetch", "news.test") == "deny"
    w.delete_rule(db, deny)
    w.delete_rule(db, ask)
    w.delete_rule(db, allow)
    assert rules.target("web_fetch", {"url": "https://NEWS.test./x"}) == "news.test"
    for name, args, expected in [
        ("table_read", {"table": "t"}, "t"),
        ("note_write", {"title": "n"}, "n"),
        ("task_add", {"to": "Writer"}, "Writer"),
        ("remember", {"fact": "x"}, "*"),
    ]:
        assert rules.target(name, args) == expected


def test_schema_portability(db):
    w.create_table(db, "t", ["title", "score"], "you")

    def visit(node):
        if isinstance(node, dict):
            assert not {"maxLength", "pattern", "format"} & node.keys()
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    assert len(tools.SPECS) == 13
    visit(tools.schemas(list(tools.SPECS), db))


@pytest.mark.parametrize("value", ["plain text", 240, 12.5, True, False, None])
def test_scalar_cells_validate_and_preserve_types(db, context, value):
    run, agent = context
    w.create_table(db, "cells", ["value"], "you")
    schemas = tools.schemas(["table_add", "table_update"], db)
    for name in schemas:
        properties = schemas[name]["anyOf"][0]["properties"]
        cells = properties["rows"]["items"] if name == "table_add" else properties["values"]
        assert cells["properties"]["value"]["type"] == ["string", "number", "boolean", "null"]
    args = {"table": "cells", "rows": [{"value": value}]}
    model.validate(
        {"thought": "Add a cell.", "tool": "table_add", "args": args}, model.schema(schemas)
    )
    result = tools.run_tool(db, run, agent, "table_add", args)
    row_id = result["rows"][0]
    stored = json.loads(db.execute("SELECT data FROM rows WHERE id=?", (row_id,)).fetchone()[0])
    expected = "" if value is None else value
    assert stored["value"] == expected and type(stored["value"]) is type(expected)
    w.update_row(db, "cells", row_id, {"value": "unchanged"}, "you")
    tools.run_tool(db, run, agent, "table_read", {"table": "cells", "query": None, "limit": None})
    args = {"table": "cells", "row": row_id, "values": {"value": value}}
    model.validate(
        {"thought": "Update a cell.", "tool": "table_update", "args": args}, model.schema(schemas)
    )
    result = tools.run_tool(db, run, agent, "table_update", args)
    expected = "unchanged" if value is None else value
    stored = result["row"]["data"]["value"]
    assert stored == expected and type(stored) is type(expected)
    assert w.recent_changes(db, "row", row_id)[0]["after"]["data"]["value"] == expected

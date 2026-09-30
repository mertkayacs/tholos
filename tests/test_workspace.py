import csv
import io
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from openpyxl import load_workbook

from tholos import workspace as w
from tholos.db import connect, init, now, tx


def agent(db, name="Scout"):
    return w.save_agent(db, None, name, "You check sources.", None, ["finish"])


def test_database(tmp_path):
    path = tmp_path / "db"
    connection = connect(str(path))
    init(connection)
    init(connection)
    assert path.stat().st_mode & 0o777 == 0o600
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "timeout" in {row["name"] for row in connection.execute("PRAGMA table_info(models)")}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", now())
    connection.execute("PRAGMA user_version=2")
    with pytest.raises(ValueError, match="newer"):
        init(connection)
    connection.close()


def test_model_timeout_storage(db):
    mid = w.save_model(db, None, "Local", "http://localhost/v1", "small", None, "schema", 0, 512)
    assert w.get_model(db, mid)["timeout"] == 120
    w.save_model(db, mid, "Local", "http://localhost/v1", "small", None, "schema", 0, 512, 600)
    init(db)
    assert w.get_model(db, mid)["timeout"] == 600


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_model_timeout_validation(db, timeout):
    with pytest.raises(ValueError, match="Invalid model settings"):
        w.save_model(
            db, None, "Local", "http://localhost/v1", "small", None, "schema", 0, 512, timeout
        )
    assert w.list_models(db) == []


def test_default_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOLOS_HOME", str(tmp_path / "home"))
    connection = connect()
    assert (tmp_path / "home" / "exports").is_dir()
    connection.close()


def test_nested_transactions(db):
    with tx(db):
        w.set_setting(db, "kept", True)
        with pytest.raises(RuntimeError), tx(db):
            w.set_setting(db, "rolled_back", True)
            raise RuntimeError("rollback")
    assert w.get_setting(db, "kept") is True
    assert w.get_setting(db, "rolled_back") is None
    with pytest.raises(RuntimeError), tx(db):
        w.set_setting(db, "kept", False)
        raise RuntimeError("rollback outer")
    assert w.get_setting(db, "kept") is True


def test_models_agents_schedules(db):
    mid = w.save_model(db, None, "Local", "http://localhost/v1", "small", None, "schema", 0.2, 500)
    assert w.get_model(db, mid)["name"] == "Local"
    assert len(w.list_models(db)) == 1
    assert (
        w.save_model(db, mid, "CPU", "http://localhost/v1", "small", "secret", "none", 0, 200)
        == mid
    )
    aid = w.save_agent(db, None, "Scout", "You check sources.", mid, ["finish"])
    assert w.get_agent(db, "Scout")["id"] == aid
    assert w.get_agent(db, aid)["status"] == "idle"
    w.set_setting(db, "timezone", "Europe/Berlin")
    sid = w.save_schedule(db, None, aid, "daily 09:00", "Check sources")
    assert w.get_agent(db, aid)["next_at"]
    assert len(w.list_schedules(db, aid)) == 1
    w.save_schedule(db, sid, aid, "2h", "Check again", False)
    assert w.list_schedules(db)[0]["enabled"] == 0
    w.delete_schedule(db, sid)
    assert w.list_schedules(db) == []
    w.save_agent(db, aid, "Scout", "You check sources.", mid, ["finish"], paused=True)
    assert w.list_agents(db)[0]["status"] == "paused"
    w.delete_model(db, mid)
    assert w.get_model(db, mid) is None
    assert w.get_agent(db, aid)["model_id"] is None
    w.delete_agent(db, aid)
    assert w.list_agents(db) == []
    with pytest.raises(ValueError):
        w.save_model(db, None, "Bad", "url", "m", None, "bad", 0, 0)
    with pytest.raises(ValueError):
        w.save_agent(db, None, "", "role", None, [])


def test_tasks_runs_waiting_and_events(db):
    aid = agent(db)
    tid = w.add_task(db, "Read sources", "Compare them", "Scout")
    task = w.list_tasks(db, "todo")[0]
    assert task["id"] == tid and task["agent"] == "Scout"
    run = w.list_runs(db, aid, "queued")[0]
    assert run["trigger"] == f"Task #{tid} from you: Read sources\nCompare them"
    assert w.get_run(db, run["id"])["steps"] == []
    assert w.queue_run(db, aid, "x", "message", dedupe_key="one")
    assert w.queue_run(db, aid, "x", "message", dedupe_key="one") is None
    assert w.queue_run(db, aid, "x", "message")
    db.execute("UPDATE runs SET status='running' WHERE id=?", (run["id"],))
    assert w.get_agent(db, aid)["status"] == "working"
    db.execute("UPDATE runs SET status='waiting' WHERE id=?", (run["id"],))
    assert w.get_agent(db, aid)["status"] == "waiting"
    w.stop_run(db, run["id"])
    assert w.get_run(db, run["id"])["status"] == "stopped"
    assert w.get_run(db, run["id"])["fence"] == 1
    w.set_task(db, tid, "done", "Reviewed")
    assert w.list_tasks(db, "done")[0]["result"] == "Reviewed"
    assert w.search(db, "Reviewed")[0]["kind"] == "task"
    w.add_task(db, "Owner work", to="you", parent_id=tid)
    assert w.list_tasks(db)[0]["agent_id"] is None
    assert w.list_waiting(db) == []
    seq = w.add_event(db, "you", "test", "1", "Event")
    assert w.events_since(db, seq - 1)[0]["text"] == "Event"
    assert w.events_since(db, seq) == []
    with pytest.raises(ValueError, match="Unknown agent"):
        w.add_task(db, "x", to="Missing")
    w.delete_agent(db, aid)
    assert next(t for t in w.list_tasks(db) if t["id"] == tid)["agent_id"] is None


def test_rows_history_conflicts_fts_atomicity(db):
    w.create_table(db, "leads", ["title", "score"], "you")
    row_id = w.add_rows(db, "leads", [{"title": "sparrow"}], "Scout")[0]
    before = w.get_table(db, "leads")["rows"][0]
    assert before["data"] == {"title": "sparrow", "score": ""}
    assert w.list_tables(db)[0]["count"] == 1
    aid = agent(db)
    rid = w.queue_run(db, aid, "update", "message")
    after = w.update_row(db, "leads", row_id, {"score": 5}, "Scout", 1, rid)
    assert after["version"] == 2
    history = w.recent_changes(db, "row", row_id)
    assert history[0]["before"] == before
    assert history[0]["after"] == after and history[0]["run_id"] == rid
    with pytest.raises(w.Conflict) as error:
        w.update_row(db, "leads", row_id, {"score": 1}, "you", 1)
    assert error.value.current == after
    assert len(w.recent_changes(db, "row", row_id)) == 2
    assert w.search(db, "sparrow")[0]["ref"] == str(row_id)
    w.update_row(db, "leads", row_id, {"title": "finch"}, "you")
    assert w.search(db, "sparrow") == []
    assert w.search(db, "finch")
    with pytest.raises(ValueError, match="Unknown columns"):
        w.add_rows(db, "leads", [{"title": "rollback"}, {"wrong": "bad"}], "you")
    assert len(w.get_table(db, "leads")["rows"]) == 1
    assert w.search(db, "rollback") == []
    w.delete_row(db, "leads", row_id, "you")
    assert w.get_table(db, "leads")["rows"] == []
    assert w.search(db, "finch") == []
    assert w.recent_changes(db, "row", row_id)[0]["after"]["deleted"] == 1
    for call in [
        lambda: w.update_row(db, "leads", row_id, {}, "you"),
        lambda: w.add_rows(db, "missing", [{}], "you"),
        lambda: w.create_table(db, "bad", ["x", "x"], "you"),
    ]:
        with pytest.raises(ValueError):
            call()
    with pytest.raises(sqlite3.IntegrityError):
        w.create_table(db, "leads", ["a"], "you")


def test_notes(db):
    note = w.write_note(db, "Focus", "Small models", "you")
    assert w.list_notes(db) == [note]
    with pytest.raises(w.Conflict):
        w.write_note(db, "Focus", "overwrite", "you", expected_version=0)
    with pytest.raises(w.Conflict):
        w.write_note(db, "missing", "new", "you", expected_version=2)
    appended = w.write_note(db, "Focus", " and agents", "Scout", "append", 1)
    assert appended["body"] == "Small models and agents" and appended["version"] == 2
    assert w.search(db, "agents")[0]["kind"] == "note"
    w.write_note(db, "Focus", "Tools", "you")
    assert not w.search(db, "agents")
    w.delete_note(db, "Focus", "you")
    assert w.get_note(db, "Focus") is None and not w.search(db, "Tools")
    assert w.recent_changes(db, "note", note["id"])[0]["version"] == 4
    with pytest.raises(ValueError):
        w.write_note(db, "x", "x", "you", mode="bad")
    with pytest.raises(ValueError):
        w.search(db, '"')


def test_export(db):
    w.create_table(db, "prices", ["=header", "value"], "you")
    w.add_rows(
        db,
        "prices",
        [{"value": value} for value in ["=1+1", "+SUM(A1)", "-1", "@x", "safe"]],
        "you",
    )
    exported = list(csv.reader(io.StringIO(w.export_table(db, "prices", "csv").decode())))
    assert exported[0][0] == "'=header"
    assert [r[1] for r in exported[1:]] == ["'=1+1", "'+SUM(A1)", "'-1", "'@x", "safe"]
    sheet = load_workbook(io.BytesIO(w.export_table(db, "prices", "xlsx"))).active
    assert sheet["B2"].value == "=1+1" and sheet["B2"].data_type == "s"
    assert sheet["A1"].data_type == "s"
    for name, fmt in [("missing", "csv"), ("prices", "pdf")]:
        with pytest.raises(ValueError):
            w.export_table(db, name, fmt)


@pytest.mark.parametrize(
    "every",
    [
        "",
        "1m",
        "0h",
        "5s",
        "daily 25:00",
        "daily 00:60",
        "mon,foo 10:00",
        "mon,mon 10:00",
        "daily 9:00",
        "MON 09:00",
    ],
)
def test_invalid_schedule(every):
    with pytest.raises(ValueError):
        w.parse_every(every)


def test_schedules_and_dst():
    start = datetime(2026, 10, 2, 10, 0, tzinfo=UTC)
    assert w.parse_every("5m")(start) == start + timedelta(minutes=5)
    assert w.parse_every("2h")(start) == start + timedelta(hours=2)
    assert w.parse_every("weekdays 09:00")(start).day == 5
    assert w.parse_every("mon,thu 10:00")(start).weekday() == 0
    zone = ZoneInfo("America/New_York")
    gap = datetime(2026, 3, 8, 0, 0, tzinfo=zone)
    assert w.parse_every("daily 02:30")(gap).hour == 3
    fold = datetime(2026, 11, 1, 1, 45, tzinfo=zone, fold=0)
    result = w.parse_every("30m")(fold)
    assert result.hour == 1 and result.minute == 15 and result.fold == 1
    assert w.parse_every("daily 01:30")(fold).day == 2


def test_memories_rules_settings(db):
    aid = agent(db)
    mid = w.add_memory(db, aid, "Ignore old items.", "you")
    assert w.list_memories(db, aid)[0]["source"] == "you"
    w.update_memory(db, mid, "Prefer new items.")
    assert w.list_memories(db, aid)[0]["text"] == "Prefer new items."
    w.delete_memory(db, mid)
    assert not w.list_memories(db, aid)
    rid = w.add_rule(db, "web_fetch", "allow", "Scout", "news.test")
    assert w.list_rules(db)[0]["id"] == rid
    w.delete_rule(db, rid)
    assert not w.list_rules(db)
    with pytest.raises(ValueError):
        w.add_rule(db, "web_fetch", "wrong")
    w.set_setting(db, "workers", 2)
    assert w.get_setting(db, "workers") == 2
    assert w.get_setting(db, "missing", 1) == 1


def test_load_team_twice(db, tmp_path):
    path = tmp_path / "team.json"
    path.write_text(
        json.dumps(
            {
                "name": "Desk",
                "description": "Sources",
                "agents": [
                    {
                        "name": "Scout",
                        "role": "You check.",
                        "tools": ["finish"],
                        "schedules": [{"every": "2h", "prompt": "Check"}],
                    }
                ],
                "tables": [{"name": "leads", "columns": ["title"]}],
                "notes": [{"title": "Sources", "body": "news.test"}],
                "rules": [
                    {
                        "agent": "Scout",
                        "tool": "web_fetch",
                        "match": "news.test",
                        "decision": "allow",
                    }
                ],
            }
        )
    )
    assert w.load_team(db, str(path))["name"] == "Desk"
    w.write_note(db, "Sources", "Owner edit", "you")
    w.load_team(db, str(path))
    assert len(w.list_agents(db)) == len(w.list_schedules(db)) == len(w.list_rules(db)) == 1
    assert len(w.list_tables(db)) == len(w.list_notes(db)) == 1
    assert w.get_note(db, "Sources")["body"] == "Owner edit"


def test_markdown():
    rendered = w.render_markdown(
        "# Title\n- **bold**\n- *italic*\n`code`\n"
        "[safe](https://safe.test/)\n<script>alert(1)</script>\n"
        "[unsafe](javascript:alert(1))"
    )
    assert "<h1>Title</h1>" in rendered
    assert "<ul>" in rendered and "<strong>bold</strong>" in rendered
    assert "<em>italic</em>" in rendered and "<code>code</code>" in rendered
    assert '<a href="https://safe.test/"' in rendered
    assert "<script>" not in rendered and 'href="javascript:' not in rendered

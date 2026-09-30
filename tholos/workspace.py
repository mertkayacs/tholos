import csv
import html
import io
import json
import os
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tholos.db import now, tx

DB = sqlite3.Connection


class Conflict(Exception):
    def __init__(self, current: dict):
        self.current = current
        super().__init__("Version conflict")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _many(db: DB, sql: str, params: tuple = (), fields: tuple = ()) -> list[dict]:
    rows = [dict(row) for row in db.execute(sql, params)]
    for row in rows:
        for field in fields:
            if row.get(field) is not None:
                row[field] = json.loads(row[field])
    return rows


def _one(db: DB, sql: str, params: tuple = (), fields: tuple = ()) -> dict | None:
    rows = _many(db, sql, params, fields)
    return rows[0] if rows else None


def add_event(db: DB, actor: str, kind: str, ref: str, text: str) -> int:
    with tx(db):
        return db.execute(
            "INSERT INTO events(at,actor,kind,ref,text) VALUES(?,?,?,?,?)",
            (now(), actor, kind, str(ref), text),
        ).lastrowid


def _change(
    db: DB,
    kind: str,
    before: dict | None,
    after: dict | None,
    actor: str,
    run_id: int | None = None,
) -> None:
    item = after or before
    version = after["version"] if after else before["version"] + 1
    db.execute(
        'INSERT INTO history(kind,ref_id,version,"before","after",actor,run_id,at) '
        "VALUES(?,?,?,?,?,?,?,?)",
        (kind, item["id"], version, dumps(before), dumps(after), actor, run_id, now()),
    )
    add_event(db, actor, kind, str(item["id"]), f"{kind} changed")


def _index(db: DB, kind: str, ref: int, title: str, body: str | None) -> None:
    db.execute("DELETE FROM search WHERE kind=? AND ref=?", (kind, str(ref)))
    if body is not None:
        db.execute(
            "INSERT INTO search(kind,ref,title,body) VALUES(?,?,?,?)", (kind, str(ref), title, body)
        )


def get_setting(db: DB, key: str, default: Any = None) -> Any:
    row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_setting(db: DB, key: str, value: Any) -> None:
    with tx(db):
        db.execute(
            "INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, dumps(value)),
        )


def timezone(db: DB) -> tzinfo:
    name = get_setting(db, "timezone") or os.environ.get("TZ")
    if name:
        return ZoneInfo(name)
    try:
        with open("/etc/localtime", "rb") as file:
            return ZoneInfo.from_file(file)
    except OSError:
        return datetime.now().astimezone().tzinfo


def list_models(db: DB) -> list[dict]:
    return _many(db, "SELECT * FROM models ORDER BY name,id")


def get_model(db: DB, id: int) -> dict | None:
    return _one(db, "SELECT * FROM models WHERE id=?", (id,))


def save_model(
    db: DB,
    id: int | None,
    name: str,
    base_url: str,
    model: str,
    api_key: str | None,
    json_mode: str,
    temperature: float,
    max_tokens: int,
) -> int:
    if json_mode not in {"schema", "object", "none"} or max_tokens < 1:
        raise ValueError("Invalid model settings")
    with tx(db):
        row = db.execute(
            "INSERT INTO models(id,name,base_url,model,api_key,json_mode,temperature,max_tokens) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
            "base_url=excluded.base_url,model=excluded.model,api_key=excluded.api_key,"
            "json_mode=excluded.json_mode,temperature=excluded.temperature,"
            "max_tokens=excluded.max_tokens "
            "RETURNING id",
            (id, name, base_url, model, api_key, json_mode, temperature, max_tokens),
        ).fetchone()
        return row[0]


def delete_model(db: DB, id: int) -> None:
    with tx(db):
        db.execute("DELETE FROM models WHERE id=?", (id,))


def list_agents(db: DB) -> list[dict]:
    return _many(
        db,
        "SELECT a.*, CASE WHEN paused THEN 'paused' "
        "WHEN EXISTS(SELECT 1 FROM runs WHERE agent_id=a.id AND status='running') "
        "THEN 'working' WHEN EXISTS(SELECT 1 FROM runs WHERE agent_id=a.id "
        "AND status='waiting') THEN 'waiting' ELSE 'idle' END AS status, "
        "(SELECT min(next_at) FROM schedules WHERE agent_id=a.id AND enabled=1) AS next_at "
        "FROM agents a ORDER BY a.name,a.id",
        fields=("tools",),
    )


def get_agent(db: DB, key: int | str) -> dict | None:
    return next(
        (agent for agent in list_agents(db) if agent["id"] == key or agent["name"] == key), None
    )


def save_agent(
    db: DB,
    id: int | None,
    name: str,
    role: str,
    model_id: int | None,
    tools: list[str],
    max_steps: int = 12,
    paused: bool = False,
) -> int:
    if not name.strip() or max_steps < 1 or not tools or len(set(tools)) != len(tools):
        raise ValueError("An agent needs a name, unique tools, and a positive step limit")
    with tx(db):
        return db.execute(
            "INSERT INTO agents(id,name,role,model_id,tools,max_steps,paused,created_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
            "role=excluded.role,model_id=excluded.model_id,tools=excluded.tools,"
            "max_steps=excluded.max_steps,paused=excluded.paused RETURNING id",
            (id, name, role, model_id, dumps(tools), max_steps, paused, now()),
        ).fetchone()[0]


def delete_agent(db: DB, id: int) -> None:
    with tx(db):
        # Tasks stay on the owner's board when their assignee is removed.
        tasks = _many(db, "SELECT id,title,details FROM tasks WHERE agent_id=?", (id,))
        db.execute("DELETE FROM agents WHERE id=?", (id,))
        for task in tasks:
            _index(db, "task", task["id"], task["title"], task["details"])


def parse_every(every: str) -> Callable[[datetime], datetime]:
    interval = re.fullmatch(r"(\d+)([mh])", every.strip())
    if interval:
        minutes = int(interval[1]) * (60 if interval[2] == "h" else 1)
        if minutes < 5:
            raise ValueError("Schedule interval must be at least 5 minutes")

        def next_interval(dt: datetime) -> datetime:
            return (dt.astimezone(UTC) + timedelta(minutes=minutes)).astimezone(dt.tzinfo)

        return next_interval
    calendar = re.fullmatch(r"(daily|weekdays|[a-z,]+) (\d{2}):(\d{2})", every.strip())
    if not calendar:
        raise ValueError("Use Nm, Nh, daily HH:MM, weekdays HH:MM, or mon,thu HH:MM")
    day, hour, minute = calendar.groups()
    if int(hour) > 23 or int(minute) > 59:
        raise ValueError("Schedule time must be between 00:00 and 23:59")
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    selected = days if day == "daily" else days[:5] if day == "weekdays" else day.split(",")
    if any(item not in days for item in selected) or len(set(selected)) != len(selected):
        raise ValueError("Day list must contain unique names from mon through sun")

    def next_calendar(dt: datetime) -> datetime:
        dt = dt if dt.tzinfo else dt.astimezone()
        for offset in range(8):
            date = dt.date() + timedelta(days=offset)
            if days[date.weekday()] not in selected:
                continue
            candidate = datetime(
                date.year, date.month, date.day, int(hour), int(minute), tzinfo=dt.tzinfo
            )
            # A nonexistent spring-forward time moves to the next valid local time.
            candidate = candidate.astimezone(UTC).astimezone(dt.tzinfo)
            if candidate.astimezone(UTC) > dt.astimezone(UTC):
                return candidate
        raise ValueError("No next schedule time")

    return next_calendar


def list_schedules(db: DB, agent_id: int | None = None) -> list[dict]:
    return _many(
        db,
        "SELECT * FROM schedules WHERE (? IS NULL OR agent_id=?) ORDER BY id",
        (agent_id, agent_id),
    )


def save_schedule(
    db: DB, id: int | None, agent_id: int, every: str, prompt: str, enabled: bool = True
) -> int:
    next_at = parse_every(every)(datetime.now(timezone(db))).astimezone(UTC)
    with tx(db):
        return db.execute(
            "INSERT INTO schedules(id,agent_id,every,prompt,next_at,enabled) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET agent_id=excluded.agent_id,every=excluded.every,"
            "prompt=excluded.prompt,next_at=excluded.next_at,enabled=excluded.enabled RETURNING id",
            (id, agent_id, every, prompt, next_at.strftime("%Y-%m-%dT%H:%M:%SZ"), enabled),
        ).fetchone()[0]


def delete_schedule(db: DB, id: int) -> None:
    with tx(db):
        db.execute("DELETE FROM schedules WHERE id=?", (id,))


def add_task(
    db: DB,
    title: str,
    details: str = "",
    to: str | None = None,
    created_by: str = "you",
    parent_id: int | None = None,
) -> int:
    agent = get_agent(db, to) if to and to != "you" else None
    if to and to != "you" and agent is None:
        raise ValueError(f"Unknown agent: {to}")
    with tx(db):
        id = db.execute(
            "INSERT INTO tasks(title,details,agent_id,created_by,parent_id,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (title, details, agent["id"] if agent else None, created_by, parent_id, now(), now()),
        ).lastrowid
        _index(db, "task", id, title, details)
        add_event(db, created_by, "task", str(id), title)
        if agent:
            from tholos.prompt import task_trigger

            queue_run(
                db, agent["id"], task_trigger(id, created_by, title, details), "task", task_id=id
            )
        return id


def list_tasks(db: DB, status: str | None = None, limit: int = 100) -> list[dict]:
    return _many(
        db,
        "SELECT t.*,a.name AS agent FROM tasks t LEFT JOIN agents a ON a.id=t.agent_id "
        "WHERE (? IS NULL OR t.status=?) ORDER BY t.id DESC LIMIT ?",
        (status, status, limit),
    )


def set_task(db: DB, id: int, status: str | None = None, result: str | None = None) -> None:
    with tx(db):
        db.execute(
            "UPDATE tasks SET status=coalesce(?,status),result=coalesce(?,result),updated_at=? "
            "WHERE id=?",
            (status, result, now(), id),
        )
        task = _one(db, "SELECT * FROM tasks WHERE id=?", (id,))
        if task:
            _index(db, "task", id, task["title"], task["details"] + " " + (task["result"] or ""))
            add_event(db, "you", "task", str(id), f"Task {task['status']}")


def queue_run(
    db: DB,
    agent_id: int,
    trigger: str,
    kind: str,
    task_id: int | None = None,
    due_at: str | None = None,
    dedupe_key: str | None = None,
) -> int | None:
    with tx(db):
        row = db.execute(
            "INSERT INTO runs(agent_id,task_id,trigger,trigger_kind,due_at,dedupe_key,created_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(dedupe_key) DO NOTHING RETURNING id",
            (agent_id, task_id, trigger, kind, due_at or now(), dedupe_key, now()),
        ).fetchone()
        if row:
            add_event(db, "worker", "run", str(row[0]), "Run queued")
        return row[0] if row else None


def list_runs(
    db: DB, agent_id: int | None = None, status: str | None = None, limit: int = 50
) -> list[dict]:
    return _many(
        db,
        "SELECT r.*,a.name AS agent FROM runs r JOIN agents a ON a.id=r.agent_id "
        "WHERE (? IS NULL OR agent_id=?) AND (? IS NULL OR r.status=?) "
        "ORDER BY r.id DESC LIMIT ?",
        (agent_id, agent_id, status, status, limit),
        ("messages",),
    )


def get_run(db: DB, id: int) -> dict | None:
    run = _one(db, "SELECT * FROM runs WHERE id=?", (id,), ("messages",))
    if run:
        run["step_count"] = run["steps"]
        run["steps"] = _many(
            db, "SELECT * FROM steps WHERE run_id=? ORDER BY n", (id,), ("args", "result")
        )
    return run


def stop_run(db: DB, id: int) -> None:
    with tx(db):
        db.execute(
            "UPDATE runs SET status='stopped',fence=fence+1,lease_until=NULL,ended_at=? "
            "WHERE id=? AND status IN ('queued','running','waiting')",
            (now(), id),
        )
        db.execute(
            "UPDATE approvals SET status='cancelled',decided_at=? "
            "WHERE run_id=? AND status='pending'",
            (now(), id),
        )
        add_event(db, "you", "run", str(id), "Run stopped")


def list_waiting(db: DB) -> list[dict]:
    return _many(
        db,
        "SELECT p.*,a.name AS agent FROM approvals p JOIN agents a ON a.id=p.agent_id "
        "WHERE p.status='pending' ORDER BY p.id",
        fields=("args",),
    )


def list_tables(db: DB) -> list[dict]:
    return _many(
        db,
        "SELECT s.*,(SELECT count(*) FROM rows WHERE sheet_id=s.id AND deleted=0) "
        "AS count FROM sheets s ORDER BY s.name,s.id",
        fields=("columns",),
    )


def get_table(db: DB, name: str) -> dict | None:
    sheet = _one(db, "SELECT * FROM sheets WHERE name=?", (name,), ("columns",))
    if sheet:
        sheet["rows"] = _many(
            db,
            "SELECT * FROM rows WHERE sheet_id=? AND deleted=0 ORDER BY id",
            (sheet["id"],),
            ("data",),
        )
    return sheet


def create_table(
    db: DB, name: str, columns: list[str], actor: str, run_id: int | None = None
) -> int:
    if (
        not name.strip()
        or not 1 <= len(columns) <= 12
        or len(set(columns)) != len(columns)
        or any(not isinstance(col, str) or not col.strip() for col in columns)
    ):
        raise ValueError("A table needs a name and 1 to 12 unique, nonempty columns")
    with tx(db):
        id = db.execute(
            "INSERT INTO sheets(name,columns,created_by,created_at,updated_at) VALUES(?,?,?,?,?)",
            (name, dumps(columns), actor, now(), now()),
        ).lastrowid
        _change(db, "table", None, get_table(db, name), actor, run_id)
        return id


def _sheet_changed(db: DB, id: int) -> None:
    db.execute("UPDATE sheets SET version=version+1,updated_at=? WHERE id=?", (now(), id))


def add_rows(
    db: DB, name: str, rows: list[dict], actor: str, run_id: int | None = None
) -> list[int]:
    with tx(db):
        sheet = get_table(db, name)
        if sheet is None:
            raise ValueError(f"Unknown table: {name}")
        ids = []
        for values in rows:
            if set(values) - set(sheet["columns"]):
                raise ValueError("Unknown columns")
            data = {column: values.get(column, "") for column in sheet["columns"]}
            id = db.execute(
                "INSERT INTO rows(sheet_id,data,updated_by,updated_at) VALUES(?,?,?,?)",
                (sheet["id"], dumps(data), actor, now()),
            ).lastrowid
            after = _one(db, "SELECT * FROM rows WHERE id=?", (id,), ("data",))
            _change(db, "row", None, after, actor, run_id)
            _index(db, "row", id, name, " ".join(str(v) for v in data.values()))
            ids.append(id)
        if ids:
            _sheet_changed(db, sheet["id"])
        return ids


def update_row(
    db: DB,
    name: str,
    row_id: int,
    values: dict,
    actor: str,
    expected_version: int | None = None,
    run_id: int | None = None,
) -> dict:
    with tx(db):
        sheet = get_table(db, name)
        before = next((r for r in sheet["rows"] if r["id"] == row_id), None) if sheet else None
        if before is None:
            raise ValueError("Unknown row")
        if expected_version is not None and before["version"] != expected_version:
            raise Conflict(before)
        if set(values) - set(sheet["columns"]):
            raise ValueError("Unknown columns")
        data = before["data"] | values
        db.execute(
            "UPDATE rows SET data=?,version=version+1,updated_by=?,updated_at=? WHERE id=?",
            (dumps(data), actor, now(), row_id),
        )
        after = _one(db, "SELECT * FROM rows WHERE id=?", (row_id,), ("data",))
        _change(db, "row", before, after, actor, run_id)
        _index(db, "row", row_id, name, " ".join(str(v) for v in data.values()))
        _sheet_changed(db, sheet["id"])
        return after


def delete_row(db: DB, name: str, row_id: int, actor: str) -> None:
    with tx(db):
        sheet = get_table(db, name)
        before = next((r for r in sheet["rows"] if r["id"] == row_id), None) if sheet else None
        if before is None:
            raise ValueError("Unknown row")
        db.execute(
            "UPDATE rows SET deleted=1,version=version+1,updated_by=?,updated_at=? WHERE id=?",
            (actor, now(), row_id),
        )
        after = _one(db, "SELECT * FROM rows WHERE id=?", (row_id,), ("data",))
        _change(db, "row", before, after, actor)
        _index(db, "row", row_id, name, None)
        _sheet_changed(db, sheet["id"])


def export_table(db: DB, name: str, fmt: str) -> bytes:
    sheet = get_table(db, name)
    if sheet is None:
        raise ValueError("Unknown table")
    values = [sheet["columns"]] + [
        [row["data"][col] for col in sheet["columns"]] for row in sheet["rows"]
    ]
    if fmt == "csv":
        out = io.StringIO(newline="")
        csv.writer(out).writerows(
            [
                ["'" + str(v) if str(v).startswith(("=", "+", "-", "@")) else v for v in row]
                for row in values
            ]
        )
        return out.getvalue().encode("utf-8")
    if fmt != "xlsx":
        raise ValueError("Export format must be csv or xlsx")
    from openpyxl import Workbook

    book = Workbook()
    for row in values:
        book.active.append(row)
        for cell in book.active[book.active.max_row]:
            if str(cell.value).startswith(("=", "+", "-", "@")):
                cell.value = str(cell.value)
                cell.data_type = "s"
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def list_notes(db: DB) -> list[dict]:
    return _many(db, "SELECT * FROM notes ORDER BY title,id")


def get_note(db: DB, title: str) -> dict | None:
    return _one(db, "SELECT * FROM notes WHERE title=?", (title,))


def write_note(
    db: DB,
    title: str,
    text: str,
    actor: str,
    mode: str = "replace",
    expected_version: int | None = None,
    run_id: int | None = None,
) -> dict:
    if mode not in {"replace", "append"}:
        raise ValueError("Note mode must be replace or append")
    with tx(db):
        before = get_note(db, title)
        if (
            expected_version is not None
            and (before or {"version": 0})["version"] != expected_version
        ):
            raise Conflict(before or {"title": title, "version": 0})
        body = before["body"] + text if before and mode == "append" else text
        db.execute(
            "INSERT INTO notes(title,body,updated_by,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(title) DO UPDATE SET body=excluded.body,version=notes.version+1,"
            "updated_by=excluded.updated_by,updated_at=excluded.updated_at",
            (title, body, actor, now()),
        )
        after = get_note(db, title)
        _change(db, "note", before, after, actor, run_id)
        _index(db, "note", after["id"], title, body)
        return after


def delete_note(db: DB, title: str, actor: str) -> None:
    with tx(db):
        before = get_note(db, title)
        if before:
            _change(db, "note", before, None, actor)
            _index(db, "note", before["id"], title, None)
            db.execute("DELETE FROM notes WHERE id=?", (before["id"],))


def list_memories(db: DB, agent_id: int) -> list[dict]:
    return _many(db, "SELECT * FROM memories WHERE agent_id=? ORDER BY id DESC", (agent_id,))


def add_memory(db: DB, agent_id: int, text: str, source: str) -> int:
    with tx(db):
        return db.execute(
            "INSERT INTO memories(agent_id,text,source,created_at) VALUES(?,?,?,?)",
            (agent_id, text, source, now()),
        ).lastrowid


def update_memory(db: DB, id: int, text: str) -> None:
    with tx(db):
        db.execute("UPDATE memories SET text=? WHERE id=?", (text, id))


def delete_memory(db: DB, id: int) -> None:
    with tx(db):
        db.execute("DELETE FROM memories WHERE id=?", (id,))


def list_rules(db: DB) -> list[dict]:
    return _many(db, "SELECT * FROM rules ORDER BY id")


def add_rule(db: DB, tool: str, decision: str, agent: str = "*", match: str = "*") -> int:
    if decision not in {"allow", "ask", "deny"}:
        raise ValueError("Rule decision must be allow, ask, or deny")
    with tx(db):
        return db.execute(
            "INSERT INTO rules(agent,tool,match,decision,created_at) VALUES(?,?,?,?,?)",
            (agent, tool, match, decision, now()),
        ).lastrowid


def delete_rule(db: DB, id: int) -> None:
    with tx(db):
        db.execute("DELETE FROM rules WHERE id=?", (id,))


def events_since(db: DB, seq: int, limit: int = 200) -> list[dict]:
    return _many(db, "SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?", (seq, limit))


def recent_changes(
    db: DB, kind: str | None = None, ref_id: int | None = None, limit: int = 50
) -> list[dict]:
    return _many(
        db,
        "SELECT * FROM history WHERE (? IS NULL OR kind=?) "
        "AND (? IS NULL OR ref_id=?) ORDER BY id DESC LIMIT ?",
        (kind, kind, ref_id, ref_id, limit),
        ("before", "after"),
    )


def search(db: DB, query: str, limit: int = 20) -> list[dict]:
    try:
        return _many(
            db,
            "SELECT kind,ref,title,snippet(search,3,'','', '...',24) AS snippet "
            "FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        )
    except sqlite3.OperationalError as exc:
        raise ValueError("Invalid search query") from exc


def load_team(db: DB, name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.is_file():
        path = Path(__file__).parent / "teams" / (name_or_path.removesuffix(".json") + ".json")
    team = json.loads(path.read_text(encoding="utf-8"))
    with tx(db):
        for table in team.get("tables", []):
            if get_table(db, table["name"]) is None:
                create_table(db, table["name"], table["columns"], "you")
        for note in team.get("notes", []):
            if get_note(db, note["title"]) is None:
                write_note(db, note["title"], note["body"], "you")
        for agent in team.get("agents", []):
            if get_agent(db, agent["name"]) is not None:
                continue
            id = save_agent(
                db,
                None,
                agent["name"],
                agent["role"],
                None,
                agent["tools"],
                agent.get("max_steps", 12),
            )
            for schedule in agent.get("schedules", []):
                save_schedule(db, None, id, schedule["every"], schedule["prompt"])
        for rule in team.get("rules", []):
            if not any(
                all(existing[k] == rule.get(k, "*") for k in ("agent", "tool", "match"))
                and existing["decision"] == rule["decision"]
                for existing in list_rules(db)
            ):
                add_rule(
                    db,
                    rule["tool"],
                    rule["decision"],
                    rule.get("agent", "*"),
                    rule.get("match", "*"),
                )
    return team


def render_markdown(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*\n]+)\*", r"<em>\1</em>", text)
    text = re.sub(
        r"\[([^]\n]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" rel="noreferrer">\1</a>', text
    )
    lines = []
    listing = False
    for line in text.splitlines():
        if line.startswith("- "):
            if not listing:
                lines.append("<ul>")
            listing = True
            lines.append(f"<li>{line[2:]}</li>")
            continue
        if listing:
            lines.append("</ul>")
            listing = False
        heading = re.match(r"^(#{1,6}) (.*)$", line)
        lines.append(
            f"<h{len(heading[1])}>{heading[2]}</h{len(heading[1])}>"
            if heading
            else f"<p>{line}</p>"
        )
    if listing:
        lines.append("</ul>")
    return "\n".join(lines)

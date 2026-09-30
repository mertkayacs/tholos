import asyncio
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx

from tholos import fetch, prompt, runner, worker
from tholos import workspace as w
from tholos.db import connect, init


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip())
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "contains" in expected:
            return str(expected["contains"]).casefold() in str(value).casefold()
        if "in" in expected:
            return any(matches(value, item) for item in expected["in"])
        if "nonempty" in expected:
            return bool(value is not None and str(value).strip()) == expected["nonempty"]
        if "gte" in expected or "lte" in expected:
            number = _number(value)
            if number is None:
                return False
            return ("gte" not in expected or number >= Decimal(str(expected["gte"]))) and (
                "lte" not in expected or number <= Decimal(str(expected["lte"]))
            )
        return isinstance(value, dict) and all(
            key in value and matches(value[key], item) for key, item in expected.items()
        )
    numeric, wanted = _number(value), _number(expected)
    if numeric is not None and wanted is not None:
        return numeric == wanted
    if isinstance(value, str) and isinstance(expected, str):
        return value.strip().casefold() == expected.strip().casefold()
    return type(value) is type(expected) and value == expected


def _where(data: dict, conditions: dict) -> bool:
    return all(key in data and matches(data[key], value) for key, value in conditions.items())


def _version(item: dict | None) -> tuple | None:
    return (item["id"], item["version"]) if item else None


def _contains_token(text: str, token: str) -> bool:
    token = token.casefold()
    pattern = re.escape(token)
    if token[0].isalnum():
        pattern = r"\b" + pattern
    if token[-1].isalnum():
        pattern += r"\b"
    return re.search(pattern, text.casefold()) is not None


def snapshot(db: w.DB) -> dict:
    return {
        "tables": {table["name"]: _version(table) for table in w.list_tables(db)},
        "notes": {note["title"]: _version(note) for note in w.list_notes(db)},
        "tasks": {task["id"] for task in w.list_tasks(db, limit=100000)},
        "memories": {row[0] for row in db.execute("SELECT id FROM memories")},
        "runs": {row[0] for row in db.execute("SELECT id FROM runs")},
    }


def evaluate(db: w.DB, run_id: int, expect: list[dict], initial: dict) -> list[dict]:
    run = w.get_run(db, run_id)
    steps = run["steps"]
    tasks = [task for task in w.list_tasks(db, limit=100000) if task["id"] not in initial["tasks"]]
    finishes = [
        step["args"]["summary"]
        for step in steps
        if step["tool"] == "finish"
        and "error" not in step["result"]
        and step["status"] != "waiting"
    ]
    failed = []
    for assertion in expect:
        kind = assertion["type"]
        passed = False
        if kind == "finished":
            passed = run["status"] == "done" and bool(finishes)
        elif kind == "status":
            passed = run["status"] == assertion["is"]
        elif kind in {"row", "rows", "table_exists"}:
            table = w.get_table(db, assertion["table"])
            if table and kind == "row":
                passed = any(
                    _where(row["data"], assertion["where"])
                    and _where(row["data"], assertion.get("has", {}))
                    for row in table["rows"]
                )
            elif table and kind == "rows":
                count = len(table["rows"])
                passed = (
                    count == assertion.get("count", count)
                    and count >= assertion.get("min", 0)
                    and count <= assertion.get("max", count)
                )
            elif table:
                passed = "columns" not in assertion or set(table["columns"]) == set(
                    assertion["columns"]
                )
        elif kind == "note":
            note = w.get_note(db, assertion["title"])
            if note:
                body = note["body"].casefold()
                passed = all(text.casefold() in body for text in assertion.get("contains", []))
                passed &= all(
                    text.casefold() not in body for text in assertion.get("not_contains", [])
                )
        elif kind == "unchanged":
            if "table" in assertion:
                name = assertion["table"]
                original = initial["tables"].get(name)
                current = w.get_table(db, name)
            else:
                name = assertion["note"]
                original = initial["notes"].get(name)
                current = w.get_note(db, name)
            passed = original is not None and _version(current) == original
        elif kind in {"task", "no_task"}:
            assigned = [
                task
                for task in tasks
                if "to" not in assertion or (task["agent"] or "you") == assertion["to"]
            ]
            found = any(
                assertion.get("title_contains", "").casefold() in task["title"].casefold()
                for task in assigned
            )
            if kind == "task":
                text = "\n".join(
                    f"{task['title']}\n{task['details']}" for task in assigned
                ).casefold()
                found &= all(
                    mention.casefold() in text for mention in assertion.get("mentions", [])
                )
            passed = found if kind == "task" else not found
        elif kind in {"called", "not_called", "asked"}:
            name = "ask" if kind == "asked" else assertion["tool"]
            found = any(
                step["tool"] == name and _where(step["args"], assertion.get("args", {}))
                for step in steps
            )
            passed = not found if kind == "not_called" else found
        elif kind == "approval":
            passed = (
                db.execute(
                    "SELECT 1 FROM approvals WHERE run_id=? AND kind='approve' AND tool=?",
                    (run_id, assertion["tool"]),
                ).fetchone()
                is not None
            )
        elif kind == "memory":
            agent = w.get_agent(db, assertion["agent"])
            passed = bool(agent) and any(
                memory["id"] not in initial["memories"]
                and assertion["contains"].casefold() in memory["text"].casefold()
                for memory in w.list_memories(db, agent["id"])
            )
        elif kind == "follow_up":
            for follow in w.list_runs(db, limit=100000):
                if follow["id"] in initial["runs"] or follow["trigger_kind"] != "follow_up":
                    continue
                minutes = (
                    datetime.fromisoformat(follow["due_at"])
                    - datetime.fromisoformat(follow["created_at"])
                ).total_seconds() / 60
                passed |= (
                    assertion["min_minutes"] - 1 / 60
                    <= minutes
                    <= assertion["max_minutes"] + 1 / 60
                )
        elif kind == "finish_contains":
            passed = any(
                all(_contains_token(summary, text) for text in assertion.get("all", []))
                and (
                    "any" not in assertion
                    or any(_contains_token(summary, text) for text in assertion["any"])
                )
                for summary in finishes
            )
        elif kind == "max_steps":
            passed = run["step_count"] <= assertion["n"]
        else:
            raise ValueError(f"Unknown assertion type: {kind}")
        if not passed:
            failed.append(assertion)
    return failed


def _load(db: w.DB, scenario: dict, profile: dict) -> int:
    mid = w.save_model(
        db,
        None,
        "Benchmark",
        profile["base_url"],
        profile["model"],
        profile.get("api_key"),
        profile.get("json_mode", "schema"),
        profile.get("temperature", 0),
        profile.get("max_tokens", 512),
    )
    workspace = scenario["workspace"]
    for agent in workspace["agents"]:
        w.save_agent(
            db,
            None,
            agent["name"],
            agent["role"],
            mid,
            agent["tools"],
            scenario["max_steps"]
            if agent["name"] == scenario["agent"]
            else agent.get("max_steps", 12),
        )
    for table in workspace.get("tables", []):
        w.create_table(db, table["name"], table["columns"], "you")
        w.add_rows(db, table["name"], table.get("rows", []), "you")
    for note in workspace.get("notes", []):
        w.write_note(db, note["title"], note["body"], "you")
    for rule in workspace.get("rules", []):
        w.add_rule(
            db, rule["tool"], rule["decision"], rule.get("agent", "*"), rule.get("match", "*")
        )
    for memory in workspace.get("memories", []):
        w.add_memory(db, w.get_agent(db, memory["agent"])["id"], memory["text"], "you")
    for task in workspace.get("tasks", []):
        id = w.add_task(db, task["title"], task.get("details", ""), task.get("to"))
        w.set_task(db, id, task.get("status", "todo"))
        for run in w.list_runs(db, limit=100000):
            if run["task_id"] == id:
                w.stop_run(db, run["id"])
    agent = w.get_agent(db, scenario["agent"])
    trigger = scenario["trigger"]
    kind = trigger["kind"]
    if kind == "task":
        task_id = w.add_task(
            db,
            trigger["title"],
            trigger.get("details", ""),
            agent["name"],
            trigger.get("from", "you"),
        )
        return db.execute("SELECT id FROM runs WHERE task_id=?", (task_id,)).fetchone()[0]
    text = {
        "schedule": lambda: prompt.schedule_trigger(trigger["prompt"]),
        "follow_up": lambda: prompt.follow_up_trigger(trigger["note"]),
        "message": lambda: prompt.message_trigger(trigger["text"]),
    }[kind]()
    return w.queue_run(db, agent["id"], text, kind)


def _interference(scenario: dict) -> Callable[[w.DB, int, str], None]:
    done = False

    def after_step(db: w.DB, run_id: int, tool: str) -> None:
        nonlocal done
        edit = scenario.get("interfere")
        if done or not edit or tool != edit["after_tool"]:
            return
        args = w.get_run(db, run_id)["steps"][-1]["args"]
        if "table" in edit and args.get("table") == edit["table"]:
            table = w.get_table(db, edit["table"])
            row = next(row for row in table["rows"] if _where(row["data"], edit["row_match"]))
            w.update_row(db, edit["table"], row["id"], edit["set"], "you")
            done = True
        elif "note" in edit and args.get("title") == edit["note"]:
            w.write_note(db, edit["note"], edit["append"], "you", "append")
            done = True

    return after_step


async def _execute(
    db: w.DB, run_id: int, scenario: dict, transport: httpx.BaseTransport | None
) -> None:
    after_step = _interference(scenario)
    while True:
        claimed = worker.claim(db, run_id=run_id)
        if claimed:
            await runner.run_one(db, claimed, transport, after_step)
        run = w.get_run(db, run_id)
        if run["status"] != "waiting":
            return
        respond = scenario.get("respond", {})
        approval = next(item for item in w.list_waiting(db) if item["run_id"] == run_id)
        if approval["kind"] == "question" and "answer" in respond:
            runner.answer(db, approval["id"], respond["answer"])
        elif approval["kind"] == "approve" and "approve" in respond:
            runner.decide(db, approval["id"], respond["approve"])
        else:
            return


def run_scenario(
    scenario: dict, profile: dict, transport: httpx.BaseTransport | None = None
) -> dict:
    previous = fetch.FIXTURES
    start = time.monotonic()
    with TemporaryDirectory(prefix="tholos-bench-") as directory:
        db = connect(str(Path(directory) / "tholos.db"))
        try:
            init(db)
            run_id = _load(db, scenario, profile)
            initial = snapshot(db)
            fetch.FIXTURES = scenario.get("fixtures", {})
            start = time.monotonic()
            asyncio.run(_execute(db, run_id, scenario, transport))
            failed = evaluate(db, run_id, scenario["expect"], initial)
            run = w.get_run(db, run_id)
            invalid = sum(
                step["result"].get("invalid_json_count", 0) + (step["status"] == "retried")
                for step in run["steps"]
            )
            return {
                "id": scenario["id"],
                "category": scenario["category"],
                "passed": not failed,
                "failed_assertions": failed,
                "steps": run["step_count"],
                "seconds": round(time.monotonic() - start, 3),
                "tokens": {"in": run["tokens_in"], "out": run["tokens_out"]},
                "invalid_json_count": invalid,
                "messages": run["messages"],
            }
        finally:
            fetch.FIXTURES = previous
            db.close()


def bench(
    base_url: str,
    model: str,
    api_key: str | None = None,
    json_mode: str = "schema",
    scenarios: str | None = None,
    only: str | None = None,
    limit: int | None = None,
    out: str | None = None,
) -> list[dict]:
    directory = Path(scenarios) if scenarios else Path(__file__).parent / "scenarios"
    items = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.rglob("*.json"))
    ]
    items = [item for item in items if only is None or item["category"] == only]
    if limit is not None:
        if limit < 0:
            raise ValueError("Benchmark limit must be nonnegative")
        items = items[:limit]
    if not items:
        raise ValueError("No benchmark scenarios matched")
    profile = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "json_mode": json_mode,
        "temperature": 0,
        "max_tokens": 512,
    }
    results = []
    counts = defaultdict(lambda: [0, 0])
    with (open(out, "w", encoding="utf-8") if out else nullcontext()) as file:
        for scenario in items:
            result = run_scenario(scenario, profile)
            results.append(result)
            counts[result["category"]][0] += result["passed"]
            counts[result["category"]][1] += 1
            if file:
                file.write(w.dumps(result) + "\n")
                file.flush()
    print(f"{'Category':<24} {'Passed':>7} {'Total':>7} {'Success':>9}")
    for category, (passed, total) in sorted(counts.items()):
        print(f"{category:<24} {passed:>7} {total:>7} {passed / total:>8.1%}")
    passed = sum(result["passed"] for result in results)
    print(f"Overall: {passed}/{len(results)} ({passed / len(results):.1%})")
    return results

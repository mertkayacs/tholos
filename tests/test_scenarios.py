import json
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "tholos" / "bench" / "scenarios"

CATEGORIES = {
    "table_read_answer": 12,
    "table_add": 14,
    "table_update": 14,
    "table_create": 8,
    "notes": 12,
    "handoff": 14,
    "approval": 14,
    "ask": 12,
    "conflict": 8,
    "memory": 10,
    "follow_up": 8,
    "web_research": 14,
    "injection": 12,
    "nothing_to_do": 8,
}

TOOLS = {
    "table_read",
    "table_create",
    "table_add",
    "table_update",
    "note_read",
    "note_write",
    "search",
    "task_add",
    "web_fetch",
    "ask",
    "remember",
    "follow_up",
    "finish",
}

TOOL_ARGS = {
    "table_read": ["table", "query", "limit"],
    "table_create": ["table", "columns"],
    "table_add": ["table", "rows"],
    "table_update": ["table", "row", "values"],
    "note_read": ["title"],
    "note_write": ["title", "text", "mode"],
    "search": ["query"],
    "task_add": ["to", "title", "details"],
    "web_fetch": ["url"],
    "ask": ["question"],
    "remember": ["fact"],
    "follow_up": ["minutes", "note"],
    "finish": ["summary"],
}

TRIGGER_KINDS = {"task", "schedule", "follow_up", "message"}

ASSERTIONS = {
    "finished": (set(), set()),
    "status": ({"is"}, set()),
    "row": ({"table", "where"}, {"has"}),
    "rows": ({"table"}, {"count", "min", "max"}),
    "table_exists": ({"table"}, {"columns"}),
    "note": ({"title"}, {"contains", "not_contains"}),
    "unchanged": (set(), {"table", "note"}),
    "task": ({"to"}, {"title_contains"}),
    "no_task": (set(), {"to"}),
    "called": ({"tool"}, {"args"}),
    "not_called": ({"tool"}, {"args"}),
    "asked": (set(), set()),
    "approval": ({"tool"}, set()),
    "memory": ({"agent", "contains"}, set()),
    "follow_up": ({"min_minutes", "max_minutes"}, set()),
    "finish_contains": ({"any"}, set()),
    "max_steps": ({"n"}, set()),
}

MATCH_KEYS = {"contains", "in", "gte", "lte", "nonempty"}


def load():
    result = {cat: [] for cat in CATEGORIES}
    for path in sorted(SCEN.rglob("*.json")):
        result.setdefault(path.parent.name, []).append(
            (path, json.loads(path.read_text(encoding="utf-8")))
        )
    return result


def all_scenarios():
    scenarios = []
    for files in load().values():
        scenarios.extend(files)
    return scenarios


def is_match(value):
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        if len(value) != 1 or not set(value) <= MATCH_KEYS:
            return False
        key, item = next(iter(value.items()))
        if key == "contains":
            return isinstance(item, str)
        if key == "in":
            return isinstance(item, list) and bool(item) and all(is_match(v) for v in item)
        if key == "nonempty":
            return item is True
        return type(item) in (int, float)
    return False


def validate_assertion(assertion, path):
    assert isinstance(assertion, dict), f"{path} must be an object"
    atype = assertion.get("type")
    assert atype in ASSERTIONS, f"{path} has unknown type {atype!r}"
    required, optional = ASSERTIONS[atype]
    keys = set(assertion) - {"type"}
    assert required <= keys, f"{path} ({atype}) missing {required - keys}"
    assert keys <= (required | optional), (
        f"{path} ({atype}) has unknown fields {keys - (required | optional)}"
    )
    for key in ("table", "title", "note", "to", "agent", "contains", "title_contains"):
        if key in assertion and not (atype == "note" and key == "contains"):
            assert isinstance(assertion[key], str) and assertion[key], f"{path} bad {key}"
    if atype == "status":
        assert assertion["is"] in {"done", "waiting", "failed"}, f"{path} bad status"
    if atype == "row":
        assert isinstance(assertion["where"], dict), f"{path} where must be object"
        for value in assertion["where"].values():
            assert is_match(value), f"{path} bad where match"
        if "has" in assertion:
            assert isinstance(assertion["has"], dict), f"{path} has must be object"
            for value in assertion["has"].values():
                assert is_match(value), f"{path} bad has match"
    if atype == "rows":
        assert any(k in assertion for k in ("count", "min", "max")), (
            f"{path} rows needs count/min/max"
        )
        assert all(type(assertion[k]) is int and assertion[k] >= 0 for k in keys - {"table"})
        assert "count" not in keys or not keys & {"min", "max"}, f"{path} mixed row bounds"
        if "min" in keys and "max" in keys:
            assert assertion["min"] <= assertion["max"], f"{path} inverted row bounds"
    if atype == "table_exists" and "columns" in assertion:
        assert isinstance(assertion["columns"], list) and assertion["columns"]
        assert all(isinstance(col, str) and col for col in assertion["columns"])
    if atype in ("note",):
        for key in ("contains", "not_contains"):
            if key in assertion:
                assert isinstance(assertion[key], list), f"{path} {key} must be a list"
                assert all(isinstance(text, str) and text for text in assertion[key])
    if atype == "unchanged":
        assert ("table" in assertion) != ("note" in assertion), (
            f"{path} needs exactly one of table/note"
        )
    if atype in ("called", "not_called"):
        assert assertion["tool"] in TOOLS, f"{path} bad tool"
        if "args" in assertion:
            assert isinstance(assertion["args"], dict), f"{path} args must be object"
            assert set(assertion["args"]) <= set(TOOL_ARGS[assertion["tool"]])
            for value in assertion["args"].values():
                assert is_match(value), f"{path} bad args match"
    if atype == "approval":
        assert assertion["tool"] in TOOLS, f"{path} bad tool"
    if atype == "finish_contains":
        assert isinstance(assertion["any"], list) and assertion["any"], (
            f"{path} any must be nonempty list"
        )
        assert all(isinstance(text, str) and text for text in assertion["any"])
    if atype == "max_steps":
        assert isinstance(assertion["n"], int), f"{path} n must be int"
    if atype == "follow_up":
        assert all(type(assertion[k]) is int for k in ("min_minutes", "max_minutes"))
        assert 5 <= assertion["min_minutes"] <= assertion["max_minutes"] <= 10080
        assert assertion["min_minutes"] <= assertion["max_minutes"], f"{path} bad window"


def validate_trigger(trigger, path):
    assert isinstance(trigger, dict), f"{path} must be object"
    assert trigger.get("kind") in TRIGGER_KINDS, f"{path} bad kind"
    kind = trigger["kind"]
    if kind == "task":
        assert {"from", "title", "details"} <= set(trigger), f"{path} task needs from/title/details"
    elif kind == "schedule":
        assert "prompt" in trigger, f"{path} schedule needs prompt"
    elif kind == "follow_up":
        assert "note" in trigger, f"{path} follow_up needs note"
    elif kind == "message":
        assert "text" in trigger, f"{path} message needs text"
    fields = {
        "task": {"from", "title", "details"},
        "schedule": {"prompt"},
        "follow_up": {"note"},
        "message": {"text"},
    }[kind]
    assert set(trigger) == fields | {"kind"}, f"{path} unknown trigger fields"
    assert all(isinstance(trigger[key], str) for key in fields), f"{path} bad trigger text"


def validate_reference(reference, tools, path):
    assert isinstance(reference, list) and reference, f"{path} reference must be nonempty list"
    for i, step in enumerate(reference):
        step_path = f"{path}[{i}]"
        assert isinstance(step, dict), f"{step_path} must be object"
        assert set(step) == {"tool", "args"}, f"{step_path} must have only tool and args"
        tool = step["tool"]
        assert tool in TOOLS, f"{step_path} unknown tool {tool!r}"
        assert tool in tools, f"{step_path} tool {tool!r} not in agent tools"
        args = step["args"]
        assert isinstance(args, dict), f"{step_path} args must be object"
        assert set(args) == set(TOOL_ARGS[tool]), (
            f"{step_path} args keys {sorted(args)} != expected {TOOL_ARGS[tool]}"
        )
    assert reference[-1]["tool"] == "finish", f"{path} reference must end with finish"
    assert all(step["tool"] != "finish" for step in reference[:-1])


def test_all_files_are_valid_json_and_fields():
    scenarios = all_scenarios()
    assert scenarios, "no scenario files found"
    for path, sc in scenarios:
        required = {
            "id",
            "category",
            "template",
            "agent",
            "workspace",
            "trigger",
            "reference",
            "expect",
            "max_steps",
            "about",
        }
        assert required <= set(sc), f"{path} missing {required - set(sc)}"
        assert isinstance(sc["id"], str) and sc["id"], f"{path} bad id"
        assert isinstance(sc["category"], str), f"{path} bad category"
        assert sc["category"] in CATEGORIES, f"{path} unknown category {sc['category']!r}"
        assert path.parent.name == sc["category"], f"{path} wrong category directory"
        assert path.stem == sc["id"], f"{path} filename must match id"
        assert isinstance(sc["template"], str) and sc["template"].startswith("b-"), (
            f"{path} template must start with b-"
        )
        assert isinstance(sc["about"], str), f"{path} bad about"
        assert isinstance(sc["max_steps"], int), f"{path} bad max_steps"
        assert 3 <= sc["max_steps"] <= 14, f"{path} max_steps out of range"

        ws = sc["workspace"]
        assert isinstance(ws, dict) and "agents" in ws, f"{path} workspace needs agents"
        assert isinstance(ws["agents"], list) and ws["agents"], f"{path} agents empty"
        names = []
        for agent in ws["agents"]:
            assert isinstance(agent, dict), f"{path} bad agent"
            assert {"name", "role", "tools"} <= set(agent), f"{path} agent missing fields"
            assert "finish" in agent["tools"], f"{path} agent {agent['name']} lacks finish"
            for tool in agent["tools"]:
                assert tool in TOOLS, f"{path} agent has unknown tool {tool!r}"
            names.append(agent["name"])
        assert sc["agent"] in names, f"{path} primary agent not in workspace.agents"
        assert len(names) == len(set(names)), f"{path} duplicate agent names"
        for rule in ws.get("rules", []):
            assert rule["tool"] in TOOLS, f"{path} unknown rule tool"
            assert rule["decision"] in {"allow", "ask", "deny"}, f"{path} bad rule decision"

        primary = next(a for a in ws["agents"] if a["name"] == sc["agent"])
        validate_reference(sc["reference"], set(primary["tools"]), f"{path}.reference")

        assert isinstance(sc["expect"], list) and sc["expect"], f"{path} expect empty"
        for i, assertion in enumerate(sc["expect"]):
            validate_assertion(assertion, f"{path}.expect[{i}]")

        validate_trigger(sc["trigger"], f"{path}.trigger")
        assert sc["max_steps"] >= len(sc["reference"]), (
            f"{path} max_steps {sc['max_steps']} < len(reference) {len(sc['reference'])}"
        )


def test_counts_per_category():
    loaded = load()
    assert set(loaded) == set(CATEGORIES), "unknown category directories"
    assert sum(map(len, loaded.values())) == 160
    for cat, expected in CATEGORIES.items():
        assert len(loaded[cat]) == expected, f"category {cat} has {len(loaded[cat])} != {expected}"


def test_unique_ids():
    seen = {}
    for path, sc in all_scenarios():
        sc_id = sc["id"]
        assert sc_id not in seen, f"duplicate id {sc_id!r} in {path} and {seen.get(sc_id)}"
        seen[sc_id] = path


def test_template_usage():
    counts = {}
    for path, sc in all_scenarios():
        template = sc["template"]
        counts.setdefault(template, []).append(path)
    for template, paths in counts.items():
        assert len(paths) <= 3, f"template {template!r} used {len(paths)} times"


def test_fixture_hosts_end_with_test():
    for path, sc in all_scenarios():
        fixtures = sc.get("fixtures", {})
        assert isinstance(fixtures, dict), f"{path} fixtures must be object"
        for url, body in fixtures.items():
            assert isinstance(url, str) and isinstance(body, str), f"{path} bad fixture"
            parts = urlsplit(url)
            assert parts.scheme in {"http", "https"}, f"{path} bad fixture scheme"
            host = parts.hostname or ""
            assert host.endswith(".test"), f"{path} fixture host {host!r} does not end with .test"
            assert len(body.encode("utf-8")) < 4096, f"{path} fixture exceeds 4 KB"


def test_respond_and_interfere_shape():
    for path, sc in all_scenarios():
        respond = sc.get("respond")
        if respond is not None:
            assert isinstance(respond, dict), f"{path} respond must be object"
            assert set(respond) <= {"approve", "answer"}, f"{path} bad respond keys"
            if "approve" in respond:
                assert isinstance(respond["approve"], bool), f"{path} approve must be bool"
            if "answer" in respond:
                assert isinstance(respond["answer"], str), f"{path} answer must be str"
        interfere = sc.get("interfere")
        if sc["category"] == "conflict":
            assert interfere is not None, f"{path} conflict needs interference"
        if interfere is not None:
            assert isinstance(interfere, dict), f"{path} interfere must be object"
            assert "after_tool" in interfere and interfere["after_tool"] in TOOLS, (
                f"{path} interfere needs valid after_tool"
            )
            if "note" in interfere:
                assert set(interfere) == {"after_tool", "note", "append"}
                assert interfere["after_tool"] == "note_read"
                assert isinstance(interfere["append"], str) and interfere["append"]
                assert interfere["note"] in {n["title"] for n in sc["workspace"]["notes"]}
            else:
                assert set(interfere) == {"after_tool", "table", "row_match", "set"}
                assert interfere["after_tool"] == "table_read"
                assert isinstance(interfere["row_match"], dict), f"{path} row_match must be object"
                assert isinstance(interfere["set"], dict), f"{path} set must be object"
                tables = {t["name"]: t for t in sc["workspace"]["tables"]}
                assert interfere["table"] in tables
                columns = set(tables[interfere["table"]]["columns"])
                assert set(interfere["row_match"]) <= columns
                assert set(interfere["set"]) <= columns
        if any(step["tool"] == "ask" for step in sc["reference"]):
            assert isinstance(sc.get("respond", {}).get("answer"), str)


def test_category_contracts():
    scenarios = [sc for _path, sc in all_scenarios()]
    approvals = [sc for sc in scenarios if sc["category"] == "approval"]
    assert sum(sc.get("respond", {}).get("approve") is True for sc in approvals) == 7
    assert sum(sc.get("respond", {}).get("approve") is False for sc in approvals) == 7
    for sc in scenarios:
        kinds = {assertion["type"] for assertion in sc["expect"]}
        if sc["category"] == "ask":
            assert "asked" in kinds
        if sc["category"] == "injection":
            assert kinds & {"not_called", "unchanged"}
            assert kinds & {"row", "note", "finish_contains", "task"}


def test_no_em_dash_anywhere():
    for path, _sc in all_scenarios():
        text = path.read_text(encoding="utf-8")
        assert "\u2014" not in text, f"em dash in {path}"


BANNED = ["acme", "example corp", "lorem", "foo bar", "john doe", "jane doe"]


def test_no_placeholder_text():
    for path, _sc in all_scenarios():
        text = path.read_text(encoding="utf-8").casefold()
        for term in BANNED:
            assert not re.search(rf"\b{re.escape(term)}\b", text), f"placeholder {term!r} in {path}"


def test_table_writes_use_known_columns():
    for path, sc in all_scenarios():
        tables = {t["name"]: t for t in sc["workspace"].get("tables", [])}
        created = {}
        for step in sc["reference"]:
            tool, args = step["tool"], step["args"]
            if tool == "table_add":
                table = tables.get(args["table"]) or created.get(args["table"])
                assert table is not None, f"{path} table_add to unknown table {args['table']!r}"
                cols = set(table["columns"])
                for row in args["rows"]:
                    assert set(row) <= cols, f"{path} table_add unknown columns in {row}"
            elif tool == "table_update":
                table = tables.get(args["table"]) or created.get(args["table"])
                assert table is not None, f"{path} table_update to unknown table {args['table']!r}"
                assert set(args["values"]) <= set(table["columns"]), (
                    f"{path} table_update unknown columns"
                )
            elif tool == "table_create":
                assert 1 <= len(args["columns"]) <= 12, f"{path} table_create bad column count"
                created[args["table"]] = {"columns": args["columns"]}

"""Tests for the training data pipeline (CPU only, no network).

Covers the M-data.md contract:
- fixture packs pass the pack validator
- every template produces valid scenarios for 20 random packs
- every generated scenario passes the bench static validator
- assertions are satisfiable: the scripted reference passes the real runner
  through a fake model
- train scenarios stay disjoint from bench scenarios (exact and near-duplicate)
- build.py's split never shares a template between train and val
"""

import copy
import importlib.util
import json
import random
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import build as B  # noqa: E402
import packs as P  # noqa: E402
import phrasing as F  # noqa: E402
import pipeline as pipeline  # noqa: E402
import rollout as R  # noqa: E402
import templates as T  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "bench_static", ROOT / "tests" / "test_scenarios.py")
bench_static = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_static)

PROFILE = {
    "base_url": "http://local.test/v1",
    "model": "fake",
    "api_key": None,
    "json_mode": "schema",
    "temperature": 0,
    "max_tokens": 512,
}

CATEGORIES = {
    "table_read_answer", "table_add", "table_update", "table_create", "notes",
    "handoff", "approval", "ask", "conflict", "memory", "follow_up",
    "web_research", "injection", "nothing_to_do",
}

SEED = "test-seed"


def fixture_packs():
    packs = T.load_packs(ROOT / "train" / "fixture_packs.jsonl")
    assert len(packs) >= 20
    return packs


def transport(reference):
    steps = iter(reference)

    def handler(request):
        value = {"thought": "Next step.", **next(steps)}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(
                value, separators=(",", ":"), ensure_ascii=True)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        })

    return httpx.MockTransport(handler)


def build_for(tid, fn, pack, i):
    return fn(pack, random.Random(f"{SEED}:{tid}:{i}"))


def static_validate(scenario, path="scenario"):
    required = {"id", "category", "template", "agent", "workspace", "trigger",
                "reference", "expect", "max_steps", "about"}
    assert required <= set(scenario), f"{path} missing {required - set(scenario)}"
    assert scenario["template"].startswith("t-"), f"{path} template must start with t-"
    assert scenario["category"] in CATEGORIES, f"{path} bad category"
    assert 3 <= scenario["max_steps"] <= 14, f"{path} max_steps out of range"
    assert scenario["max_steps"] >= len(scenario["reference"]), f"{path} steps"
    assert isinstance(scenario["about"], str) and scenario["about"], f"{path} about"

    ws = scenario["workspace"]
    assert ws["agents"] and isinstance(ws["agents"], list), f"{path} agents"
    names = []
    for agent in ws["agents"]:
        assert {"name", "role", "tools"} <= set(agent), f"{path} agent fields"
        assert "finish" in agent["tools"], f"{path} agent lacks finish"
        for tool in agent["tools"]:
            assert tool in bench_static.TOOLS, f"{path} unknown tool {tool}"
        names.append(agent["name"])
    assert scenario["agent"] in names, f"{path} primary agent missing"
    primary = next(a for a in ws["agents"] if a["name"] == scenario["agent"])

    bench_static.validate_reference(scenario["reference"], set(primary["tools"]), path)
    assert scenario["expect"], f"{path} expect empty"
    for i, assertion in enumerate(scenario["expect"]):
        bench_static.validate_assertion(assertion, f"{path}.expect[{i}]")
    bench_static.validate_trigger(scenario["trigger"], f"{path}.trigger")

    fixtures = scenario.get("fixtures", {})
    for url, body in fixtures.items():
        host = url.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
        assert host.endswith(".test"), f"{path} fixture host {host}"
        assert len(body.encode()) < 4096, f"{path} fixture over 4 KB"

    respond = scenario.get("respond")
    if respond is not None:
        assert set(respond) <= {"approve", "answer"}, f"{path} respond keys"
    interfere = scenario.get("interfere")
    if interfere is not None:
        assert interfere["after_tool"] in bench_static.TOOLS, f"{path} interfere tool"
        if "note" in interfere:
            assert "append" in interfere
        else:
            assert {"table", "row_match", "set"} <= set(interfere)

    tables = {t["name"]: t for t in ws.get("tables", [])}
    created = {}
    for step in scenario["reference"]:
        tool, args = step["tool"], step["args"]
        if tool == "table_create":
            created[args["table"]] = {"columns": args["columns"]}
        elif tool in {"table_add", "table_update"}:
            table = tables.get(args["table"]) or created.get(args["table"])
            assert table is not None, f"{path} write to unknown table"
            cells = args["rows"] if tool == "table_add" else [args["values"]]
            for cell in cells:
                assert set(cell) <= set(table["columns"]), f"{path} unknown columns"

    assert "\u2014" not in json.dumps(scenario), f"{path} contains an em dash"


def test_fixture_packs_are_valid():
    for pack in fixture_packs():
        error = T.validate_pack(pack)
        assert error is None, (pack.get("domain"), error)


def test_template_coverage():
    assert len(T.TEMPLATES) >= 40
    ids = [tid for tid, _, _ in T.TEMPLATES]
    assert len(set(ids)) == len(ids)
    assert all(tid.startswith("t-") for tid in ids)
    counts = {}
    for _, category, _ in T.TEMPLATES:
        counts[category] = counts.get(category, 0) + 1
    assert set(counts) == CATEGORIES
    assert all(n >= 2 for n in counts.values())
    for more in ("table_update", "handoff", "approval", "web_research", "injection"):
        assert counts[more] >= 3, f"{more} needs at least 3 templates"


@pytest.mark.parametrize("tid,category,fn", T.TEMPLATES, ids=[t[0] for t in T.TEMPLATES])
def test_template_valid_scenarios(tid, category, fn):
    packs = fixture_packs()[:20]
    produced = 0
    for i, pack in enumerate(packs):
        scenario = build_for(tid, fn, pack, i)
        if scenario is None:
            continue
        produced += 1
        static_validate(scenario, f"{tid}[{i}]")
    assert produced >= 19, f"{tid} produced only {produced}/20 scenarios"


@pytest.mark.parametrize("tid,category,fn", T.TEMPLATES, ids=[t[0] for t in T.TEMPLATES])
def test_template_reference_passes_runner(tid, category, fn):
    from tholos.bench import runner as bench

    packs = fixture_packs()[:20]
    for i, pack in enumerate(packs):
        scenario = build_for(tid, fn, pack, i)
        if scenario is None:
            continue
        result = bench.run_scenario(scenario, PROFILE, transport(scenario["reference"]))
        assert result["passed"], (
            tid, i, result["failed_assertions"], scenario["id"])


def audit_pack():
    return {
        "domain": "a sailing club's checkout register",
        "tables": [{"name": "checkouts", "columns": [
            "checkout_id", "boat", "sailor", "minutes", "returned"], "rows": [
                {"checkout_id": f"co_{i}", "boat": "opti" if i < 4 else "laser",
                 "sailor": "theo" if i % 2 else "mira", "minutes": i * 10,
                 "returned": i % 2 == 0} for i in range(1, 9)]}],
        "notes": [{"title": "Checkout rules", "body": "\n".join(
            f"Rule {i}: return boat {i} before closing." for i in range(1, 9))},
            {"title": "Repairs", "body": "Check each rudder and sail before checkout."}],
        "team": [{"name": "Mira", "role": "You manage boat checkouts and returns."},
                 {"name": "Theo", "role": "You repair sails and rudders for the fleet."}],
        "fresh": [f"co_{i}, opti seabird, theo lund, {i}, true" for i in range(109, 114)],
    }


def test_fresh_csv_values_belong_in_separate_columns():
    table = audit_pack()["tables"][0]
    row = T._fresh_row(table, random.Random(1), "co_109, opti seabird, theo lund, 14, true")
    assert row == {"checkout_id": "co_109", "boat": "opti seabird",
                   "sailor": "theo lund", "minutes": 14, "returned": True}


def test_count_accepts_a_bare_number():
    from tholos.bench import runner as bench

    scenario = T.t_read_count(audit_pack(), random.Random(1))
    reference = copy.deepcopy(scenario["reference"])
    count = T._substr_count(scenario["workspace"]["tables"][0],
                           *reference[0]["args"]["query"].split("=", 1))
    reference[-1] = T._finish(str(count))
    assert bench.run_scenario(scenario, PROFILE, transport(reference))["passed"]


def test_all_log_triggers_specify_the_schema():
    for seed in range(15):
        scenario = T.t_create_log(audit_pack(), random.Random(seed))
        for column in scenario["reference"][0]["args"]["columns"]:
            assert column in _trigger_text(scenario)


def test_all_role_handoffs_identify_the_recipient():
    for seed in range(15):
        scenario = T.t_handoff_role(audit_pack(), random.Random(seed))
        assert scenario["reference"][0]["args"]["to"] in _trigger_text(scenario)


def test_note_cleanup_keeps_all_lines():
    scenario = T.t_notes_replace(audit_pack(), random.Random(2))
    title = scenario["reference"][1]["args"]["title"]
    original = next(n["body"] for n in scenario["workspace"]["notes"] if n["title"] == title)
    assert all(line in scenario["reference"][1]["args"]["text"]
               for line in original.splitlines())


def semantic_packs():
    nullable = audit_pack()
    nullable["tables"][0]["rows"][0]["sailor"] = None
    nullable["tables"].append({"name": "repairs", "columns": ["job", "detail", "state"],
                               "rows": [{"job": f"job_{i}", "detail": "sail",
                                         "state": "open" if i % 2 else "done"}
                                        for i in range(1, 9)]})
    return [*fixture_packs()[:2], fixture_packs()[4], audit_pack(), nullable]


def alternative_solution(scenario):
    steps = copy.deepcopy(scenario["reference"])
    for step in steps:
        args = step["args"]
        if step["tool"] == "table_read":
            args.update(query="*", limit=None)
        elif step["tool"] == "table_create":
            args["columns"].reverse()
        elif step["tool"] in {"table_add", "table_update"}:
            rows = args["rows"] if step["tool"] == "table_add" else [args["values"]]
            for row in rows:
                for col, value in row.items():
                    if isinstance(value, bool):
                        row[col] = str(value).lower()
                    elif isinstance(value, (int, float)):
                        row[col] = str(value)
        elif step["tool"] == "task_add":
            args["details"] = args["title"] + "\nPlease review: " + args["details"]
            args["title"] = f"Request for {args['to']}"
        elif step["tool"] == "note_write":
            args["text"] = args["text"].replace("\n- ", "\n* ")
        elif step["tool"] == "follow_up":
            args["note"] = "Please revisit: " + args["note"]
    for check in scenario.get("checks", []):
        if check["kind"] == "count":
            words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                     "nine", "ten", "eleven", "twelve", "thirteen", "fourteen"]
            value = words[check["value"]] if check["value"] < len(words) else str(check["value"])
            if check["tool"] == "finish":
                steps[-1]["args"]["summary"] = f"Matching entries: {value}."
            else:
                write = next(step for step in reversed(steps) if step["tool"] == "note_write")
                write["args"]["text"] = f"\n{check['context']} contains {value} entries."
                if any(c.get("style") == "one_line" for c in scenario.get("checks", [])):
                    write["args"]["text"] = write["args"]["text"].strip()
        elif check["kind"] == "compare":
            steps[-1]["args"]["summary"] = (
                f"{check['winner']} occurs more often than {check['loser']}.")
        elif check["kind"] == "preference":
            alternatives = {
                T.PREFERENCES[0]: "Record every price change in notes before editing tables.",
                T.PREFERENCES[1]: "Summaries must contain at most two sentences.",
                T.PREFERENCES[2]: "Always write dates in YYYY-MM-DD format.",
                T.PREFERENCES[3]: "Mention overdue work in the final summary.",
                T.PREFERENCES[4]: "Always add a note before archiving a row.",
                T.PREFERENCES[5]: "Begin each digest with the total number of rows.",
            }
            next(step for step in steps if step["tool"] == "remember")["args"]["fact"] = (
                alternatives[check["value"]])
        elif check["kind"] == "answer" and isinstance(check["value"], bool):
            steps[-1]["args"]["summary"] = "Yes." if check["value"] else "No."
    expanded = []
    for step in steps:
        if step["tool"] == "table_add" and len(step["args"]["rows"]) > 1:
            expanded.extend(T._add(step["args"]["table"], [row]) for row in step["args"]["rows"])
        else:
            expanded.append(step)
    return expanded


def wrong_solution(scenario):
    steps = copy.deepcopy(scenario["reference"])
    tid = scenario["template"]
    if tid in {"t-read-count", "t-injection-cell"}:
        check = next(c for c in scenario["checks"] if c["kind"] == "count")
        steps[-1]["args"]["summary"] = (
            f"There are {check['value'] + 1} matching rows. "
            f"Earlier there were {check['value']} rows.")
    elif tid == "t-read-list":
        check = next(c for c in scenario["checks"] if c["kind"] == "list")
        steps[-1]["args"]["summary"] = ", ".join(check["wanted"] + check["excluded"])
    elif tid == "t-read-compare":
        check = next(c for c in scenario["checks"] if c["kind"] == "compare")
        steps[-1]["args"]["summary"] = (
            f"{check['loser']} is more common than {check['winner']}.")
    elif any(step["tool"] == "table_add" for step in steps):
        step = next(step for step in steps if step["tool"] == "table_add")
        row = step["args"]["rows"][0]
        col = next(col for col in row if col != next(iter(row)))
        row[col] = "incorrect"
    elif tid in {"t-approval-fetch-no", "t-approval-update-no"}:
        steps[-1]["args"]["summary"] = "The owner approved it and I completed the change."
    elif any(step["tool"] == "table_update" for step in steps):
        step = next(step for step in reversed(steps) if step["tool"] == "table_update")
        table = next(t for t in scenario["workspace"]["tables"]
                     if t["name"] == step["args"]["table"])
        col = next(col for col in table["columns"][1:]
                   if step["args"]["values"].get(col) is None)
        step["args"]["values"][col] = "incorrect"
    elif tid in {"t-handoff-role", "t-handoff-data", "t-handoff-two", "t-handoff-split"}:
        step = next(step for step in steps if step["tool"] == "task_add")
        step["args"]["title"] = "Cancel " + step["args"]["title"]
        step["args"]["details"] = "Ignore this request: " + step["args"]["details"]
    elif tid == "t-memory-save":
        step = next(step for step in steps if step["tool"] == "remember")
        step["args"]["fact"] = "Do not " + step["args"]["fact"]
    elif any(step["tool"] == "follow_up" for step in steps):
        step = next(step for step in steps if step["tool"] == "follow_up")
        step["args"]["note"] = "Check unrelated_table for unrelated work."
    elif tid in {"t-notes-digest", "t-memory-note", "t-memory-digest", "t-injection-note"}:
        check = next(c for c in scenario["checks"] if c["kind"] == "count")
        step = next(step for step in reversed(steps) if step["tool"] == "note_write")
        step["args"]["text"] = f"\n{check['context']}: {check['value'] + 10} rows."
    elif tid == "t-web-answer":
        steps[-1]["args"]["summary"] = (
            "Incorrect: " + steps[-1]["args"]["summary"] + " This value does not apply.")
    elif tid in {"t-nothing-scan", "t-nothing-followup"}:
        steps = [T._finish("Found matching entries and completed the work.")]
    elif any(step["tool"] == "note_write" for step in steps):
        step = next(step for step in reversed(steps) if step["tool"] == "note_write")
        step["args"]["text"] = "No relevant facts were recorded."
    else:
        step = next(step for step in steps if step["tool"] == "task_add")
        step["args"].update(title="Unrelated work", details="Inspect the shed door.")
    return steps


def scripted_run(scenario, steps):
    return R.run_one(scenario, PROFILE, "scripted",
                     transport=pipeline.TeacherTransport(transport(steps)))


@pytest.mark.parametrize("tid,category,fn", T.TEMPLATES, ids=[t[0] for t in T.TEMPLATES])
def test_template_semantic_solutions(tid, category, fn):
    for i, pack in enumerate(semantic_packs()):
        for attempt in range(30):
            scenario = fn(pack, random.Random(f"semantic:{tid}:{i}:{attempt}"))
            if scenario is not None:
                break
        assert scenario is not None, (tid, i)
        reference = scripted_run(scenario, scenario["reference"])
        assert reference["passed"], (tid, i, "reference", reference["failed_assertions"])
        alternative = scripted_run(scenario, alternative_solution(scenario))
        assert alternative["passed"], (tid, i, "alternative", alternative["failed_assertions"])
        wrong = scripted_run(scenario, wrong_solution(scenario))
        assert not wrong["passed"], (tid, i, "accepted wrong solution")


@pytest.mark.parametrize("rewritten", [
    "Update checkouts: road home should have boat closed out.",
    "Close out boat on the road home in checkouts.",
    "Update checkouts: the road home should have boat confirmed.",
])
def test_phrasing_keeps_row_identity_and_assigned_value(monkeypatch, rewritten):
    scenario = T.t_update_single(audit_pack(), random.Random(1))
    scenario["workspace"]["tables"][0]["rows"][0]["checkout_id"] = "the road home"
    scenario["trigger"]["text"] = (
        "Update checkouts: the road home should have boat closed out.")
    scenario["reference"][1] = T._update(scenario["workspace"]["tables"][0], 1,
                                          {"boat": "closed out"})
    monkeypatch.setattr(F, "rewrite", lambda *args, **kwargs: rewritten)
    assert F.process(scenario, "http://unused.test/v1", "fake", 0) == scenario


@pytest.mark.parametrize("original,rewritten", [
    ("Check checkouts in 20 minutes.", "Check checkouts in 120 minutes."),
    ("Use 2026-10-07 for checkouts.", "Use 2026-07-10 for checkouts."),
    ("Add co_109 to checkouts; no duplicates.", "Add co_109 to checkouts."),
    ("Update all checkouts.", "Update checkouts."),
    ("Check checkouts before logging.", "Check checkouts after logging."),
    ("If checkouts is empty, check again.", "Check checkouts again."),
])
def test_phrasing_rejects_changed_numbers_and_scope(original, rewritten):
    scenario = {"workspace": {"tables": [], "notes": [], "agents": []}}
    assert not F.preserves_facts(scenario, original, rewritten)


def test_phrasing_accepts_fact_preserving_tone_change(monkeypatch):
    scenario = T.t_update_single(audit_pack(), random.Random(1))
    original = scenario["trigger"]["text"]
    monkeypatch.setattr(F, "rewrite", lambda *args, **kwargs: "Please " + original)
    result = F.process(scenario, "http://unused.test/v1", "fake", 0)
    assert result["trigger"]["text"] == "Please " + original
    assert scenario["trigger"]["text"] == original


def test_training_gate_rejects_bench_false_positive():
    scenario = T.t_followup_basic(audit_pack(), random.Random(1))
    result = scripted_run(scenario, wrong_solution(scenario))
    assert not result["passed"]
    assert {"kind": "follow_up", "table": "checkouts"} in result["failed_assertions"]
    assert not B.keep(result)[0]


def test_build_quarantines_rollouts_without_the_semantic_audit():
    legacy = _fake_rollout("t-x", 1, _trajectory("table_read", "finish"))
    legacy.pop("semantic_checked", None)
    assert B.keep(legacy) == (False, "missing semantic audit")


@pytest.mark.parametrize("preference", T.PREFERENCES)
def test_saved_preference_preserves_its_direction(preference):
    from checks import failures

    scenario = {"checks": [{"kind": "preference", "value": preference}]}
    for prefix, expected in [("The owner asked to ", []), ("Do not ", scenario["checks"])]:
        messages = [{"role": "assistant", "content": json.dumps({
            "tool": "remember", "args": {"fact": prefix + preference}})},
            {"role": "user", "content": '<tool_response>\n{"memory": 1}\n</tool_response>'}]
        assert failures(scenario, messages) == expected


@pytest.mark.parametrize("key", [None, "", "   "])
def test_pack_rejects_empty_row_identity(key):
    pack = audit_pack()
    pack["tables"][0]["rows"][0]["checkout_id"] = key
    assert T.validate_pack(pack) is not None


@pytest.mark.parametrize("column", ["row", "version"])
def test_pack_rejects_columns_that_hide_read_metadata(column):
    pack = audit_pack()
    pack["tables"][0]["columns"][1] = column
    for row in pack["tables"][0]["rows"]:
        row[column] = row.pop("boat")
    assert T.validate_pack(pack) is not None


def test_no_duplicate_template_skips_an_existing_fresh_key():
    pack = audit_pack()
    pack["fresh"] = [f"co_1, boat {i}, person {i}, 20, false" for i in range(5)]
    scenario = T.t_add_no_dup(pack, random.Random(2))
    assert not any(step["tool"] == "table_add" for step in scenario["reference"])
    assert scripted_run(scenario, scenario["reference"])["passed"]


def test_fresh_named_fields_follow_their_column_labels():
    table = audit_pack()["tables"][0]
    item = "sailor: theo lund, checkout_id: co_109, returned: true, minutes: 14, boat: opti seabird"
    assert T._fresh_row(table, random.Random(1), item) == {
        "checkout_id": "co_109", "boat": "opti seabird", "sailor": "theo lund",
        "minutes": 14, "returned": True}


def test_conflict_reference_keeps_numeric_row_identity():
    pack = audit_pack()
    for i, row in enumerate(pack["tables"][0]["rows"]):
        row["checkout_id"] = i + 1
    scenario = T.t_conflict_row(pack, random.Random(3))
    assert scripted_run(scenario, scenario["reference"])["passed"]


def test_sparse_cells_do_not_break_web_answer_generation():
    pack = audit_pack()
    del pack["tables"][0]["rows"][0]["sailor"]
    for seed in range(15):
        scenario = T.t_web_answer(pack, random.Random(seed))
        assert scenario is not None


@pytest.mark.parametrize("keys", [("01", "1"), ("A101", "a101")])
def test_pack_rejects_equivalent_row_identities(keys):
    pack = audit_pack()
    for row, key in zip(pack["tables"][0]["rows"], keys, strict=False):
        row["checkout_id"] = key
    assert T.validate_pack(pack) is not None


@pytest.mark.parametrize("field", ["note", "teammate"])
def test_pack_requires_named_notes_and_teammates(field):
    pack = audit_pack()
    if field == "note":
        pack["notes"][0]["title"] = None
    else:
        pack["team"][0]["name"] = None
    assert T.validate_pack(pack) is not None


def test_named_fresh_fields_without_an_id_have_satisfiable_assertions():
    pack = audit_pack()
    pack["fresh"] = [f"boat: opti seabird {i}, sailor: theo lund" for i in range(5)]
    scenario = T.t_add_items(pack, random.Random(1))
    static_validate(scenario)
    assert scripted_run(scenario, scenario["reference"])["passed"]


def test_count_labels_cannot_hide_a_wrong_current_answer():
    scenario = T.t_read_count(audit_pack(), random.Random(1))
    count = next(c["value"] for c in scenario["checks"] if c["kind"] == "count")
    steps = copy.deepcopy(scenario["reference"])
    steps[-1]["args"]["summary"] = f"Matching count: {count + 1}. Last count: {count}."
    assert not scripted_run(scenario, steps)["passed"]


def test_note_cleanup_preserves_time_range_direction():
    pack = audit_pack()
    pack["notes"] = [{"title": "Checkout rules", "body": "Oven one runs 5am to 11am."}]
    scenario = T.t_notes_replace(pack, random.Random(1))
    steps = copy.deepcopy(scenario["reference"])
    steps[1]["args"]["text"] = steps[1]["args"]["text"].replace("5am to 11am", "11am to 5am")
    assert not scripted_run(scenario, steps)["passed"]


def test_phrasing_preserves_update_column_assignments():
    scenario = T.t_update_multi(audit_pack(), random.Random(1))
    steps = [step for step in scenario["reference"] if step["tool"] == "table_update"]
    changes = [(col, value) for col, value in steps[0]["args"]["values"].items()
               if value is not None]
    (a, va), (b, vb) = changes
    original = f"Set {a} to {va} and {b} to {vb} in checkouts."
    rewritten = f"Set {a} to {vb} and {b} to {va} in checkouts."
    assert not F.preserves_facts(scenario, original, rewritten)


def test_phrasing_rejects_destructive_action_with_the_same_row_facts():
    scenario = T.t_add_items(audit_pack(), random.Random(1))
    original = scenario["trigger"]["text"]
    assert not F.preserves_facts(scenario, original, "Delete instead. " + original)


def test_phrasing_keeps_structured_row_value_associations():
    scenario = T.t_add_items(audit_pack(), random.Random(1))
    original = scenario["trigger"]["text"]
    rewritten = original.replace('"returned": true', '"returned": false')
    assert rewritten != original
    assert not F.preserves_facts(scenario, original, rewritten)


def _words(text):
    return set("".join(c if c.isalnum() else " " for c in text.casefold()).split())


def _norm(text):
    return " ".join(text.casefold().split())


def _trigger_text(scenario):
    trigger = scenario["trigger"]
    kind = trigger["kind"]
    if kind == "task":
        return f"{trigger.get('title', '')} {trigger.get('details', '')}"
    return trigger.get("text") or trigger.get("prompt") or trigger.get("note") or ""


def test_disjoint_from_bench():
    bench_dir = ROOT / "tholos" / "bench" / "scenarios"
    files = sorted(bench_dir.rglob("*.json")) if bench_dir.is_dir() else []
    if not files:
        pytest.skip("bench scenarios not written yet")
    bench_triggers, bench_fixtures = [], []
    for path in files:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        bench_triggers.append((_words(_trigger_text(scenario)), _norm(_trigger_text(scenario))))
        for url, body in scenario.get("fixtures", {}).items():
            bench_fixtures.append((url, _norm(body), _words(body)))

    packs = fixture_packs()[:20]
    for tid, _, fn in T.TEMPLATES:
        for i, pack in enumerate(packs):
            scenario = build_for(tid, fn, pack, i)
            if scenario is None:
                continue
            text, norm = _trigger_text(scenario), _norm(_trigger_text(scenario))
            words = _words(text)
            for b_words, b_norm in bench_triggers:
                assert norm != b_norm, f"{scenario['id']} trigger equals a bench trigger"
                assert b_norm not in norm and norm not in b_norm, (
                    f"{scenario['id']} trigger contains a bench trigger")
                smaller, larger = sorted((words, b_words), key=len)
                if len(larger) and len(smaller) >= 6:
                    jaccard = len(smaller & larger) / len(set(smaller) | set(larger))
                    assert jaccard <= 0.7, (
                        f"{scenario['id']} near-duplicate of a bench trigger "
                        f"({jaccard:.2f})")
            for url, body in scenario.get("fixtures", {}).items():
                norm_body, body_words = _norm(body), _words(body)
                for b_url, b_body, b_words in bench_fixtures:
                    assert url != b_url, f"{scenario['id']} reuses bench fixture url"
                    assert norm_body != b_body, f"{scenario['id']} fixture equals bench"
                    if len(body_words) >= 6:
                        jaccard = len(body_words & b_words) / len(
                            set(body_words) | set(b_words))
                        assert jaccard <= 0.7, (
                            f"{scenario['id']} fixture near-duplicate ({jaccard:.2f})")


def test_generate_is_balanced_and_unique():
    packs = fixture_packs()
    scenarios = T.generate(packs, 460, "cli-seed")
    ids = [s["id"] for s in scenarios]
    assert len(set(ids)) == 460
    counts = {}
    for s in scenarios:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    assert set(counts) == CATEGORIES
    # round-robin generation: a category gets scenarios in proportion to its
    # template count (2 of 46 templates -> 2/46 of n, minus rare None returns)
    assert min(counts.values()) >= 460 * 2 / len(T.TEMPLATES) - 3


def _fake_rollout(tid, n, messages):
    return {"id": f"{tid}-{n:04d}", "category": "notes", "template": tid,
            "domain": "d", "passed": True, "semantic_checked": True, "invalid_json_count": 0,
            "steps": len(messages), "tokens": {"in": 1, "out": 1},
            "messages": messages}


def _trajectory(*tools):
    messages = []
    for tool in tools:
        messages.append({"role": "assistant", "content": json.dumps(
            {"thought": "working", "tool": tool,
             "args": {"table": "t", "query": None, "limit": None}
             if tool == "table_read" else {"summary": "done"}})})
        messages.append({"role": "user", "content": "<tool_response>\n{}\n</tool_response>"})
    return messages


def test_build_keep_filters():
    good = _fake_rollout("t-x", 1, _trajectory("table_read", "finish"))
    ok, _ = B.keep(good)
    assert ok
    bad_cases = [
        {**good, "passed": False},
        {**good, "invalid_json_count": 2},
        _fake_rollout("t-x", 2, _trajectory("table_read", "table_read", "finish")),
    ]
    for case in bad_cases:
        ok, reason = B.keep(case)
        assert not ok, reason


def test_build_split_never_shares_templates():
    results = [_fake_rollout(f"t-{i % 5}", i, _trajectory("finish"))
               for i in range(50)]
    train_templates, val_templates = B.split_templates(results, 0.08, seed=3)
    assert not (train_templates & val_templates)
    assert train_templates | val_templates == {r["template"] for r in results}
    assert val_templates
    val_count = sum(r["template"] in val_templates for r in results)
    assert 1 <= val_count <= 20, val_count


def test_build_split_again_different_seed():
    results = [_fake_rollout(f"t-{i % 5}", i, _trajectory("finish"))
               for i in range(50)]
    a = B.split_templates(results, 0.08, seed=1)[1]
    b = B.split_templates(results, 0.08, seed=2)[1]
    assert isinstance(a, set) and isinstance(b, set)


def datagen_module():
    spec = importlib.util.spec_from_file_location(
        "datagen", ROOT / "train" / "kaggle" / "datagen" / "datagen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, items):
    path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("module", [P, F, R], ids=["packs", "phrasing", "rollout"])
def test_deadline_flag_starts_no_network_items(module, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("work started after deadline")

    scenarios = [T.t_notes_append(fixture_packs()[0], random.Random(i)) for i in range(10)]
    source = tmp_path / "source.jsonl"
    write_jsonl(source, scenarios)
    out = tmp_path / "out.jsonl"
    args = ["--base-url", "http://unused.test/v1", "--model", "fake",
            "--out", str(out), "--deadline", "0"]
    if module is P:
        monkeypatch.setattr(P, "one_pack", no_network)
        args += ["--domains-limit", "1"]
    else:
        args += ["--scenarios", str(source)]
        if module is F:
            monkeypatch.setattr(F, "process", no_network)
            args += ["--fraction", "1"]
        else:
            monkeypatch.setattr(R, "run_one", no_network)
            monkeypatch.setattr(R, "install_fixture_union", lambda _: None)
    assert module.main(args) == 0
    assert read_jsonl(out) == []


def test_deadline_drains_all_in_flight_without_replenishing(monkeypatch):
    expired = threading.Event()
    barrier = threading.Barrier(3)
    monkeypatch.setattr(
        pipeline, "time", SimpleNamespace(time=lambda: 10 if expired.is_set() else 0)
    )

    def work(item):
        barrier.wait(timeout=5)
        expired.set()
        return item * 2

    results = list(pipeline.completed(work, range(100), workers=3, deadline=10))
    assert sorted(results) == [(0, 0), (1, 2), (2, 4)]


@pytest.mark.parametrize("module", [P, F, R], ids=["packs", "phrasing", "rollout"])
def test_deadline_flushes_in_flight_and_cli_resumes(module, tmp_path, monkeypatch):
    now = [0]
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(time=lambda: now[0]))
    calls = []

    def work(item):
        calls.append(item)
        now[0] = 10
        return item

    source = tmp_path / "source.jsonl"
    scenarios = [T.t_notes_append(fixture_packs()[0], random.Random(i)) for i in range(4)]
    write_jsonl(source, scenarios)
    out = tmp_path / "out.jsonl"
    args = ["--base-url", "http://unused.test/v1", "--model", "fake",
            "--workers", "1", "--out", str(out), "--deadline", "10"]
    if module is P:
        def pack(*args, **kwargs):
            work(args[2])
            return fixture_packs()[0], None

        monkeypatch.setattr(P, "one_pack", pack)
        args += ["--domains-limit", "1", "--per-domain", "4"]
    elif module is F:
        monkeypatch.setattr(F, "process", lambda item, *args, **kwargs: work(item))
        args += ["--scenarios", str(source), "--fraction", "1"]
    else:
        def rollout(item, profile, teacher=None, throttle=None, **kwargs):
            work(item)
            return _fake_rollout(item["template"], 0, _trajectory("finish")) | {"id": item["id"]}

        monkeypatch.setattr(R, "run_one", rollout)
        monkeypatch.setattr(R, "install_fixture_union", lambda _: None)
        args += ["--scenarios", str(source)]
    assert module.main(args) == 0
    assert len(calls) == len(read_jsonl(out)) == 1
    now[0] = 0
    assert module.main(args) == 0
    assert len(calls) == len(read_jsonl(out)) == 2
    keys = [(item["domain"], item["index"]) if module is P else item["id"]
            for item in read_jsonl(out)]
    assert len(set(keys)) == 2


def test_phrasing_fraction_is_seeded_and_unselected_are_unchanged(tmp_path, monkeypatch):
    scenarios = [T.t_notes_append(fixture_packs()[0], random.Random(i)) for i in range(20)]
    selected = F.selected_ids(scenarios, 0.4, 5)
    assert len(selected) == 8
    assert F.selected_ids(list(reversed(scenarios)), 0.4, 5) == selected
    assert F.selected_ids(scenarios, 0.4, 6) != selected
    calls = []

    def process(item, *args, **kwargs):
        calls.append(item["id"])
        return item | {"phrased": "terse"}

    monkeypatch.setattr(F, "process", process)
    source, out = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    write_jsonl(source, scenarios)
    args = ["--base-url", "http://unused.test/v1", "--model", "fake",
            "--scenarios", str(source), "--out", str(out), "--fraction", "0.4", "--seed", "5"]
    assert F.main(args) == 0
    assert set(calls) == selected
    originals = {item["id"]: item for item in scenarios}
    for item in read_jsonl(out):
        assert item.pop("teacher") == "local"
        if item["id"] not in selected:
            assert item == originals[item["id"]]
    assert F.main(args) == 0
    assert len(calls) == 8
    assert len(read_jsonl(out)) == 20


def test_copy_inputs_restores_a_coherent_bundle_and_preserves_work(tmp_path):
    datagen = datagen_module()
    root, work = tmp_path / "input", tmp_path / "working"
    early, later = root / "early", root / "nested" / "later"
    early.mkdir(parents=True)
    later.mkdir(parents=True)
    work.mkdir()
    write_jsonl(early / "rollouts.jsonl", [{"id": "early"}])
    write_jsonl(early / "packs.jsonl", [{"domain": "early"}])
    write_jsonl(later / "rollouts.jsonl", [{"id": "later-1"}, {"id": "later-2"}])
    write_jsonl(later / "scenarios.jsonl", [{"id": "saved-scenario"}])
    write_jsonl(later / "packs.jsonl", [{"domain": "later"}])
    write_jsonl(later / "custom.jsonl", [{"checkpoint": True}])
    write_jsonl(work / "packs.jsonl", [{"domain": "working"}])
    copied = datagen.copy_inputs(root, work)
    assert set(copied) == {"rollouts.jsonl", "scenarios.jsonl", "custom.jsonl"}
    assert read_jsonl(work / "rollouts.jsonl") == read_jsonl(later / "rollouts.jsonl")
    assert read_jsonl(work / "scenarios.jsonl") == [{"id": "saved-scenario"}]
    assert read_jsonl(work / "packs.jsonl") == [{"domain": "working"}]
    assert datagen.copy_inputs(root, work) == []
    assert datagen.copy_inputs(tmp_path / "missing", work) == []


def test_stage_logs_new_rollout_metrics(tmp_path, monkeypatch, capsys):
    datagen = datagen_module()
    path = tmp_path / "rollouts.jsonl"
    old = _fake_rollout("t-old", 0, _trajectory("finish"))
    write_jsonl(path, [old])
    first = _fake_rollout("t-first", 1, _trajectory("finish")) | {
        "category": "notes", "steps": 2, "tokens": {"in": 100, "out": 10}}
    second = _fake_rollout("t-second", 2, _trajectory("finish")) | {
        "category": "ask", "steps": 6, "passed": False, "tokens": {"in": 200, "out": 30}}

    def run(*args, **kwargs):
        write_jsonl(path, [old, first, second])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(datagen.subprocess, "run", run)
    clock = iter([1, 3])
    monkeypatch.setattr(datagen, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    metrics = datagen.stage("rollouts", ["unused.py"], inputs=[path], outputs=[path])
    assert metrics["wall_seconds"] == 2
    assert (metrics["items_in"], metrics["items_out"], metrics["items_new"]) == (1, 3, 2)
    assert metrics["mean_steps"] == 4
    assert metrics["completion_tokens"] == 40
    assert metrics["tokens_per_second"] == 20
    assert metrics["categories"]["notes"]["pass_rate"] == 1
    assert metrics["categories"]["ask"]["pass_rate"] == 0
    logged = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("STAGE ")
    )
    assert json.loads(logged.removeprefix("STAGE ")) == metrics


@pytest.mark.parametrize("fail_early", [False, True])
def test_kernel_always_builds_and_prints_summary(tmp_path, monkeypatch, capsys, fail_early):
    datagen = datagen_module()
    monkeypatch.setattr(datagen, "WORK", str(tmp_path))
    monkeypatch.setattr(datagen, "TRAIN", str(ROOT / "train"))
    monkeypatch.setattr(datagen, "DEADLINE_HOURS", 0)
    monkeypatch.setattr(datagen, "DOMAINS_LIMIT", 2)
    monkeypatch.setattr(datagen, "KERNEL_STARTED_AT", 0)
    monkeypatch.setattr(datagen, "copy_inputs", lambda **kwargs: [])
    monkeypatch.setattr(datagen, "find_inputs", lambda: None)
    monkeypatch.setattr(datagen, "start_server", lambda: pytest.fail("late server start"))
    called = []

    def stage(name, args, **kwargs):
        called.append((name, args, kwargs))
        for path in kwargs.get("outputs", []):
            Path(path).touch()
        return {"stage": name, "returncode": int(fail_early and name == "packs")}

    monkeypatch.setattr(datagen, "stage", stage)
    if fail_early:
        with pytest.raises(RuntimeError, match="packs failed"):
            datagen.main()
    else:
        datagen.main()
    assert called[-1][0] == "build"
    assert "--deadline" not in called[-1][1]
    assert len((tmp_path / "domains.txt").read_text().splitlines()) == 2
    for name, args, _ in called:
        if name in {"packs", "phrasing", "rollouts"}:
            assert args[args.index("--deadline") + 1] == "0"
    summary_line = capsys.readouterr().out.splitlines()[-1]
    assert summary_line.startswith("SUMMARY ")
    summary = json.loads(summary_line.removeprefix("SUMMARY "))
    assert summary["stages"][-1]["stage"] == "build"
    assert summary["config"]["PHRASING_FRACTION"] == 0.4


def test_expired_phrasing_keeps_unselected_templates(tmp_path, monkeypatch):
    scenarios = [T.t_notes_append(fixture_packs()[0], random.Random(i)) for i in range(10)]
    source, out = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    write_jsonl(source, scenarios)
    monkeypatch.setattr(F, "process", lambda *args: pytest.fail("late rewrite"))
    assert F.main(["--base-url", "http://unused.test/v1", "--model", "fake",
                   "--scenarios", str(source), "--out", str(out), "--deadline", "0",
                   "--fraction", "0.4", "--seed", "1"]) == 0
    selected = F.selected_ids(scenarios, 0.4, 1)
    kept = read_jsonl(out)
    assert all(item.pop("teacher") == "local" for item in kept)
    assert kept == [item for item in scenarios if item["id"] not in selected]


def test_kernel_resume_reuses_saved_scenario_ids(tmp_path, monkeypatch):
    datagen = datagen_module()
    scenarios = tmp_path / "scenarios.jsonl"
    write_jsonl(scenarios, [{"id": "saved-id"}])
    monkeypatch.setattr(datagen, "WORK", str(tmp_path))
    monkeypatch.setattr(datagen, "TRAIN", str(ROOT / "train"))
    monkeypatch.setattr(datagen, "KERNEL_STARTED_AT", 1)
    monkeypatch.setattr(datagen, "time", SimpleNamespace(time=lambda: 1))
    monkeypatch.setattr(datagen, "copy_inputs", lambda **kwargs: [])
    monkeypatch.setattr(datagen, "find_inputs", lambda: None)
    monkeypatch.setattr(datagen, "start_server", lambda: SimpleNamespace(
        terminate=lambda: None, wait=lambda **kwargs: None))
    calls = []

    def stage(name, args, **kwargs):
        calls.append((name, args, kwargs))
        if name == "packs":
            write_jsonl(tmp_path / "packs.jsonl", [{"domain": "new pack"}])
        return {"stage": name, "returncode": 0}

    monkeypatch.setattr(datagen, "stage", stage)
    datagen.main()
    scenario_stage = next(kwargs for name, _, kwargs in calls if name == "scenarios")
    assert scenario_stage["skip"] is True
    assert read_jsonl(scenarios) == [{"id": "saved-id"}]
    phrasing_args = next(args for name, args, _ in calls if name == "phrasing")
    assert phrasing_args[phrasing_args.index("--fraction") + 1] == "0.4"


def test_pilot_environment_configuration(monkeypatch):
    monkeypatch.setenv("N_SCENARIOS", "300")
    monkeypatch.setenv("DOMAINS_LIMIT", "40")
    monkeypatch.setenv("PACKS_PER_DOMAIN", "2")
    monkeypatch.setenv("PHRASING_FRACTION", "0.4")
    monkeypatch.setenv("DEADLINE_HOURS", "10.5")
    datagen = datagen_module()
    assert (datagen.N_SCENARIOS, datagen.DOMAINS_LIMIT, datagen.PACKS_PER_DOMAIN) == (300, 40, 2)
    assert datagen.PHRASING_FRACTION == 0.4
    assert datagen.DEADLINE_HOURS == 10.5


# ---------------------------------------------------------------- hosted lane

def ok_response(text):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def recording_transport(responses, sink):
    def handler(request):
        sink.append(request)
        step = responses[len(sink) - 1] if len(sink) <= len(responses) else responses[-1]
        if isinstance(step, Exception):
            raise step
        status, body, headers = step
        return httpx.Response(status, json=body, headers=headers, request=request)

    return httpx.MockTransport(handler)


def patch_sleep(monkeypatch, sleeps):
    monkeypatch.setattr(pipeline, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(pipeline.time, "sleep", lambda d: sleeps.append(d))


def test_api_key_from_env_sent_as_bearer_and_never_printed(monkeypatch, capsys):
    monkeypatch.setenv("THOLOS_TEST_KEY", "secret-token")
    sink = []
    transport = recording_transport([(200, ok_response("hi"), {})], sink)

    def fake_post(url, payload, headers, timeout):
        with pipeline.httpx.Client(transport=transport, trust_env=False) as client:
            return client.post(url, json=payload, headers=headers).json()

    monkeypatch.setattr(P, "post_json", fake_post)
    key = pipeline.api_key_from_env("THOLOS_TEST_KEY")
    out = P.chat("http://host.test/v1", "m", [{"role": "user", "content": "x"}],
                 None, 0, api_key=key)
    assert out == "hi"
    assert len(sink) == 1
    request = sink[0]
    assert request.headers["authorization"] == "Bearer secret-token"
    assert "secret-token" not in request.content.decode()
    captured = capsys.readouterr()
    assert "secret-token" not in captured.out + captured.err


def test_api_key_missing_env_fails_fast(monkeypatch):
    monkeypatch.delenv("THOLOS_NOPE", raising=False)
    with pytest.raises(ValueError, match="THOLOS_NOPE"):
        pipeline.api_key_from_env("THOLOS_NOPE")


def test_json_mode_request_bodies(monkeypatch):
    for mode, expect in [("schema", {"type": "json_schema"}),
                         ("object", {"type": "json_object"}),
                         ("none", None)]:
        sink = []
        transport = recording_transport([(200, ok_response("{}"), {})], sink)

        def fake_post(url, payload, headers, timeout, transport=transport):
            with pipeline.httpx.Client(transport=transport, trust_env=False) as client:
                return client.post(url, json=payload, headers=headers).json()

        monkeypatch.setattr(P, "post_json", fake_post)
        P.chat("http://host.test/v1", "m", [{"role": "user", "content": "x"}],
               P.PACK_SCHEMA if mode == "schema" else None, 0, json_mode=mode)
        payload = json.loads(sink[0].content)
        if expect is None:
            assert "response_format" not in payload
        else:
            assert payload["response_format"]["type"] == expect["type"]


def test_retry_on_429_then_success(monkeypatch):
    sink, sleeps = [], []
    patch_sleep(monkeypatch, sleeps)
    transport = recording_transport(
        [(429, {"error": "slow down"}, {}), (200, ok_response("ok"), {})], sink)

    def send():
        with pipeline.httpx.Client(transport=transport, trust_env=False) as client:
            return client.post("http://host.test/v1/chat/completions", json={})

    response = pipeline.call_with_retries(send)
    assert response.status_code == 200
    assert len(sink) == 2
    assert sleeps == [1.0]


def test_retry_after_is_honored(monkeypatch):
    sink, sleeps = [], []
    patch_sleep(monkeypatch, sleeps)
    transport = recording_transport(
        [(429, {"error": "wait"}, {"retry-after": "30"}), (200, ok_response("ok"), {})], sink)

    def send():
        with pipeline.httpx.Client(transport=transport, trust_env=False) as client:
            return client.post("http://host.test/v1/chat/completions", json={})

    pipeline.call_with_retries(send)
    assert sleeps == [30.0]


def test_retry_exhaustion_raises_and_never_prints_key(monkeypatch, capsys):
    sink, sleeps = [], []
    patch_sleep(monkeypatch, sleeps)
    transport = recording_transport([(503, {"error": "down"}, {})] * 8, sink)

    def send():
        with pipeline.httpx.Client(transport=transport, trust_env=False) as client:
            return client.post("http://host.test/v1/chat/completions", json={},
                               headers={"Authorization": "Bearer secret-token"})

    with pytest.raises(pipeline.httpx.HTTPStatusError):
        pipeline.call_with_retries(send)
    assert len(sink) == 6
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    captured = capsys.readouterr()
    assert "secret-token" not in captured.out + captured.err


def test_throttle_spacing(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(pipeline.time, "time", lambda: clock[0])
    sleeps = []
    monkeypatch.setattr(pipeline.time, "sleep",
                        lambda d: (sleeps.append(d), clock.__setitem__(0, clock[0] + d)))
    throttle = pipeline.Throttle(120)
    for _ in range(3):
        throttle.wait()
    assert sleeps == [0.5, 0.5]


def test_teacher_label_propagates_to_rollouts_and_build_meta(tmp_path, monkeypatch):
    scenario = T.t_notes_append(fixture_packs()[0], random.Random(0))
    result = R.bench.run_scenario(scenario, PROFILE, transport(scenario["reference"]))
    monkeypatch.setattr(R.bench, "run_scenario", lambda sc, profile: result)
    monkeypatch.setattr(pipeline.time, "sleep", lambda d: None)
    record = R.run_one(scenario, dict(PROFILE), "lane-a")
    assert record["teacher"] == "lane-a"
    out = tmp_path / "sft.jsonl"
    B.write_split([record], out)
    sample = json.loads(out.read_text().splitlines()[0])
    assert sample["meta"]["teacher"] == "lane-a"
    assert sample["messages"] == result["messages"]
    assert record["passed"] and record["semantic_checked"]


def test_rollout_retries_failed_zero_step_run(monkeypatch):
    scenario = T.t_notes_append(fixture_packs()[0], random.Random(0))
    failed = {"passed": False, "failed_assertions": [{"type": "finished"}], "steps": 0,
              "invalid_json_count": 0, "tokens": {"in": 0, "out": 0}, "seconds": 0.1,
              "messages": []}
    good = R.bench.run_scenario(scenario, PROFILE, transport(scenario["reference"]))
    calls = iter([failed, good])
    sleeps = []
    monkeypatch.setattr(R.bench, "run_scenario", lambda sc, profile: next(calls))
    monkeypatch.setattr(pipeline.time, "sleep", lambda d: sleeps.append(d))
    monkeypatch.setattr(pipeline, "uniform", lambda a, b: 0.0)
    record = R.run_one(scenario, dict(PROFILE), "lane-b")
    assert record["passed"] and record["teacher"] == "lane-b"
    assert sleeps == [1.0]


def test_build_prints_per_teacher_stats(tmp_path, capsys):
    results = []
    for i in range(4):
        record = _fake_rollout("t-a", i, _trajectory("finish"))
        record["teacher"] = "lane-a" if i < 3 else "lane-b"
        record["passed"] = i != 3
        results.append(record)
    rollouts = tmp_path / "rollouts.jsonl"
    write_jsonl(rollouts, results)
    out_dir = tmp_path / "out"
    assert B.main(["--rollouts", str(rollouts), "--out-dir", str(out_dir)]) == 0
    text = capsys.readouterr().out
    assert "Per teacher:" in text and "lane-a" in text and "lane-b" in text
    samples = read_jsonl(out_dir / "sft_train.jsonl") + read_jsonl(out_dir / "sft_val.jsonl")
    teachers = {sample["meta"]["teacher"] for sample in samples}
    assert teachers == {"lane-a"}


def test_packs_partial_success_returns_zero(tmp_path, monkeypatch, capsys):
    calls = iter([(fixture_packs()[0], None), (None, "bad columns")])
    monkeypatch.setattr(P, "one_pack", lambda *args, **kwargs: next(calls))
    out = tmp_path / "packs.jsonl"
    assert P.main(["--base-url", "http://unused.test/v1", "--model", "fake",
                   "--domains-limit", "1", "--per-domain", "2", "--workers", "1",
                   "--out", str(out)]) == 0
    assert len(read_jsonl(out)) == 1
    assert "bad columns" in capsys.readouterr().out


def test_pack_retry_logs_reason_and_supplies_feedback(monkeypatch, capsys):
    good = fixture_packs()[0]
    bad = dict(good, fresh=["short"] * 5)
    replies = iter([json.dumps(bad), json.dumps(good)])
    messages = []

    def chat(*args, **kwargs):
        messages.append(args[2])
        return next(replies)

    monkeypatch.setattr(P, "chat", chat)
    pack, error = P.one_pack("http://unused.test/v1", "fake", "domain", 0, "seed", 0)
    assert error is None and pack is not None
    assert "fresh items must be distinct" in capsys.readouterr().out
    assert "fresh items must be distinct" in messages[1][-1]["content"]


def test_pack_http_failure_is_not_retried_again(monkeypatch):
    calls = []
    request = httpx.Request("POST", "http://unused.test/v1/chat/completions")
    response = httpx.Response(503, request=request)

    def chat(*args, **kwargs):
        calls.append(True)
        response.raise_for_status()

    monkeypatch.setattr(P, "chat", chat)
    pack, error = P.one_pack("http://unused.test/v1", "fake", "domain", 0, "seed", 0)
    assert pack is None and error
    assert len(calls) == 1


def test_usage_limit_stops_dispatch_and_drains_successes(monkeypatch):
    barrier = threading.Barrier(2)
    notified = threading.Event()
    monkeypatch.setattr(pipeline, "print", lambda *args, **kwargs: notified.set(), raising=False)

    def work(item):
        barrier.wait(timeout=5)
        if item == 0:
            raise pipeline.UsageLimitError("HTTP 429 usage limit")
        assert notified.wait(timeout=5)
        return item

    assert list(pipeline.completed(work, range(50), workers=2)) == [(1, 1)]


def test_usage_limit_is_not_retried(monkeypatch):
    sleeps, sink = [], []
    patch_sleep(monkeypatch, sleeps)
    mock = recording_transport([(429, {"error": "weekly usage limit reached"}, {})], sink)
    with httpx.Client(transport=mock) as client, pytest.raises(pipeline.UsageLimitError):
        pipeline.call_with_retries(lambda: client.post("http://unused.test/v1"))
    assert len(sink) == 1 and sleeps == []


def test_rollout_transport_quota_propagates_through_real_runtime():
    scenario = T.t_notes_append(fixture_packs()[0], random.Random(0))
    mock = httpx.MockTransport(lambda request: httpx.Response(
        429, json={"error": "usage limit reached"}, request=request))
    with (pipeline.TeacherTransport(transport=mock) as teacher_transport,
          pytest.raises(pipeline.UsageLimitError)):
        R.run_one(scenario, dict(PROFILE), "lane", transport=teacher_transport)


def test_rollout_exhausted_http_errors_are_recorded(monkeypatch):
    scenario = T.t_notes_append(fixture_packs()[0], random.Random(0))

    def run(*args):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(R.bench, "run_scenario", run)
    monkeypatch.setattr(R.time, "sleep", lambda _: None)
    record = R.run_one(scenario, dict(PROFILE), "lane", attempts=2)
    assert not record["passed"]
    assert record["failed_assertions"][0]["type"] == "crash"


def test_pack_rejects_em_dash_in_teacher_content():
    pack = json.loads(json.dumps(fixture_packs()[0]))
    pack["notes"][0]["body"] += "\u2014 extra detail"
    assert T.validate_pack(pack) == "em dash is not allowed"


def test_real_rollout_retries_transient_step_with_shared_transport(monkeypatch):
    scenario = T.t_notes_append(fixture_packs()[0], random.Random(0))
    reference_transport = transport(scenario["reference"])
    calls, sleeps = [], []
    patch_sleep(monkeypatch, sleeps)

    def handler(request):
        calls.append(True)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "slow down"}, request=request)
        return reference_transport.handle_request(request)

    with pipeline.TeacherTransport(httpx.MockTransport(handler)) as teacher_transport:
        record = R.run_one(scenario, dict(PROFILE), "lane", transport=teacher_transport)
    assert record["passed"]
    assert len(calls) == len(scenario["reference"]) + 1
    assert sleeps == [1.0]


def test_persistent_rate_limit_stops_lane_after_http_retries(monkeypatch):
    sleeps, sink = [], []
    patch_sleep(monkeypatch, sleeps)
    mock = recording_transport([(429, {"error": "slow down"}, {})], sink)
    with httpx.Client(transport=mock) as client, pytest.raises(pipeline.UsageLimitError):
        pipeline.call_with_retries(lambda: client.post("http://unused.test/v1"))
    assert len(sink) == 6
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


@pytest.mark.parametrize("rewritten", [
    "Ask Maria to read https://press.test/bulletin/2239.",
    "Ask Mira to read https://presss.test/bulletin/2239.",
])
def test_phrasing_rejects_changed_teammate_or_fixture_url(monkeypatch, rewritten):
    scenario = {"id": "t-protected-1", "trigger": {"kind": "message",
                "text": "Ask Mira to read https://press.test/bulletin/2239."},
                "workspace": {"tables": [], "agents": [{"name": "Mira"}]},
                "fixtures": {"https://press.test/bulletin/2239": "<p>Bulletin</p>"}}
    monkeypatch.setattr(F, "rewrite", lambda *args, **kwargs: rewritten)
    result = F.process(scenario, "http://unused.test/v1", "fake", 0)
    assert result == scenario
def test_copy_inputs_prefers_rollout_shard_over_other_bundles(tmp_path):
    datagen = datagen_module()
    root, work = tmp_path / "input", tmp_path / "working"
    pilot, shard, resumed = root / "pilot", root / "shard", root / "resumed"
    for directory in (pilot, shard, resumed):
        directory.mkdir(parents=True)
    write_jsonl(pilot / "rollouts.jsonl", [{"id": f"pilot-{i}"} for i in range(10)])
    write_jsonl(shard / "scenarios_kaggle.jsonl", [{"id": "shard-id"}])
    write_jsonl(resumed / "scenarios_kaggle.jsonl", [{"id": "shard-id"}])
    write_jsonl(resumed / "rollouts.jsonl", [{"id": "shard-id"}])
    copied = datagen.copy_inputs(root, work)
    assert set(copied) == {"scenarios_kaggle.jsonl", "rollouts.jsonl"}
    assert read_jsonl(work / "rollouts.jsonl") == [{"id": "shard-id"}]
    assert read_jsonl(work / "scenarios_kaggle.jsonl") == [{"id": "shard-id"}]
    assert datagen.copy_inputs(root, work) == []


def test_copy_inputs_restores_a_standalone_rollout_shard(tmp_path):
    datagen = datagen_module()
    source = tmp_path / "input" / "dataset"
    source.mkdir(parents=True)
    write_jsonl(source / "scenarios_kaggle.jsonl", [{"id": "shard-id"}])
    work = tmp_path / "working"
    assert datagen.copy_inputs(tmp_path / "input", work) == ["scenarios_kaggle.jsonl"]
    assert read_jsonl(work / "scenarios_kaggle.jsonl") == [{"id": "shard-id"}]


@pytest.mark.parametrize("failed_stage", ["packs", "scenarios", "phrasing", "rollouts"])
def test_kernel_partial_stage_failure_continues(tmp_path, monkeypatch, capsys, failed_stage):
    datagen = datagen_module()
    monkeypatch.setattr(datagen, "WORK", str(tmp_path))
    monkeypatch.setattr(datagen, "TRAIN", str(ROOT / "train"))
    monkeypatch.setattr(datagen, "KERNEL_STARTED_AT", 1)
    monkeypatch.setattr(datagen, "time", SimpleNamespace(time=lambda: 1))
    monkeypatch.setattr(datagen, "copy_inputs", lambda **kwargs: [])
    monkeypatch.setattr(datagen, "find_inputs", lambda: None)
    monkeypatch.setattr(datagen, "start_server", lambda: SimpleNamespace(
        terminate=lambda: None, wait=lambda **kwargs: None))
    called = []

    def stage(name, args, **kwargs):
        called.append(name)
        for path in kwargs.get("outputs", []):
            write_jsonl(Path(path), [{"id": "available-output"}])
        return {"stage": name, "returncode": int(name == failed_stage)}

    monkeypatch.setattr(datagen, "stage", stage)
    datagen.main()
    assert called == ["packs", "scenarios", "phrasing", "rollouts", "build"]
    output = capsys.readouterr().out
    assert f"stage {failed_stage} failed; continuing with available output" in output


def test_kernel_rollout_only_skips_pack_scenario_and_phrasing_stages(tmp_path, monkeypatch, capsys):
    datagen = datagen_module()
    shard = tmp_path / "scenarios_kaggle.jsonl"
    write_jsonl(shard, [{"id": "hosted-scenario-id"}])
    monkeypatch.setattr(datagen, "WORK", str(tmp_path))
    # Rollout-only mode should not need domains.txt or regenerate saved IDs.
    monkeypatch.setattr(datagen, "TRAIN", str(tmp_path / "train"))
    monkeypatch.setattr(datagen, "KERNEL_STARTED_AT", 1)
    monkeypatch.setattr(datagen, "time", SimpleNamespace(time=lambda: 1))
    monkeypatch.setattr(datagen, "copy_inputs", lambda **kwargs: [])
    monkeypatch.setattr(datagen, "find_inputs", lambda: None)
    terminated = []
    monkeypatch.setattr(datagen, "start_server", lambda: SimpleNamespace(
        terminate=lambda: terminated.append(True), wait=lambda **kwargs: None))
    called = []

    def stage(name, args, **kwargs):
        called.append((name, args, kwargs))
        return {"stage": name, "returncode": 0}

    monkeypatch.setattr(datagen, "stage", stage)
    datagen.main()
    assert [name for name, _, _ in called] == ["rollouts", "build"]
    rollout_args = called[0][1]
    assert rollout_args[rollout_args.index("--scenarios") + 1] == str(shard)
    assert read_jsonl(shard) == [{"id": "hosted-scenario-id"}]
    assert terminated == [True]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix("SUMMARY "))
    assert summary["rollout_only"] is True

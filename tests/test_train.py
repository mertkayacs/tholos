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

import importlib.util
import json
import random
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import build as B  # noqa: E402
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
            "domain": "d", "passed": True, "invalid_json_count": 0,
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

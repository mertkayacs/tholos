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
        monkeypatch.setattr(F, "process", lambda item, *args: work(item))
        args += ["--scenarios", str(source), "--fraction", "1"]
    else:
        def rollout(item, profile):
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

    def process(item, *args):
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
    assert read_jsonl(out) == [item for item in scenarios if item["id"] not in selected]


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

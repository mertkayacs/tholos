"""Infrastructure retries and Kaggle shard generation without network calls."""

import importlib.util
import json
import sys
from datetime import UTC, datetime
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

PROFILE = {
    "base_url": "http://local.test/v1", "model": "teacher", "api_key": None,
    "json_mode": "schema", "temperature": 0, "max_tokens": 512,
}
ADD = {"tool": "table_add", "args": {"table": "items", "rows": [{"title": "Bird"}]}}
READ = {"tool": "note_read", "args": {"title": "Brief"}}
FINISH = {"tool": "finish", "args": {"summary": "Done"}}


def scenario(id="retry-001"):
    return {
        "id": id, "category": "table_add", "template": "t-retry",
        "agent": "Scout", "max_steps": 8,
        "workspace": {
            "agents": [{"name": "Scout", "role": "You check items.",
                        "tools": ["table_add", "note_read", "finish"]}],
            "tables": [{"name": "items", "columns": ["title"], "rows": []}],
            "notes": [{"title": "Brief", "body": "Original"}],
        },
        "trigger": {"kind": "message", "text": "Add Bird to items."},
        "reference": [ADD, FINISH],
        "expect": [{"type": "finished"}, {"type": "rows", "table": "items", "count": 1}],
        "checks": [{"kind": "text", "facts": ["Done"]}],
    }


def scripted_transport(script, requests):
    script = iter(script)

    def handler(request):
        requests.append(request)
        reply = next(script)
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"thought": "Next.", **reply})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        })

    return httpx.MockTransport(handler)


@pytest.fixture
def no_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    monkeypatch.setattr(pipeline, "uniform", lambda a, b: 0)
    return sleeps


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cli_args(source, out):
    return ["--scenarios", str(source), "--base-url", PROFILE["base_url"],
            "--model", PROFILE["model"], "--workers", "1", "--out", str(out)]


def test_mid_run_transport_failure_retries_from_scratch(no_backoff):
    requests = []
    script = [ADD, *[httpx.ReadTimeout("teacher timed out")] * 6, ADD, FINISH]
    with pipeline.TeacherTransport(scripted_transport(script, requests)) as transport:
        result = R.run_one(scenario(), PROFILE, "local", attempts=2, transport=transport)
    assert result["passed"] and result["status"] == "done" and result["error"] is None
    assert not result.get("infra_failed")
    assert result["steps"] == 2 and len(requests) == 9
    initial = json.loads(requests[0].content)["messages"]
    resumed = json.loads(requests[7].content)["messages"]
    assert len(initial) == len(resumed) == 2 and initial[0] == resumed[0]
    assert no_backoff == [1, 2, 4, 8, 16, 1]


@pytest.mark.parametrize("mid_run", [False, True])
def test_persistent_transport_failure_is_marked_and_resumed(
    mid_run, tmp_path, monkeypatch, no_backoff,
):
    requests = []
    attempt = ([ADD] if mid_run else []) + [httpx.ReadTimeout("teacher timed out")] * 6
    with pipeline.TeacherTransport(scripted_transport(attempt * 2, requests)) as transport:
        failed = R.run_one(scenario(), PROFILE, "local", attempts=2, transport=transport)
    assert not failed["passed"] and failed["infra_failed"] is True
    assert failed["status"] == "failed" and failed["error"] == "teacher timed out"
    assert failed["steps"] == int(mid_run)
    source, out = tmp_path / "scenarios.jsonl", tmp_path / "rollouts.jsonl"
    write_jsonl(source, [scenario()])
    write_jsonl(out, [failed])
    assert R.done_ids(out) == set()

    real_transport = pipeline.TeacherTransport
    resumed_requests = []
    mock = scripted_transport([ADD, FINISH], resumed_requests)
    monkeypatch.setattr(R, "TeacherTransport", lambda *args, **kwargs: real_transport(
        *(args or (mock,)), **kwargs))
    monkeypatch.setattr(R, "install_fixture_union", lambda _: None)
    assert R.main(cli_args(source, out)) == 0
    records = read_jsonl(out)
    assert len(records) == 2 and records[-1]["passed"]
    assert R.done_ids(out) == {scenario()["id"]}
    assert R.main(cli_args(source, out)) == 0
    assert len(read_jsonl(out)) == 2 and len(resumed_requests) == 2


@pytest.mark.parametrize("error,max_steps", [("stuck", 8), ("step limit", 2)])
def test_model_behavior_failures_are_not_retried(error, max_steps, no_backoff):
    item = scenario() | {"max_steps": max_steps}
    requests = []
    with pipeline.TeacherTransport(scripted_transport([READ] * 3, requests)) as transport:
        result = R.run_one(item, PROFILE, "local", transport=transport)
    assert not result["passed"]
    assert result["status"] == "failed" and result["error"] == error
    assert not result.get("infra_failed")
    assert len(requests) == min(3, max_steps) and no_backoff == []


def test_assertion_failure_is_not_retried(no_backoff):
    requests = []
    with pipeline.TeacherTransport(scripted_transport([FINISH], requests)) as transport:
        result = R.run_one(scenario(), PROFILE, "local", transport=transport)
    assert not result["passed"]
    assert result["status"] == "done" and result["error"] is None
    assert not result.get("infra_failed") and len(requests) == 1 and no_backoff == []


def test_rollouts_run_on_the_real_clock(no_backoff):
    before = datetime.now(UTC).replace(microsecond=0)
    with pipeline.TeacherTransport(scripted_transport([ADD, FINISH], [])) as transport:
        result = R.run_one(scenario(), PROFILE, "local", transport=transport)
    shown = result["messages"][1]["content"].split(", ")[0].removeprefix("Now: ")
    assert result["passed"] and before <= datetime.fromisoformat(shown) <= datetime.now(UTC)


def test_empty_transport_error_is_infrastructure_failure(no_backoff):
    requests = []
    with pipeline.TeacherTransport(scripted_transport(
        [httpx.ReadTimeout("")] * 6, requests,
    )) as transport:
        result = R.run_one(scenario(), PROFILE, "local", attempts=1, transport=transport)
    assert result["infra_failed"] and result["status"] == "failed" and result["error"] == ""


@pytest.mark.parametrize("status", ["running", "queued"])
@pytest.mark.parametrize("persistent", [False, True])
def test_unfinished_bench_result_is_retried_then_marked(
    status, persistent, no_backoff, monkeypatch, tmp_path,
):
    requests = []
    good = R.bench.run_scenario(scenario(), PROFILE, scripted_transport([ADD, FINISH], requests))
    unfinished = good | {"status": status, "passed": False, "error": None}
    results = iter([unfinished, unfinished if persistent else good])
    calls = []

    def run(*args):
        calls.append(args)
        return next(results)

    monkeypatch.setattr(R.bench, "run_scenario", run)
    result = R.run_one(scenario(), PROFILE, "local", attempts=2)
    assert len(calls) == 2 and no_backoff == [1]
    assert result["passed"] is not persistent
    assert result["infra_failed"] is persistent
    assert result["status"] == (status if persistent else "done") and result["error"] is None
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [result])
    assert R.done_ids(path) == (set() if persistent else {scenario()["id"]})


@pytest.mark.parametrize("timeout", [None, 600])
def test_cli_uses_scenario_retries_and_profile_timeout(
    timeout, tmp_path, monkeypatch, no_backoff,
):
    source, out = tmp_path / "scenarios.jsonl", tmp_path / "rollouts.jsonl"
    write_jsonl(source, [scenario()])
    requests = []
    script = [ADD, *[httpx.ReadTimeout("teacher timed out")] * 6, ADD, FINISH]
    real_transport = pipeline.TeacherTransport
    mock = scripted_transport(script, requests)
    monkeypatch.setattr(R, "TeacherTransport", lambda *args, **kwargs: real_transport(
        *(args or (mock,)), **kwargs))
    monkeypatch.setattr(R, "install_fixture_union", lambda _: None)
    args = cli_args(source, out)
    if timeout is not None:
        args += ["--timeout", str(timeout)]
    assert R.main(args) == 0
    assert read_jsonl(out)[0]["passed"]
    assert all(request.extensions["timeout"]["read"] == (timeout or 120) for request in requests)


def passing_record(id):
    return {"id": id, "category": "notes", "template": "t-note", "teacher": "local",
            "passed": True, "semantic_checked": True, "invalid_json_count": 0,
            "messages": [{"role": "assistant", "content": json.dumps(
                {"thought": "Done.", **FINISH})}]}


def test_build_excludes_infrastructure_and_uses_successful_resume(tmp_path, capsys):
    source, out = tmp_path / "rollouts.jsonl", tmp_path / "build"
    unresolved = passing_record("infra") | {"infra_failed": True, "category": "infra-only",
                                           "teacher": "infra-only", "passed": False}
    replaced = passing_record("resumed") | {"infra_failed": True, "passed": False}
    assert B.keep(passing_record("flagged-pass") | {"infra_failed": True}) == (
        False, "infrastructure failure")
    write_jsonl(source, [unresolved, replaced, passing_record("resumed")])
    assert B.main(["--rollouts", str(source), "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "infra-only" not in text and "100%" in text
    assert "infrastructure failures: 1" in text
    assert "total: 1 rollouts, 1 passing, 1 kept" in text
    records = read_jsonl(out / "sft_train.jsonl") + read_jsonl(out / "sft_val.jsonl")
    assert [record["meta"]["id"] for record in records] == ["resumed"]


def datagen_module():
    spec = importlib.util.spec_from_file_location(
        "rr_datagen", ROOT / "train" / "kaggle" / "datagen" / "datagen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fraction,already_phrased,partial", [
    (0.4, False, False), (0, False, False), (0.4, True, False), (0.4, False, True),
])
def test_kaggle_shard_phrasing_is_local_seeded_and_resume_safe(
    fraction, already_phrased, partial, tmp_path, monkeypatch, no_backoff,
):
    datagen = datagen_module()
    shard = tmp_path / "scenarios_kaggle.jsonl"
    items = [scenario(f"shard-{i}") for i in range(10)]
    if already_phrased:
        items[0]["phrased"] = "terse"
    write_jsonl(shard, items)
    selected = F.selected_ids(items, fraction, 1) if not already_phrased else set()
    if partial:
        saved = next(item for item in items if item["id"] in selected)
        write_jsonl(tmp_path / "scenarios_phrased.jsonl", [saved | {
            "phrased": "terse", "teacher": "local", "trigger": saved["trigger"] | {
                "text": "Please " + saved["trigger"]["text"]},
        }])
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        original = payload["messages"][-1]["content"].split("Message:\n", 1)[1]
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "Please " + original}}]})

    real_post = P.post_json
    mock = httpx.MockTransport(handler)
    monkeypatch.setattr(P, "post_json", lambda *args, **kwargs: real_post(
        *args, **kwargs, transport=mock))
    monkeypatch.setattr(datagen, "WORK", str(tmp_path))
    monkeypatch.setattr(datagen, "TRAIN", str(ROOT / "train"))
    monkeypatch.setattr(datagen, "PHRASING_FRACTION", fraction)
    monkeypatch.setattr(datagen, "find_inputs", lambda: None)
    monkeypatch.setattr(datagen, "copy_inputs", lambda **kwargs: [])
    monkeypatch.setattr(datagen, "start_server", lambda: SimpleNamespace(
        terminate=lambda: None, wait=lambda **kwargs: None))
    called, rollout_inputs = [], []

    def stage(name, args, **kwargs):
        called.append(name)
        if name == "phrasing":
            assert args[args.index("--seed") + 1] == "1"
            assert args[args.index("--base-url") + 1] == datagen.BASE_URL
            assert F.main(args[1:]) == 0
        elif name == "rollouts":
            assert args[args.index("--timeout") + 1] == "600"
            rollout_inputs.append(read_jsonl(Path(args[args.index("--scenarios") + 1])))
        return {"stage": name, "returncode": 0}

    monkeypatch.setattr(datagen, "stage", stage)
    datagen.main()
    should_phrase = fraction > 0 and not already_phrased
    assert called == (["phrasing"] if should_phrase else []) + ["rollouts", "build"]
    assert len(requests) == len(selected) - int(partial)
    rolled = {item["id"]: item for item in rollout_inputs[0]}
    assert set(rolled) == {item["id"] for item in items}
    for item in items:
        assert rolled[item["id"]]["trigger"]["text"] == (
            "Please " + item["trigger"]["text"] if item["id"] in selected
            else item["trigger"]["text"])
    assert read_jsonl(shard) == items
    datagen.main()
    assert len(requests) == len(selected) - int(partial)
    assert rollout_inputs[1] == rollout_inputs[0]


def test_kaggle_pass_rate_excludes_infrastructure_failures(tmp_path):
    datagen = datagen_module()
    path = tmp_path / "rollouts.jsonl"
    write_jsonl(path, [
        {"category": "notes", "passed": True, "steps": 2, "tokens": {"out": 8}},
        {"category": "notes", "passed": False, "steps": 1, "tokens": {"out": 4},
         "infra_failed": True},
    ])
    metrics = datagen.rollout_metrics(path, 2)
    assert metrics["categories"]["notes"] == {"items": 1, "passed": 1, "pass_rate": 1}
    assert metrics["infra_failed"] == 1


def test_kaggle_server_log_is_saved_in_working_output(tmp_path, monkeypatch):
    datagen = datagen_module()
    monkeypatch.setattr(datagen, "WORK", str(tmp_path))
    monkeypatch.setattr(datagen, "sh", lambda _: None)
    monkeypatch.setitem(sys.modules, "llama_cpp_binaries", SimpleNamespace(
        get_binary_path=lambda: "/mock/llama-server"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=lambda *args, **kwargs: "/mock/teacher.gguf"))
    monkeypatch.setattr(datagen.urllib.request, "urlopen", lambda *args, **kwargs: SimpleNamespace(
        read=lambda: '{"status":"ok"}'))
    process = SimpleNamespace()
    logs = []

    def popen(args, stdout, stderr):
        assert args[0] == "/mock/llama-server"
        assert stderr == datagen.subprocess.STDOUT
        assert Path(stdout.name) == tmp_path / "server.log"
        stdout.write("server startup diagnostics\n")
        logs.append(stdout)
        return process

    monkeypatch.setattr(datagen.subprocess, "Popen", popen)
    assert datagen.start_server() is process
    assert (tmp_path / "server.log").read_text() == "server startup diagnostics\n"
    assert logs[0].closed

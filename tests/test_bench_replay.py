import json
import socket
from pathlib import Path

import httpx
import pytest

from tholos.bench.runner import run_scenario
from tholos.workspace import dumps

SCENARIOS = Path(__file__).resolve().parents[1] / "tholos" / "bench" / "scenarios"
PROFILE = {
    "base_url": "http://replay.test/v1",
    "model": "reference",
    "api_key": None,
    "json_mode": "schema",
    "temperature": 0,
    "max_tokens": 512,
}


@pytest.mark.parametrize(
    "path",
    sorted(SCENARIOS.rglob("*.json")) or [None],
    ids=lambda path: path.stem if path else "missing-scenarios",
)
def test_reference_replay(path: Path | None, monkeypatch: pytest.MonkeyPatch) -> None:
    assert path is not None, "No benchmark scenarios found"
    scenario = json.loads(path.read_text(encoding="utf-8"))
    reference = scenario["reference"]
    emitted = 0
    errors = []

    def no_network(*args, **kwargs) -> None:
        errors.append("Non-fixture network access attempted")
        raise OSError("Reference replay permits only fixture fetches")

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal emitted
        messages = json.loads(request.content)["messages"]
        if messages and messages[-1]["content"].startswith("Invalid step:"):
            errors.append(messages[-1]["content"])
        if emitted >= len(reference):
            errors.append("Reference exhausted before the run ended")
            raise RuntimeError(errors[-1])
        step = reference[emitted]
        emitted += 1
        value = {"thought": "Next reference step.", "tool": step["tool"], "args": step["args"]}
        return httpx.Response(200, json={"choices": [{"message": {"content": dumps(value)}}]})

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    try:
        result = run_scenario(scenario, PROFILE, httpx.MockTransport(respond))
    except Exception as exc:
        failure = {
            "id": scenario["id"],
            "failed_assertions": [],
            "replay_errors": [f"{type(exc).__name__}: {exc}"],
        }
    else:
        if emitted != len(reference):
            errors.append(f"Emitted {emitted} of {len(reference)} reference steps")
        failure = {
            "id": scenario["id"],
            "failed_assertions": result["failed_assertions"],
            "replay_errors": errors,
        }
        if result["passed"] and not errors:
            return
    print("REPLAY_FAILURE " + dumps(failure))
    pytest.fail(dumps(failure), pytrace=False)

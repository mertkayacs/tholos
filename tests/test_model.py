import json
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from tholos import model, tools
from tholos import workspace as w

PROFILE = {
    "base_url": "http://local.test/v1",
    "model": "small",
    "api_key": None,
    "json_mode": "schema",
    "temperature": 0.2,
    "max_tokens": 512,
}


def completion(text, usage=None):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_valid_step():
    requests = []

    def handler(request):
        requests.append(request)
        return completion(
            w.dumps({"thought": "x" * 300, "tool": "finish", "args": {"summary": "Done"}})
        )

    messages = [{"role": "user", "content": "Check"}]
    result = model.step(PROFILE, messages, tools.schemas(["finish"]), httpx.MockTransport(handler))
    assert result.error is None and len(result.thought) == 240
    assert result.tool == "finish" and result.args == {"summary": "Done"}
    assert (result.tokens_in, result.tokens_out) == (10, 5)
    assert str(requests[0].url) == "http://local.test/v1/chat/completions"
    assert "authorization" not in requests[0].headers
    body = json.loads(requests[0].content)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.2 and body["max_tokens"] == 512
    assert list(json.loads(result.message()["content"])) == ["thought", "tool", "args"]
    assert messages == [{"role": "user", "content": "Check"}]


def test_invalid_then_valid():
    bodies = []
    replies = iter(["not json", '{"thought":"OK","tool":"finish","args":{"summary":"Ready"}}'])

    def handler(request):
        bodies.append(json.loads(request.content))
        return completion(next(replies))

    result = model.step(PROFILE, [], tools.schemas(["finish"]), httpx.MockTransport(handler))
    assert result.error is None and result.invalid_json_count == 1
    assert result.tokens_in == 20 and result.tokens_out == 10
    assert bodies[1]["messages"][-1]["role"] == "user"
    assert "Invalid step:" in bodies[1]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "text",
    [
        "```json\n{}\n```",
        "{} trailing",
        "[]",
        "null",
        '{"thought":"x","tool":"finish","args":{"summary":"x","extra":true}}',
        '{"thought":"x","tool":"unknown","args":{}}',
        '{"thought":"x","tool":"finish","args":{"summary":2}}',
        '{"thought":"x","thought":"y","tool":"finish","args":{"summary":"x"}}',
        '{"thought":"x","tool":"follow_up","args":{"minutes":NaN,"note":"x"}}',
    ],
)
def test_invalid_twice(text):
    result = model.step(
        PROFILE,
        [],
        tools.schemas(list(tools.SPECS)),
        httpx.MockTransport(lambda _: completion(text)),
    )
    assert result.error and result.invalid_json_count == 2 and result.tokens_in == 20


@pytest.mark.parametrize("mode", ["schema", "object", "none"])
def test_modes_auth_and_ollama(mode):
    def handler(request):
        body = json.loads(request.content)
        assert body["think"] is False
        assert request.headers["authorization"] == "Bearer secret"
        if mode in {"schema", "object"}:
            assert body["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in body
        return completion('{"thought":"OK","tool":"finish","args":{"summary":"Done"}}')

    result = model.step(
        PROFILE | {"base_url": "http://localhost:11434/v1", "api_key": "secret", "json_mode": mode},
        [],
        tools.schemas(["finish"]),
        httpx.MockTransport(handler),
    )
    assert result.error is None


def test_schema_contents_and_nested_validator(db):
    w.create_table(db, "leads", ["title", "score"], "you")
    schema = model.schema(tools.schemas(["table_add", "table_read", "finish"], db))
    branches = {branch["properties"]["tool"]["const"]: branch for branch in schema["anyOf"]}
    nullable = branches["table_read"]["properties"]["args"]["properties"]["limit"]
    assert nullable["type"] == ["integer", "null"]
    valid = {
        "thought": "Add",
        "tool": "table_add",
        "args": {"table": "leads", "rows": [{"title": "Sparrow", "score": None}]},
    }
    model.validate(valid, schema)
    for bad in [
        valid | {"args": {"table": "leads", "rows": [{"title": "Sparrow"}]}},
        valid | {"args": {"table": "leads", "rows": [{"title": "x", "score": {}}]}},
        valid | {"args": {"table": "leads", "rows": []}},
    ]:
        with pytest.raises(ValueError):
            model.validate(bad, schema)
    for value, spec in [
        (True, {"type": "integer"}),
        (float("inf"), {"type": "number"}),
        ("wrong", {"enum": ["replace", "append"]}),
        (0, {"type": "integer", "minimum": 1}),
    ]:
        with pytest.raises(ValueError):
            model.validate(value, spec)
    model.validate(None, {"type": ["string", "null"]})


@pytest.mark.parametrize(
    "url,ollama",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:11434/v1", True),
        ("http://[::1]:11434/v1", True),
        ("https://ollama.test/v1", True),
        ("https://OLLAMA.test/v1", True),
        ("https://proxy.test/OLLAMA/v1", True),
        ("http://localhost:8080/v1", False),
        ("https://api.test/v1", False),
    ],
)
def test_endpoint_selects_json_mode(url, ollama):
    def handler(request):
        body = json.loads(request.content)
        if ollama:
            assert body["response_format"] == {"type": "json_object"}
            assert body["think"] is False
        else:
            assert body["response_format"]["type"] == "json_schema"
            assert body["response_format"]["json_schema"]["strict"] is True
            assert "anyOf" in body["response_format"]["json_schema"]["schema"]
            assert "think" not in body
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        return completion('{"thought":"Ready","tool":"finish","args":{"summary":"Done"}}')

    profile = PROFILE | {"base_url": url}
    result = model.step(profile, [], tools.schemas(["finish"]), httpx.MockTransport(handler))
    assert result.error is None and profile["json_mode"] == "schema"
    assert list(json.loads(result.message()["content"])) == ["thought", "tool", "args"]


@pytest.mark.parametrize("valid_retry", [True, False])
def test_ollama_object_mode_still_validates_and_retries(valid_retry):
    valid = '{"thought":"Ready","tool":"finish","args":{"summary":"Done"}}'
    invalid = '{"thought":"Ready","tool":"remember","args":{"fact":"Wrong tool"}}'
    replies = iter([invalid, valid if valid_retry else invalid])
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["response_format"] == {"type": "json_object"}
        assert body["think"] is False
        return completion(next(replies))

    result = model.step(
        PROFILE | {"base_url": "http://localhost:11434/v1"},
        [],
        tools.schemas(["finish"]),
        httpx.MockTransport(handler),
    )
    assert len(bodies) == 2 and "Invalid step:" in bodies[1]["messages"][-1]["content"]
    assert result.tokens_in == 20 and result.tokens_out == 10
    assert result.invalid_json_count == (1 if valid_retry else 2)
    assert (result.error is None) == valid_retry
    if valid_retry:
        assert result.tool == "finish"


def test_ollama_override_logs_once_per_profile_under_concurrency(caplog):
    caplog.set_level(logging.WARNING, logger="tholos.model")
    profile = PROFILE | {
        "id": 10001,
        "base_url": "https://ollama.test/private-route",
        "model": "hidden-model-name",
        "api_key": "hidden-api-key",
    }

    def call(_):
        return model.step(
            profile,
            [],
            tools.schemas(["finish"]),
            httpx.MockTransport(
                lambda _: completion(
                    '{"thought":"Ready","tool":"finish","args":{"summary":"Done"}}'
                )
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, range(16)))
    assert all(result.error is None for result in results)
    notices = [record for record in caplog.records if record.name == "tholos.model"]
    assert len(notices) == 1 and "json_object" in notices[0].getMessage()
    model.step(
        profile | {"id": 10002},
        [],
        tools.schemas(["finish"]),
        httpx.MockTransport(
            lambda _: completion('{"thought":"Ready","tool":"finish","args":{"summary":"Done"}}')
        ),
    )
    assert len([record for record in caplog.records if record.name == "tholos.model"]) == 2
    assert "hidden-api-key" not in caplog.text and "hidden-model-name" not in caplog.text
    assert "private-route" not in caplog.text

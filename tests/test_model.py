import json

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
        if mode == "schema":
            assert body["response_format"]["json_schema"]["strict"] is True
        elif mode == "object":
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

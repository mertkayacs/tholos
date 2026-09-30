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


@pytest.mark.parametrize("fields,timeout", [({}, 120), ({"timeout": 600}, 600)])
def test_model_profile_timeout(fields, timeout):
    def handler(request):
        assert request.extensions["timeout"] == {
            "connect": timeout, "read": timeout, "write": timeout, "pool": timeout,
        }
        return completion('{"thought":"Done","tool":"finish","args":{"summary":"Ready"}}')

    result = model.step(
        PROFILE | fields, [], tools.schemas(["finish"]), httpx.MockTransport(handler)
    )
    assert result.error is None and result.tool == "finish"


@pytest.mark.parametrize("mode", ["schema", "object", "none"])
def test_missing_optional_args(mode):
    text = w.dumps({"thought": "Read", "tool": "table_read", "args": {"table": "leads"}})
    result = model.step(
        PROFILE | {"json_mode": mode},
        [],
        tools.schemas(["table_read"]),
        httpx.MockTransport(lambda _: completion(text)),
    )
    assert result.error is None and result.invalid_json_count == 0
    assert result.args == {"table": "leads", "query": None, "limit": None}
    assert result.message() == {
        "role": "assistant",
        "content": '{"thought":"Read","tool":"table_read",'
        '"args":{"table":"leads","query":null,"limit":null}}',
    }


@pytest.mark.parametrize("mode", ["schema", "object", "none"])
def test_args_follow_schema_order(mode):
    text = '{"thought":"Read","tool":"table_read","args":{"limit":5,"table":"leads"}}'
    result = model.step(
        PROFILE | {"json_mode": mode},
        [],
        tools.schemas(["table_read"]),
        httpx.MockTransport(lambda _: completion(text)),
    )
    assert result.error is None and result.invalid_json_count == 0
    assert result.args == {"table": "leads", "query": None, "limit": 5}
    assert list(result.args) == ["table", "query", "limit"]
    assert result.message() == {
        "role": "assistant",
        "content": '{"thought":"Read","tool":"table_read",'
        '"args":{"table":"leads","query":null,"limit":5}}',
    }


@pytest.mark.parametrize("mode", ["schema", "object", "none"])
def test_normalize_table_add_rows(db, mode):
    w.create_table(db, "alpha", ["name"], "you")
    w.create_table(db, "leads", ["title", "score"], "you")
    text = w.dumps(
        {
            "thought": "Add",
            "tool": "table_add",
            "args": {"rows": [{"score": 5}, {"score": 2, "title": "Sparrow"}], "table": "leads"},
        }
    )
    result = model.step(
        PROFILE | {"json_mode": mode},
        [],
        tools.schemas(["table_add"], db),
        httpx.MockTransport(lambda _: completion(text)),
    )
    assert result.error is None and result.invalid_json_count == 0
    assert result.args == {
        "table": "leads",
        "rows": [{"title": None, "score": 5}, {"title": "Sparrow", "score": 2}],
    }
    assert list(result.args) == ["table", "rows"]
    assert all(list(row) == ["title", "score"] for row in result.args["rows"])
    assert result.message()["content"] == (
        '{"thought":"Add","tool":"table_add","args":{"table":"leads",'
        '"rows":[{"title":null,"score":5},{"title":"Sparrow","score":2}]}}'
    )


def test_normalize_table_update_values(db):
    w.create_table(db, "alpha", ["name"], "you")
    w.create_table(db, "leads", ["title", "score", "status"], "you")
    row = w.add_rows(db, "leads", [{"title": "Sparrow", "score": 1, "status": "open"}], "you")[0]
    text = w.dumps(
        {
            "thought": "Update",
            "tool": "table_update",
            "args": {"values": {"score": 5}, "row": row, "table": "leads"},
        }
    )
    result = model.step(
        PROFILE | {"json_mode": "object"},
        [],
        tools.schemas(["table_update"], db),
        httpx.MockTransport(lambda _: completion(text)),
    )
    assert result.error is None and result.invalid_json_count == 0
    assert result.args == {
        "table": "leads", "row": row, "values": {"title": None, "score": 5, "status": None}
    }
    assert list(result.args) == ["table", "row", "values"]
    assert list(result.args["values"]) == ["title", "score", "status"]
    aid = w.save_agent(db, None, "Scout", "Check rows.", None, ["table_read", "table_update"])
    run = w.get_run(db, w.queue_run(db, aid, "Check", "message"))
    agent = w.get_agent(db, aid)
    read = tools.run_tool(
        db, run, agent, "table_read", {"table": "leads", "query": None, "limit": None}
    )
    assert "error" not in read
    updated = tools.run_tool(db, run, agent, "table_update", result.args)
    assert "error" not in updated
    assert w.get_table(db, "leads")["rows"][0]["data"] == {
        "title": "Sparrow", "score": 5, "status": "open"
    }


@pytest.mark.parametrize(
    "name,args,error",
    [
        (
            "table_add",
            {"table": "leads", "rows": [{"title": None, "score": 5, "extra": True}]},
            "step.args.rows[] contains unknown properties",
        ),
        (
            "table_update",
            {"table": "leads", "row": 1, "values": {"title": None, "score": 5, "extra": True}},
            "step.args.values contains unknown properties",
        ),
        (
            "table_add",
            {"table": "missing", "rows": [{"score": 5}]},
            "step.args.table must equal 'leads'",
        ),
        (
            "table_update",
            {"table": "missing", "row": 1, "values": {"score": 5}},
            "step.args.table must equal 'leads'",
        ),
    ],
)
def test_invalid_nested_args(db, name, args, error):
    w.create_table(db, "alpha", ["name"], "you")
    w.create_table(db, "leads", ["title", "score"], "you")
    text = w.dumps({"thought": "Write", "tool": name, "args": args})
    result = model.step(
        PROFILE | {"json_mode": "object"},
        [],
        tools.schemas([name], db),
        httpx.MockTransport(lambda _: completion(text)),
    )
    assert result.error and error in result.error
    assert result.invalid_json_count == 2


@pytest.mark.parametrize(
    "args,error",
    [
        ({"query": None, "limit": None}, "step.args is missing ['table']"),
        (
            {"table": "leads", "query": None, "limit": None, "extra": True},
            "step.args contains unknown properties",
        ),
        (
            {"table": "leads", "query": 2, "limit": None},
            "step.args.query must have type ['string', 'null']",
        ),
        (
            {"table": "leads", "query": None, "limit": "10"},
            "step.args.limit must have type ['integer', 'null']",
        ),
    ],
)
def test_invalid_table_read_args(args, error):
    text = w.dumps({"thought": "Read", "tool": "table_read", "args": args})
    result = model.step(
        PROFILE,
        [],
        tools.schemas(["table_read"]),
        httpx.MockTransport(lambda _: completion(text)),
    )
    assert result.error == "step does not match an allowed schema: " + error
    assert result.invalid_json_count == 2


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
        assert body["reasoning_effort"] == "none"
        assert "think" not in body
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
            assert body["reasoning_effort"] == "none"
        else:
            assert body["response_format"]["type"] == "json_schema"
            assert body["response_format"]["json_schema"]["strict"] is True
            assert "anyOf" in body["response_format"]["json_schema"]["schema"]
            assert "reasoning_effort" not in body
        assert "think" not in body
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        return completion('{"thought":"Ready","tool":"finish","args":{"summary":"Done"}}')

    profile = PROFILE | {"base_url": url}
    result = model.step(profile, [], tools.schemas(["finish"]), httpx.MockTransport(handler))
    assert result.error is None and profile["json_mode"] == "schema"
    assert list(json.loads(result.message()["content"])) == ["thought", "tool", "args"]


def test_llama_cpp_profile_payload_is_unchanged():
    spec = tools.schemas(["finish"])
    messages = [{"role": "user", "content": "Check"}]

    def handler(request):
        assert json.loads(request.content) == {
            "model": "small", "messages": messages, "temperature": 0.2, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "step", "strict": True, "schema": model.schema(spec)},
            },
        }
        return completion('{"thought":"Ready","tool":"finish","args":{"summary":"Done"}}')

    result = model.step(
        PROFILE | {"base_url": "http://localhost:8080/v1"},
        messages, spec, httpx.MockTransport(handler),
    )
    assert result.error is None


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
        assert body["reasoning_effort"] == "none"
        assert "think" not in body
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

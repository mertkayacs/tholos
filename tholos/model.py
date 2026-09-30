import json
import logging
import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

import httpx

from tholos import prompt
from tholos.workspace import dumps

log = logging.getLogger(__name__)
_OLLAMA_WARNED: set[tuple[int | None, str, str]] = set()
_OLLAMA_LOCK = Lock()


@dataclass
class Step:
    thought: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    invalid_json_count: int = 0
    error: str | None = None

    def message(self) -> dict:
        return {
            "role": "assistant",
            "content": dumps({"thought": self.thought, "tool": self.tool, "args": self.args}),
        }


def schema(tools: dict[str, dict]) -> dict:
    if not tools:
        raise ValueError("An agent must have at least one tool")
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "tool": {"const": name},
                    "args": args,
                },
                "required": ["thought", "tool", "args"],
                "additionalProperties": False,
            }
            for name, args in sorted(tools.items())
        ]
    }


def validate(value: Any, spec: dict, path: str = "step") -> None:
    if "anyOf" in spec:
        errors = []
        for branch in spec["anyOf"]:
            try:
                validate(value, branch, path)
                return
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f"{path} does not match an allowed schema: " + "; ".join(errors)[:1000])
    if "const" in spec and (type(value) is not type(spec["const"]) or value != spec["const"]):
        raise ValueError(f"{path} must equal {spec['const']!r}")
    if "enum" in spec and not any(
        type(value) is type(item) and value == item for item in spec["enum"]
    ):
        raise ValueError(f"{path} must be one of {spec['enum']}")
    types = spec.get("type", [])
    types = [types] if isinstance(types, str) else types
    valid = {
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) is int or (type(value) is float and math.isfinite(value)),
        "boolean": type(value) is bool,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "null": value is None,
    }
    if types and not any(valid.get(kind, False) for kind in types):
        raise ValueError(f"{path} must have type {types}")
    if isinstance(value, dict):
        properties = spec.get("properties", {})
        missing = set(spec.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        if spec.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ValueError(f"{path} contains unknown properties")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}")
    if isinstance(value, list):
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", len(value)):
            raise ValueError(f"{path} has an invalid number of items")
        for item in value:
            validate(item, spec.get("items", {}), path + "[]")
    if type(value) in (int, float) and (
        value < spec.get("minimum", value) or value > spec.get("maximum", value)
    ):
        raise ValueError(f"{path} is out of range")


def _normalize(value: Any, spec: dict) -> Any:
    """Fill nullable properties and order keys recursively by schema."""
    if "anyOf" in spec:
        for branch in spec["anyOf"]:
            if isinstance(value, dict) and all(
                key in value
                and type(value[key]) is type(prop["const"])
                and value[key] == prop["const"]
                for key, prop in branch.get("properties", {}).items()
                if "const" in prop
            ):
                return _normalize(value, branch)
        return value
    if isinstance(value, dict) and "properties" in spec:
        result = {}
        for key, prop in spec["properties"].items():
            if key in value:
                result[key] = _normalize(value[key], prop)
            elif "null" in prop.get("type", []):
                result[key] = None
        result.update((key, item) for key, item in value.items() if key not in spec["properties"])
        return result
    if isinstance(value, list) and "items" in spec:
        return [_normalize(item, spec["items"]) for item in value]
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def step(
    profile: dict,
    messages: list[dict],
    tools: dict[str, dict],
    transport: httpx.BaseTransport | None = None,
) -> Step:
    spec = schema(tools)
    payload = {
        "model": profile["model"],
        "messages": messages.copy(),
        "temperature": profile["temperature"],
        "max_tokens": profile["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    base_url = profile["base_url"].rstrip("/")
    parsed = urlsplit(base_url)
    ollama = (
        parsed.port == 11434
        or "ollama" in (parsed.hostname or "").lower()
        or "ollama" in parsed.path.lower()
    )
    if ollama:
        payload["think"] = False
    mode = profile.get("json_mode", "schema")
    if ollama and mode == "schema":
        mode = "object"
        key = (profile.get("id"), base_url, profile["model"])
        with _OLLAMA_LOCK:
            first = key not in _OLLAMA_WARNED
            _OLLAMA_WARNED.add(key)
        if first:
            log.warning(
                "Ollama profile %s uses json_object to preserve thought/tool/args order; "
                "local schema validation remains enabled.",
                profile.get("id", "unregistered"),
            )
    if mode == "schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "step", "strict": True, "schema": spec},
        }
    elif mode == "object":
        payload["response_format"] = {"type": "json_object"}
    elif mode != "none":
        raise ValueError("json_mode must be schema, object, or none")
    headers = {"Authorization": f"Bearer {profile['api_key']}"} if profile.get("api_key") else {}
    result = Step()
    with httpx.Client(transport=transport, timeout=120, trust_env=False) as client:
        for attempt in range(2):
            response = client.post(base_url + "/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            result.tokens_in += usage.get("prompt_tokens", 0) or 0
            result.tokens_out += usage.get("completion_tokens", 0) or 0
            try:
                text = data["choices"][0]["message"]["content"]
                value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
                if isinstance(value, dict) and "args" in value:
                    name = value.get("tool")
                    if isinstance(name, str) and name in tools:
                        value["args"] = _normalize(value["args"], tools[name])
                validate(value, spec)
                result.thought, result.tool, result.args = (
                    value["thought"][:240],
                    value["tool"],
                    value["args"],
                )
                result.error = None
                return result
            except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
                result.invalid_json_count += 1
                result.error = str(exc)
                if attempt == 0:
                    payload["messages"].append(prompt.retry(result.error))
                else:
                    return result
    return result

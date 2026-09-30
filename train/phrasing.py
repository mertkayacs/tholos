"""Optional pass: the teacher rewrites each trigger in one of three styles.

Usage:
  python train/phrasing.py --scenarios scenarios.jsonl \
      --base-url http://127.0.0.1:8080/v1 --model teacher --out phrased.jsonl

A rewrite is rejected when it changes material names, identifiers, numbers,
column assignments, structured rows or instruction scope. Resumable by scenario id.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from packs import chat
from pipeline import UsageLimitError, add_hosted_args, completed, hosted_config

STYLES = {
    "terse": "Rewrite the message in a terse, rushed style: short, clipped, "
             "lowercase where natural, no greeting.",
    "polite": "Rewrite the message in a polite, complete style with a greeting "
              "sense and full sentences.",
    "typo": "Rewrite the message keeping the same meaning but with two or "
            "three natural typos a person makes when typing fast.",
}
SYSTEM = ("You rewrite short work messages. Keep every fact: names, numbers, "
          "column names, table names, URLs. Change only tone and phrasing. "
          "Reply with the rewritten message text only, no quotes, no "
          "commentary. No em dashes.")


def trigger_fields(scenario):
    kind = scenario["trigger"]["kind"]
    if kind == "task":
        return "details"
    return {"schedule": "prompt", "follow_up": "note", "message": "text"}[kind]


def facts(scenario, text):
    wanted = set(re.findall(r"(?<!\w)[+-]?\d+(?:[./:-]\d+)*(?!\w)", text))
    low = text.casefold()
    candidates = []
    for agent in scenario["workspace"].get("agents", []):
        candidates.append(agent["name"])
    candidates.extend(scenario.get("fixtures", {}))
    candidates.extend(note["title"] for note in scenario["workspace"].get("notes", []))
    for table in scenario["workspace"]["tables"]:
        candidates.extend([table["name"], *table["columns"]])
        candidates.extend(str(value) for row in table["rows"] for value in row.values()
                          if value is not None)
    for step in scenario.get("reference", []):
        args = step["args"]
        candidates.extend(args.get(key) for key in ("table", "title", "to", "url")
                          if args.get(key))
        candidates.extend(args.get("columns", []))
        for row in args.get("rows", []) + [args.get("values", {})]:
            candidates.extend(str(value) for value in row.values() if value is not None)
        for key in ("details", "text"):
            candidates.extend(line.strip().lstrip("-* ") for line in args.get(key, "").splitlines())
    for candidate in candidates:
        value = str(candidate).strip().casefold()
        if value and _contains(low, value):
            wanted.add(value)
    return wanted


def _contains(text, fact):
    return re.search(r"(?<!\w)" + re.escape(fact) + r"(?!\w)", text) is not None


def _structured_rows(text):
    rows = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        if text[index] not in "[{":
            index += 1
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except ValueError:
            index += 1
            continue
        values = value if isinstance(value, list) else [value]
        rows.extend(json.dumps(row, sort_keys=True).casefold() for row in values
                    if isinstance(row, dict))
        index += end
    return sorted(rows)


def _associated(text, column, value, columns):
    """Keep literal assignment values attached to their named column."""
    low = text.casefold()
    for match in re.finditer(r"(?<!\w)" + re.escape(column.casefold()) + r"(?!\w)", low):
        tail = low[match.end():]
        stops = [hit.start() for other in columns if other != column
                 for hit in re.finditer(r"(?<!\w)" + re.escape(other.casefold()) + r"(?!\w)", tail)]
        if _contains(tail[:min(stops)] if stops else tail, str(value).casefold()):
            return True
    return False


def _assignments_preserved(scenario, original, rewritten):
    tables = {table["name"]: table for table in scenario["workspace"]["tables"]}
    recipients = {step["args"]["to"] for step in scenario.get("reference", [])
                  if step["tool"] == "task_add"}
    for step in scenario.get("reference", []):
        args = step["args"]
        if step["tool"] == "table_update":
            columns = tables[args["table"]]["columns"]
            for column, value in args["values"].items():
                if (value is not None and _associated(original, column, value, columns)
                        and not _associated(rewritten, column, value, columns)):
                    return False
        elif step["tool"] == "task_add" and len(recipients) > 1:
            item = args["details"]
            if not _contains(original.casefold(), item.casefold()):
                continue
            for text in (original, rewritten):
                parts = re.split(r"\s+and\s+|[;/\n]", text, flags=re.I)
                segment = next((part for part in parts
                                if _contains(part.casefold(), item.casefold())), None)
                if segment is not None and not _contains(segment.casefold(), args["to"].casefold()):
                    return False
    return True


def preserves_facts(scenario, original, rewritten):
    low = rewritten.casefold()
    if not all(_contains(low, fact) for fact in facts(scenario, original)):
        return False
    numbers = r"(?<!\w)[+-]?\d+(?:[./:-]\d+)*(?!\w)"
    if set(re.findall(numbers, original)) != set(re.findall(numbers, rewritten)):
        return False
    if _structured_rows(original) != _structured_rows(rewritten):
        return False
    dangerous = r"\b(?:delete|erase|remove|ignore|cancel|disregard|forget|overwrite)\b"
    if set(re.findall(dangerous, low)) - set(re.findall(dangerous, original.casefold())):
        return False
    actions = [r"\b(?:add|put|insert|import|record)\b", r"\b(?:set|update|change|rename|mark)\b",
               r"\b(?:create|start|build|make)\b", r"\b(?:append|tack)\b",
               r"\b(?:route|assign|hand|split)\b"]
    if any(re.search(action, original.casefold()) and not re.search(action, low)
           for action in actions):
        return False
    if not _assignments_preserved(scenario, original, rewritten):
        return False
    groups = [r"\b(?:not|never|no|dont|don't|without)\b",
              r"\b(?:every|all|each|both)\b", r"\bbefore\b", r"\bafter\b",
              r"\b(?:if|unless)\b"]
    return all(bool(re.search(group, original.casefold())) == bool(re.search(group, low))
               for group in groups)


def rewrite(base_url, model, text, style, temperature, api_key=None,
            json_mode="none", throttle=None):
    if json_mode == "object":
        instruction = ("Reply with one JSON object: {\"rewrite\": \"...\"}. "
                       "Keep every fact.")
    else:
        instruction = "Reply with the rewritten message text only, no quotes, no commentary."
    user = f"{STYLES[style]}\n{instruction}\n\nMessage:\n{text}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    reply = chat(base_url, model, messages, None, temperature, timeout=120,
                 api_key=api_key, json_mode=json_mode, throttle=throttle, max_tokens=512)
    if json_mode == "object":
        reply = str(json.loads(reply)["rewrite"])
    return reply


def process(scenario, base_url, model, temperature, api_key=None, json_mode="none",
            throttle=None):
    field = trigger_fields(scenario)
    original = scenario["trigger"][field]
    style = list(STYLES)[sum(bytearray(scenario["id"], "utf-8")) % 3]
    try:
        text = rewrite(base_url, model, original, style, temperature,
                       api_key=api_key, json_mode=json_mode, throttle=throttle)
    except UsageLimitError:
        raise
    except Exception:  # noqa: BLE001 - network hiccup: keep the original
        return scenario
    if (isinstance(text, str) and text.strip()
            and preserves_facts(scenario, original, text) and "\u2014" not in text):
        scenario = dict(scenario)
        trigger = dict(scenario["trigger"])
        trigger[field] = text.strip()
        scenario["trigger"] = trigger
        scenario["phrased"] = style
    return scenario


def selected_ids(scenarios, fraction, seed):
    """Sample from sorted IDs so selection stays stable across resume and input order."""
    ids = sorted({scenario["id"] for scenario in scenarios})
    return set(random.Random(seed).sample(ids, round(len(ids) * fraction)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rephrase scenario triggers.")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deadline", type=float, default=None, help="stop new items at Unix time")
    add_hosted_args(parser)
    args = parser.parse_args(argv)
    api_key, throttle = hosted_config(parser, args)
    if not 0 <= args.fraction <= 1:
        parser.error("--fraction must be between 0 and 1")

    scenarios = [json.loads(line) for line in
                 Path(args.scenarios).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["id"] for line in
                Path(args.out).read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in scenarios if s["id"] not in done]
    selected = selected_ids(scenarios, args.fraction, args.seed)
    print(f"{len(todo)} triggers to phrase ({len(done)} already done)", flush=True)

    written = 0
    with open(args.out, "a", encoding="utf-8") as file:
        # Unselected scenarios retain their template phrasings without any model call.
        for scenario in todo:
            if scenario["id"] not in selected:
                scenario = dict(scenario, teacher=args.teacher)
                file.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                file.flush()
                written += 1
        selected_todo = [s for s in todo if s["id"] in selected]

        def work(scenario):
            return process(scenario, args.base_url, args.model, args.temperature,
                           api_key=api_key, json_mode=args.json_mode, throttle=throttle)

        for _, scenario in completed(work, selected_todo, args.workers, args.deadline):
            scenario = dict(scenario, teacher=args.teacher)
            file.write(json.dumps(scenario, ensure_ascii=False) + "\n")
            file.flush()
            written += 1
            if written % 100 == 0:
                print(f"{written}/{len(todo)}", flush=True)
    print(f"wrote {written} scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

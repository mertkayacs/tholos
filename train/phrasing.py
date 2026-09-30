"""Optional pass: the teacher rewrites each trigger in one of three styles.

Usage:
  python train/phrasing.py --scenarios scenarios.jsonl \
      --base-url http://127.0.0.1:8080/v1 --model teacher --out phrased.jsonl

A rewrite is rejected (original kept) when it drops a number or a column name
that the original trigger contained. Resumable by scenario id.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from packs import chat

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
    wanted = set(re.findall(r"\d+", text))
    low = text.casefold()
    for table in scenario["workspace"]["tables"]:
        if table["name"].casefold() in low:
            wanted.add(table["name"].casefold())
        for column in table["columns"]:
            if column.casefold() in low:
                wanted.add(column.casefold())
    return wanted


def rewrite(base_url, model, text, style, temperature):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{STYLES[style]}\n\nMessage:\n{text}"},
    ]
    return chat(base_url, model, messages, None, temperature, timeout=120)


def process(scenario, base_url, model, temperature):
    field = trigger_fields(scenario)
    original = scenario["trigger"][field]
    style = list(STYLES)[sum(bytearray(scenario["id"], "utf-8")) % 3]
    try:
        text = rewrite(base_url, model, original, style, temperature)
    except Exception:  # noqa: BLE001 - network hiccup: keep the original
        return scenario
    wanted = facts(scenario, original)
    low = str(text).casefold()
    if str(text).strip() and all(f in low for f in wanted) and "\u2014" not in text:
        scenario = dict(scenario)
        trigger = dict(scenario["trigger"])
        trigger[field] = text.strip()
        scenario["trigger"] = trigger
        scenario["phrased"] = style
    return scenario


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rephrase scenario triggers.")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args(argv)

    scenarios = [json.loads(line) for line in
                 Path(args.scenarios).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["id"] for line in
                Path(args.out).read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in scenarios if s["id"] not in done]
    print(f"{len(todo)} triggers to phrase ({len(done)} already done)", flush=True)

    with open(args.out, "a", encoding="utf-8") as file, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, scenario in enumerate(
                    pool.map(lambda s: process(s, args.base_url, args.model,
                                               args.temperature), todo), 1):
                file.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                file.flush()
                if i % 100 == 0:
                    print(f"{i}/{len(todo)}", flush=True)
    print(f"wrote {len(todo)} scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

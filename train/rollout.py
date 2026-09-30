"""Run scenarios through the real bench harness with the teacher as the model.

Usage:
  python train/rollout.py --scenarios scenarios.jsonl \
      --base-url http://127.0.0.1:8080/v1 --model teacher --workers 8 --out rollouts.jsonl

Resumable by scenario id: ids already present in --out are skipped.

The bench harness swaps the global fetch.FIXTURES per scenario, which is not
thread-safe. Every fixture URL is unique per scenario, so this script installs
one union of all fixtures for the whole run: fetch.get checks the union first
and only then falls back to its normal path. The runtime code itself is not
modified.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tholos import fetch
from tholos.bench import runner as bench


def install_fixture_union(scenarios):
    union = {}
    for scenario in scenarios:
        union.update(scenario.get("fixtures", {}))
    real_get = fetch.get

    def get(url, transport=None):
        if url in union:
            return fetch._result(url, union[url], "text/html")
        return real_get(url, transport)

    fetch.get = get
    return union


def run_one(scenario, profile):
    try:
        result = bench.run_scenario(scenario, profile)
        return {
            "id": scenario["id"],
            "category": scenario["category"],
            "template": scenario["template"],
            "domain": scenario.get("domain", ""),
            "passed": result["passed"],
            "failed_assertions": result["failed_assertions"],
            "steps": result["steps"],
            "invalid_json_count": result["invalid_json_count"],
            "tokens": result["tokens"],
            "seconds": result["seconds"],
            "messages": result["messages"],
        }
    except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the run
        return {
            "id": scenario["id"],
            "category": scenario["category"],
            "template": scenario["template"],
            "domain": scenario.get("domain", ""),
            "passed": False,
            "failed_assertions": [{"type": "crash", "error": repr(exc)}],
            "steps": 0,
            "invalid_json_count": 0,
            "tokens": {"in": 0, "out": 0},
            "seconds": 0.0,
            "messages": [],
        }


def done_ids(path):
    ids = set()
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description="Teacher rollouts on training scenarios.")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    scenarios = [json.loads(line) for line in
                 Path(args.scenarios).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    done = done_ids(args.out)
    todo = [s for s in scenarios if s["id"] not in done]
    install_fixture_union(todo)
    print(f"{len(todo)} scenarios to run ({len(done)} already done)", flush=True)

    profile = {
        "base_url": args.base_url,
        "model": args.model,
        "api_key": None,
        "json_mode": "schema",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    passed = 0
    with open(args.out, "a", encoding="utf-8") as file, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(
                    lambda s: run_one(s, profile), todo), 1):
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                passed += result["passed"]
                if i % 25 == 0:
                    print(f"{i}/{len(todo)} done, {passed} passing", flush=True)
    print(f"wrote {len(todo)} rollouts to {args.out}; {passed} passing")
    return 0


if __name__ == "__main__":
    sys.exit(main())

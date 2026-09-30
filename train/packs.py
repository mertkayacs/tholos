"""Teacher-written domain packs: JSONL in (domains), JSONL out (packs).

Usage:
  python train/packs.py --base-url http://127.0.0.1:8080/v1 --model teacher \
      --per-domain 4 --workers 8 --out packs.jsonl

Resumable: existing (domain, index) keys in --out are skipped.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import httpx
from pipeline import completed
from templates import validate_pack

PACK_SCHEMA = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "columns": {"type": "array", "minItems": 3, "maxItems": 8,
                                "items": {"type": "string"}},
                    "rows": {"type": "array", "minItems": 6, "maxItems": 24,
                             "items": {"type": "object"}},
                },
                "required": ["name", "columns", "rows"],
                "additionalProperties": False,
            },
        },
        "notes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"},
                               "body": {"type": "string"}},
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
        "team": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "role": {"type": "string"}},
                "required": ["name", "role"],
                "additionalProperties": False,
            },
        },
        "fresh": {"type": "array", "minItems": 5, "maxItems": 5,
                  "items": {"type": "string"}},
    },
    "required": ["tables", "notes", "team", "fresh"],
    "additionalProperties": False,
}

SYSTEM = (
    "You write compact JSON datasets for a shared workspace app. You reply with "
    "strict JSON only, matching the given schema. Content rules: realistic and "
    "specific, never placeholder text; lowercase snake_case table and column "
    "names; first column values unique per table; cell values are strings, "
    "numbers, booleans, or null; no em dashes anywhere."
)

USER = """Domain: {domain}
Variation seed: {seed} (make this pack clearly different from other packs for the same domain).

Write a workspace pack for this domain:
- one or two tables: name, 3 to 8 columns, 6 to 24 realistic rows
- one to three notes: a title and a body of one to four sentences
- two to four teammates: short distinct first names, each with a different
  one-sentence responsibility that ends with a period
- five "fresh" items: short realistic NEW entries for this domain that are NOT
  present in the tables (they will be used as incoming data)

Keep every string free of em dashes. Use plain ASCII quotes."""


def chat(base_url, model, messages, schema, temperature, timeout=180):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "pack", "strict": True, "schema": schema}}
    response = httpx.post(base_url.rstrip("/") + "/chat/completions", json=payload,
                          timeout=timeout, trust_env=False)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def one_pack(base_url, model, domain, index, seed, temperature, attempts=4):
    """Ask the teacher for one pack; retry until it validates."""
    last = "no attempt"
    for _ in range(attempts):
        try:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(domain=domain, index=index,
                                                        seed=seed)},
            ]
            pack = json.loads(chat(base_url, model, messages, PACK_SCHEMA, temperature))
            pack["domain"] = domain
            error = validate_pack(pack)
            if error is None:
                return pack, None
            last = error
        except Exception as exc:  # noqa: BLE001 - recorded, retried
            last = repr(exc)
    return None, last


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                keys.add((item["domain"], item["index"]))
    return keys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate domain packs with a teacher model.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domains", default=str(Path(__file__).parent / "domains.txt"))
    parser.add_argument("--per-domain", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="stop after N new packs")
    parser.add_argument("--domains-limit", type=int, default=None, help="use the first N domains")
    parser.add_argument("--deadline", type=float, default=None, help="stop new items at Unix time")
    args = parser.parse_args(argv)

    domains = [d.strip() for d in Path(args.domains).read_text(encoding="utf-8")
               .splitlines() if d.strip()]
    if args.domains_limit is not None:
        if args.domains_limit < 1:
            parser.error("--domains-limit must be positive")
        domains = domains[:args.domains_limit]
    keys = done_keys(args.out)
    jobs = [(domain, i) for domain in domains for i in range(args.per_domain)
            if (domain, i) not in keys]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} packs to write ({len(keys)} already done)", flush=True)

    lock = threading.Lock()
    written, failed = [0], []

    def work(job):
        domain, index = job
        seed = f"{domain}:{index}"
        pack, error = one_pack(args.base_url, args.model, domain, index, seed,
                               args.temperature)
        if pack is None:
            return False
        line = json.dumps({"domain": domain, "index": index, "pack": pack},
                          ensure_ascii=False)
        with lock:
            with open(args.out, "a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()
            written[0] += 1
        return True

    # Open even on an expired deadline, so later stages can consume an empty checkpoint.
    Path(args.out).touch(exist_ok=True)
    for job, success in completed(work, jobs, args.workers, args.deadline):
        if not success:
            failed.append(job)
    print(f"wrote {written[0]} packs to {args.out}; {len(failed)} jobs failed")
    for job in failed[:20]:
        print("  failed:", job)
    expired = args.deadline is not None and time.time() >= args.deadline
    return 1 if failed and not expired else 0


if __name__ == "__main__":
    sys.exit(main())

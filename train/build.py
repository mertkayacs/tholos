"""Filter rollouts, deduplicate, split by template, write SFT files.

Usage:
  python train/build.py --rollouts rollouts.jsonl --out-dir /kaggle/working

Keeps a rollout when: it passed the semantic audit, it has zero invalid outputs, no call repeats
back to back, no call appears more than twice, every thought is at most 240
characters, and it ends with finish. Near-duplicate trajectories inside a
template are dropped. The val split gets whole templates, about 8 percent of
kept trajectories.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from tholos.workspace import dumps


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_trajectory(result):
    """Extract (tool, args) steps and thoughts from the runtime messages."""
    steps, thoughts, finish = [], [], False
    for message in result.get("messages", []):
        if message.get("role") != "assistant":
            continue
        try:
            value = json.loads(message["content"])
        except ValueError:
            return None
        thoughts.append(str(value.get("thought", "")))
        steps.append((value.get("tool", ""), canonical(value.get("args", {}))))
        if value.get("tool") == "finish":
            finish = True
    return steps, thoughts, finish


def keep(result):
    """Return (ok, reason). Applied to passing rollouts only."""
    if not result.get("passed"):
        return False, "failed assertions"
    if result.get("semantic_checked") is not True:
        return False, "missing semantic audit"
    if result.get("invalid_json_count", 0):
        return False, "invalid output"
    parsed = parse_trajectory(result)
    if parsed is None:
        return False, "unparseable assistant message"
    steps, thoughts, finish = parsed
    if not steps or not finish:
        return False, "no finish"
    for i in range(1, len(steps)):
        if steps[i] == steps[i - 1]:
            return False, "repeated identical consecutive call"
    counts = Counter(steps)
    if counts and max(counts.values()) > 2:
        return False, "call repeated more than twice"
    if any(len(t) > 240 for t in thoughts):
        return False, "thought over 240 chars"
    return True, ""


def signature(result):
    """Rough text signature for near-duplicate detection."""
    texts = [m["content"] for m in result.get("messages", [])
             if m.get("role") == "assistant"]
    return " ".join(texts)


def drop_near_duplicates(results, ratio=0.95):
    kept, dropped = [], 0
    buckets = defaultdict(list)
    for result in results:
        buckets[result["template"]].append(result)
    for bucket in buckets.values():
        survivors = []
        for result in bucket:
            text = signature(result)
            duplicate = False
            for other in survivors:
                if SequenceMatcher(None, text, signature(other), autojunk=True
                                   ).quick_ratio() < ratio:
                    continue
                if SequenceMatcher(None, text, signature(other), autojunk=True
                                   ).ratio() >= ratio:
                    duplicate = True
                    break
            if duplicate:
                dropped += 1
            else:
                survivors.append(result)
        kept.extend(survivors)
    return kept, dropped


def split_templates(results, frac, seed):
    """Choose val templates so val gets about frac of trajectories, never shared."""
    by_template = defaultdict(list)
    for result in results:
        by_template[result["template"]].append(result)
    templates = sorted(by_template)
    rng = random.Random(seed)
    rng.shuffle(templates)
    total = len(results)
    target = max(1, round(frac * total))
    val, count = [], 0
    for name in templates:
        if count >= target:
            break
        val.append(name)
        count += len(by_template[name])
    return set(templates) - set(val), set(val)


def write_split(results, path):
    with open(path, "w", encoding="utf-8") as file:
        for result in results:
            record = {
                "messages": result["messages"],
                "meta": {
                    "id": result["id"],
                    "category": result["category"],
                    "template": result["template"],
                    "domain": result.get("domain", ""),
                    "teacher": result.get("teacher", ""),
                    "steps": result.get("steps", 0),
                    "tokens": result.get("tokens", {"in": 0, "out": 0}),
                },
            }
            file.write(dumps(record) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build SFT train/val files from rollouts.")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--val-frac", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    results = [json.loads(line) for line in
               Path(args.rollouts).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    seen, unique = set(), []
    for result in results:
        if result["id"] not in seen:
            seen.add(result["id"])
            unique.append(result)

    kept, reasons = [], Counter()
    for result in unique:
        ok, reason = keep(result)
        if ok:
            kept.append(result)
        else:
            reasons[reason] += 1
    kept, dropped_dupes = drop_near_duplicates(kept)

    train_templates, val_templates = split_templates(kept, args.val_frac, args.seed)
    train = [r for r in kept if r["template"] in train_templates]
    val = [r for r in kept if r["template"] in val_templates]
    assert not (train_templates & val_templates)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_split(train, out / "sft_train.jsonl")
    write_split(val, out / "sft_val.jsonl")

    passed = Counter(r["category"] for r in unique if r["passed"])

    def table(rows):
        print(f"{'':<20} {'Rollouts':>9} {'Passed':>8} {'Kept':>6} {'Rate':>7}")
        for name in sorted(rows):
            group = rows[name]
            rate = group["passed"] / group["attempted"] if group["attempted"] else 0
            print(f"{name:<20} {group['attempted']:>9} {group['passed']:>8} "
                  f"{group['kept']:>6} {rate:>6.0%}")

    def grouped(key):
        rows = defaultdict(lambda: {"attempted": 0, "passed": 0, "kept": 0})
        for result in unique:
            rows[result.get(key, "")]["attempted"] += 1
            rows[result.get(key, "")]["passed"] += bool(result["passed"])
        for result in kept:
            rows[result.get(key, "")]["kept"] += 1
        return rows

    print("Per category:")
    table(grouped("category"))
    print("Per teacher:")
    table(grouped("teacher"))
    print(f"total: {len(unique)} rollouts, {sum(passed.values())} passing, "
          f"{len(kept)} kept ({dropped_dupes} near-duplicates dropped)")
    for reason, count in reasons.most_common():
        print(f"  dropped {count}: {reason}")
    print(f"train {len(train)} / val {len(val)} "
          f"({len(val_templates)} of {len(train_templates | val_templates)} templates in val)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

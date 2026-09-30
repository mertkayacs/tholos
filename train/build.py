"""Filter rollouts, deduplicate, split by template, write SFT files.

Usage:
  python train/build.py --rollouts rollouts.jsonl --out-dir /kaggle/working
  python train/build.py --rollouts rollouts.jsonl --scenarios scenarios.jsonl --regrade

Keeps a rollout when: it passed the semantic audit, it has zero invalid outputs, no call repeats
back to back, no call appears more than twice, every thought is at most 240
characters, and it ends with finish. Near-duplicate trajectories inside a
template are dropped. The val split gets whole templates, about 8 percent of
kept trajectories.

--regrade first recomputes the training-check failures of each rollout with the current
checks.py, using the scenario with the same id from the --scenarios files. Runtime failures
stay as recorded. Use it for rollouts generated before a grading fix. Ids repeat across
scenario file versions, so a scenario whose template, category or trigger text differs from
its rollout is an error.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from checks import failures

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
    if result.get("infra_failed"):
        return False, "infrastructure failure"
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


def load_scenarios(paths):
    """Map scenario id to scenario across JSONL files. An id may appear only once."""
    scenarios = {}
    for path in paths:
        with open(path, encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    scenario = json.loads(line)
                    if scenario["id"] in scenarios:
                        raise ValueError(f"duplicate scenario id {scenario['id']}")
                    scenarios[scenario["id"]] = scenario
    return scenarios


def trigger_text(trigger):
    """The part of the trigger that the runtime copies verbatim into the first user message."""
    kind = trigger["kind"]
    if kind == "task":
        return f"{trigger['title']}\n{trigger.get('details', '')}"
    return trigger[{"message": "text", "schedule": "prompt", "follow_up": "note"}[kind]]


def matches_rollout(scenario, result):
    """Is this the scenario the rollout ran on?

    Ids are positional and repeat across scenario file versions, so compare what the rollout
    records: its template, its category and the trigger text in its first user message.
    """
    opening = next((m["content"] for m in result["messages"] if m["role"] == "user"), "")
    return (scenario["template"] == result["template"]
            and scenario["category"] == result["category"]
            and trigger_text(scenario["trigger"]) in opening)


def regrade(results, scenarios):
    """Recompute training-check failures with the current checks.

    Runtime failures (entries with a "type" key) come from the bench's final-state evaluation
    and cannot be recomputed from the messages, so they stay as recorded. Returns the new
    records and, per template, how many went from fail to pass (gained) and from pass to
    fail (lost).
    """
    regraded, gained, lost = [], Counter(), Counter()
    for result in results:
        scenario = scenarios.get(result["id"])
        if scenario is None:
            raise ValueError(f"no scenario for rollout {result['id']}")
        if not matches_rollout(scenario, result):
            raise ValueError(f"scenario {result['id']} differs from its rollout in template, "
                             "category or trigger text")
        failed = [entry for entry in result["failed_assertions"] if "type" in entry]
        failed += failures(scenario, result["messages"])
        passed = not failed
        if passed and not result["passed"]:
            gained[result["template"]] += 1
        elif result["passed"] and not passed:
            lost[result["template"]] += 1
        regraded.append({**result, "passed": passed, "failed_assertions": failed,
                         "semantic_checked": bool(scenario.get("checks"))})
    return regraded, gained, lost


def print_regrade(total, gained, lost):
    if gained or lost:
        print("Regrade, records changed per template:")
        print(f"{'':<20} {'fail->pass':>10} {'pass->fail':>10}")
        for name in sorted(set(gained) | set(lost)):
            print(f"{name:<20} {gained[name]:>10} {lost[name]:>10}")
    print(f"regraded {total} rollouts: {sum(gained.values())} fail->pass, "
          f"{sum(lost.values())} pass->fail")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build SFT train/val files from rollouts.")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--val-frac", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", nargs="+",
                        help="scenario JSONL files holding each rollout's checks, for --regrade")
    parser.add_argument("--regrade", action="store_true",
                        help="recompute training-check failures with the current checks")
    args = parser.parse_args(argv)
    if bool(args.scenarios) != args.regrade:
        parser.error("--scenarios and --regrade must be used together")

    results = [json.loads(line) for line in
               Path(args.rollouts).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_id = {}
    for result in results:
        previous = by_id.get(result["id"])
        if previous is None or previous.get("infra_failed"):
            by_id[result["id"]] = result
    infra_failed = sum(bool(result.get("infra_failed")) for result in by_id.values())
    unique = [result for result in by_id.values() if not result.get("infra_failed")]
    if args.regrade:
        try:
            unique, gained, lost = regrade(unique, load_scenarios(args.scenarios))
        except ValueError as exc:
            parser.error(str(exc))
        print_regrade(len(unique), gained, lost)

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
    print(f"infrastructure failures: {infra_failed} (excluded)")
    print(f"total: {len(unique)} rollouts, {sum(passed.values())} passing, "
          f"{len(kept)} kept ({dropped_dupes} near-duplicates dropped)")
    for reason, count in reasons.most_common():
        print(f"  dropped {count}: {reason}")
    print(f"train {len(train)} / val {len(val)} "
          f"({len(val_templates)} of {len(train_templates | val_templates)} templates in val)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

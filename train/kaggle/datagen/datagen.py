# Tholos-2B training data generation on Kaggle 2x T4.
#
# Serving method proven by kaggle/probe/probe.py (v5): the llama_cpp_binaries
# CUDA wheel provides a llama-server that runs on Kaggle (the official
# llama.cpp release needs glibc 2.38; Ollama sorts JSON keys, so neither is
# usable). Each kernel version starts with an empty /kaggle/working. To resume,
# attach the previous kernel's output as a kernel source. At startup we restore
# its JSONL bundle from /kaggle/input, retaining the saved scenarios and IDs.
# DEADLINE_HOURS defaults to 10.5 hours from kernel start, including setup.
# Stages drain in-flight items and flush before building the available rollouts.
# A session killed at 12 hours can lose its outputs; save a completed version
# and attach those outputs explicitly for the next run.
# Pilot kernel (tholos-datagen-pilot): N_SCENARIOS=300, DOMAINS_LIMIT=40,
# PACKS_PER_DOMAIN=2. Set these environment variables or edit the constants.
# Attach scenarios_kaggle.jsonl to run rollouts only, preserving hosted scenario IDs.

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

KERNEL_STARTED_AT = time.time()

WHEEL = ("https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.138.0/"
         "llama_cpp_binaries-0.138.0+cu124-py3-none-linux_x86_64.whl")
GGUF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
GGUF_FILE = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MODEL = "teacher"
BASE_URL = "http://127.0.0.1:8080/v1"
WORK = "/kaggle/working"
TRAIN = None  # train/ scripts from the attached dataset, set in find_inputs()

N_SCENARIOS = int(os.environ.get("N_SCENARIOS", "10000"))
PACKS_PER_DOMAIN = int(os.environ.get("PACKS_PER_DOMAIN", "4"))
_domains_limit = os.environ.get("DOMAINS_LIMIT", "None")
DOMAINS_LIMIT = None if _domains_limit.lower() in {"none", ""} else int(_domains_limit)
PHRASING_FRACTION = float(os.environ.get("PHRASING_FRACTION", "0.4"))
DEADLINE_HOURS = float(os.environ.get("DEADLINE_HOURS", "10.5"))
WORKERS = int(os.environ.get("WORKERS", "8"))
TEACHER = os.environ.get("TEACHER", "local")


def count_items(path):
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(bool(line.strip()) for line in file)


def copy_inputs(input_root="/kaggle/input", work=WORK):
    """Restore one coherent output bundle, preferring the most completed rollouts."""
    names = {"packs.jsonl", "scenarios.jsonl", "scenarios_phrased.jsonl", "rollouts.jsonl",
             "scenarios_kaggle.jsonl"}
    parents = {p.parent for p in Path(input_root).rglob("*.jsonl") if p.name in names}
    if not parents:
        return []
    # A shard run must not restore rollouts from an unrelated full-pipeline bundle.
    shard_parents = {p for p in parents if (p / "scenarios_kaggle.jsonl").is_file()}
    if shard_parents:
        parents = shard_parents

    def rank(directory):
        return tuple(count_items(directory / name)
                     for name in ("rollouts.jsonl", "scenarios_kaggle.jsonl",
                                  "scenarios.jsonl", "packs.jsonl"))

    # Mixing stages from different versions can associate saved IDs with changed tasks.
    source = max(sorted(parents), key=rank)
    destination = Path(work)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.glob("*.jsonl")):
        target = destination / path.name
        if target.is_file() and target.stat().st_size:
            continue
        shutil.copy2(path, target)
        copied.append(path.name)
    print(f"restored {source}: {copied}", flush=True)
    return copied


def rollout_metrics(path, seconds, skip=0):
    categories = {}
    steps, tokens, total = 0, 0, 0
    if Path(path).is_file():
        with open(path, encoding="utf-8") as file:
            for index, line in enumerate(line for line in file if line.strip()):
                if index < skip:
                    continue
                item = json.loads(line)
                count = categories.setdefault(item["category"], {"items": 0, "passed": 0})
                count["items"] += 1
                count["passed"] += bool(item["passed"])
                steps += item["steps"]
                tokens += item["tokens"]["out"]
                total += 1
    for count in categories.values():
        count["pass_rate"] = count["passed"] / count["items"]
    return {"categories": categories, "mean_steps": steps / total if total else 0,
            "completion_tokens": tokens,
            "tokens_per_second": tokens / seconds if seconds > 0 else 0}


def sh(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{result.stdout[-2000:]}{result.stderr[-2000:]}", flush=True)
    if result.returncode != 0:
        sys.exit(f"command failed: {cmd}")


def find_inputs():
    """Locate the tholos wheel and the train/ scripts in attached datasets."""
    global TRAIN
    wheels = glob.glob("/kaggle/input/**/tholos*.whl", recursive=True)
    train_dirs = glob.glob("/kaggle/input/**/train", recursive=True)
    train_dirs = [d for d in train_dirs if os.path.isfile(os.path.join(d, "templates.py"))]
    if not train_dirs:
        sys.exit("attach a private dataset containing the tholos wheel and train/")
    TRAIN = train_dirs[0]
    if wheels:
        sh(f"pip install -q '{wheels[0]}'")
    else:
        sh("pip install -q /kaggle/input/tholos-src")


def start_server():
    sh(f"pip install -q '{WHEEL}' huggingface_hub hf_transfer 2>&1 | tail -3")
    import llama_cpp_binaries

    server = llama_cpp_binaries.get_binary_path()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir="/tmp/models")
    print(f"gguf download {time.time() - t0:.0f}s", flush=True)
    log = open("/tmp/server.log", "w")  # noqa: SIM115 - outlives the call
    process = subprocess.Popen(
        [server, "-m", gguf, "-ngl", "99", "-c", "65536", "--parallel", "8",
         "--jinja", "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "--port", "8080"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(900):
        try:
            if json.load(urllib.request.urlopen(
                    "http://127.0.0.1:8080/health", timeout=2)).get("status") == "ok":
                print("server up", flush=True)
                return process
        except Exception:
            time.sleep(1)
    sh("tail -60 /tmp/server.log")
    sys.exit("server never became healthy")


def stage(name, args, inputs=(), outputs=(), skip=False):
    print(f"=== {name} ===", flush=True)
    before = sum(count_items(path) for path in outputs)
    incoming = sum(count_items(path) for path in inputs)
    start = time.monotonic()
    result = None if skip else subprocess.run([sys.executable] + args, check=False)
    seconds = time.monotonic() - start
    outgoing = sum(count_items(path) for path in outputs)
    metrics = {"stage": name, "wall_seconds": seconds, "items_in": incoming,
               "items_out": outgoing, "items_existing": before,
               "items_new": max(0, outgoing - before), "skipped": skip,
               "returncode": result.returncode if result else 0}
    if name == "rollouts":
        metrics.update(rollout_metrics(outputs[0], seconds, skip=before))
    print("STAGE " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def main():
    deadline = KERNEL_STARTED_AT + DEADLINE_HOURS * 3600
    Path(WORK).mkdir(parents=True, exist_ok=True)
    restored = copy_inputs(work=WORK)
    stages = []
    server = None
    packs = f"{WORK}/packs.jsonl"
    scenarios = f"{WORK}/scenarios.jsonl"
    phrased = f"{WORK}/scenarios_phrased.jsonl"
    rollouts = f"{WORK}/rollouts.jsonl"
    kaggle_scenarios = f"{WORK}/scenarios_kaggle.jsonl"
    rollout_only = Path(kaggle_scenarios).is_file()
    common = ["--base-url", BASE_URL, "--model", MODEL, "--deadline", str(deadline),
              "--teacher", TEACHER]

    def run_stage(*args, **kwargs):
        result = stage(*args, **kwargs)
        stages.append(result)
        if result["returncode"]:
            if not any(count_items(path) for path in kwargs.get("outputs", ())):
                raise RuntimeError(f"stage {result['stage']} failed")
            print(f"stage {result['stage']} failed; continuing with available output", flush=True)

    try:
        find_inputs()
        if time.time() < deadline:
            server = start_server()
        if rollout_only:
            rollout_source = kaggle_scenarios
            print(f"rollout-only input: {rollout_source}", flush=True)
        else:
            domains = Path(TRAIN) / "domains.txt"
            limited_domains = Path(WORK) / "domains.txt"
            domain_lines = [line for line in domains.read_text().splitlines() if line.strip()]
            if DOMAINS_LIMIT is not None:
                domain_lines = domain_lines[:DOMAINS_LIMIT]
            limited_domains.write_text("\n".join(domain_lines) + "\n", encoding="utf-8")
            run_stage("packs", [f"{TRAIN}/packs.py", *common,
                                "--domains", str(limited_domains),
                                "--per-domain", str(PACKS_PER_DOMAIN),
                                "--workers", str(WORKERS), "--out", packs],
                      inputs=[limited_domains], outputs=[packs])
            # Keep persisted scenario IDs stable when new packs arrive on a later run.
            reuse = count_items(scenarios) > 0
            if not reuse:
                Path(scenarios).touch()
            run_stage("scenarios", [f"{TRAIN}/templates.py", "--packs", packs,
                                    "--n", str(N_SCENARIOS), "--out", scenarios, "--seed", "1"],
                      inputs=[packs], outputs=[scenarios],
                      skip=reuse or not count_items(packs) or time.time() >= deadline)
            run_stage("phrasing", [f"{TRAIN}/phrasing.py", "--scenarios", scenarios, *common,
                                   "--fraction", str(PHRASING_FRACTION), "--seed", "1",
                                   "--workers", str(WORKERS), "--out", phrased],
                      inputs=[scenarios], outputs=[phrased])
            rollout_source = phrased
        run_stage("rollouts", [f"{TRAIN}/rollout.py", "--scenarios", rollout_source, *common,
                               "--workers", str(WORKERS), "--out", rollouts,
                               "--temperature", "0.2"], inputs=[rollout_source], outputs=[rollouts])
    finally:
        try:
            # Build runs even when the deadline or an earlier stage ends generation.
            Path(rollouts).touch(exist_ok=True)
            if TRAIN is None:
                raise RuntimeError("train scripts unavailable for build")
            run_stage("build", [f"{TRAIN}/build.py", "--rollouts", rollouts,
                                "--out-dir", WORK], inputs=[rollouts],
                      outputs=[f"{WORK}/sft_train.jsonl", f"{WORK}/sft_val.jsonl"])
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            summary = {"wall_seconds": time.time() - KERNEL_STARTED_AT,
                       "deadline": deadline, "restored": restored, "stages": stages,
                       "rollout_only": rollout_only,
                       "config": {"N_SCENARIOS": N_SCENARIOS,
                                  "PACKS_PER_DOMAIN": PACKS_PER_DOMAIN,
                                  "DOMAINS_LIMIT": DOMAINS_LIMIT,
                                  "PHRASING_FRACTION": PHRASING_FRACTION,
                                  "DEADLINE_HOURS": DEADLINE_HOURS,
                                  "TEACHER": TEACHER}}
            print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

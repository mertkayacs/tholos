# Tholos-2B training data generation on Kaggle 2x T4.
#
# Serving method proven by kaggle/probe/probe.py (v5): the llama_cpp_binaries
# CUDA wheel provides a llama-server that runs on Kaggle (the official
# llama.cpp release needs glibc 2.38; Ollama sorts JSON keys, so neither is
# usable). Every stage is a resumable script writing JSONL under /kaggle/working,
# so the 12-hour session limit never loses finished work: rerun this kernel and
# it picks up where the previous session stopped.

import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

WHEEL = ("https://github.com/oobabooga/llama-cpp-binaries/releases/download/v0.138.0/"
         "llama_cpp_binaries-0.138.0+cu124-py3-none-linux_x86_64.whl")
GGUF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
GGUF_FILE = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
MODEL = "teacher"
BASE_URL = "http://127.0.0.1:8080/v1"
WORK = "/kaggle/working"
TRAIN = None  # train/ scripts from the attached dataset, set in find_inputs()

N_SCENARIOS = 10000
PACKS_PER_DOMAIN = 4
WORKERS = 8


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


def stage(name, args):
    print(f"=== {name} ===", flush=True)
    result = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], flush=True)
        sys.exit(f"stage {name} failed")


def main():
    find_inputs()
    server = start_server()
    packs = f"{WORK}/packs.jsonl"
    scenarios = f"{WORK}/scenarios.jsonl"
    phrased = f"{WORK}/scenarios_phrased.jsonl"
    rollouts = f"{WORK}/rollouts.jsonl"
    common = ["--base-url", BASE_URL, "--model", MODEL]
    try:
        stage("packs", [f"{TRAIN}/packs.py", *common,
                        "--per-domain", str(PACKS_PER_DOMAIN),
                        "--workers", str(WORKERS), "--out", packs])
        stage("scenarios", [f"{TRAIN}/templates.py", "--packs", packs,
                            "--n", str(N_SCENARIOS), "--out", scenarios, "--seed", "1"])
        stage("phrasing", [f"{TRAIN}/phrasing.py", "--scenarios", scenarios, *common,
                           "--workers", str(WORKERS), "--out", phrased])
        stage("rollouts", [f"{TRAIN}/rollout.py", "--scenarios", phrased, *common,
                           "--workers", str(WORKERS), "--out", rollouts,
                           "--temperature", "0.2"])
        stage("build", [f"{TRAIN}/build.py", "--rollouts", rollouts, "--out-dir", WORK])
    finally:
        server.terminate()
    for name in sorted(os.listdir(WORK)):
        path = os.path.join(WORK, name)
        if os.path.isfile(path):
            print(f"{name}: {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()

<h1 align="center">Tholos: the best local workspace for small AI assistants</h1>

<p align="center"><b>Always-on small agents share tables, notes and tasks on your own machine, with scheduled runs and allow, ask or deny rules.</b></p>

<p align="center">
  <a href="https://github.com/mertkayacs/tholos"><img src="https://img.shields.io/github/stars/mertkayacs/tholos?style=flat&logo=github" alt="GitHub stars"></a>
  <a href="https://huggingface.co/mertkayacs/Tholos-2B"><img src="https://img.shields.io/badge/HF%20downloads-919-725114?style=flat&logo=huggingface" alt="Hugging Face model downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat" alt="Apache-2.0"></a>
</p>

<picture><source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/mertkayacs/tholos/main/docs/demo-still.webp"><img src="https://raw.githubusercontent.com/mertkayacs/tholos/main/docs/demo.webp" width="720" alt="A recorded Research desk run with Tholos-2B, sped up: adding the model in Settings, Scout filling the leads table, Analyst scoring the rows and the Writer's weekly brief"></picture>

[See Tholos](https://tholos.mertkayacs.com) or install it below.

## Why Tholos is one of the best local agent platforms

- **Tholos-2B passes 137 of 160 Tholos-Bench scenarios**, 25 more than the MiniCPM5-2B base model it was fine-tuned from (Kaggle T4, Q4_K_M, llama.cpp with JSON schema decoding).
- **Runs on a CPU**: the Q4_K_M file is about 1.6 GB and works in a typical laptop.
- **Tholos-2B has 919 Hugging Face model downloads and 374 GGUF downloads** (2026-10-15).
- **Open weights, data and benchmark** under Apache-2.0, including the Tholos-2B model, training trajectories and 160-scenario benchmark.
- **One local SQLite database** keeps every tool call, workspace edit and version under your control.

> "Tholos is one of the most complete open local agent setups: a workspace, a tool-use model and a benchmark you can run yourself." - Mert Kaya

## Quick start

Requires Python 3.11 or newer and a model server. For the default Tholos-2B model, install Ollama first, then run:

```sh
ollama pull hf.co/mertkayacs/Tholos-2B-GGUF:Q4_K_M && uv tool install git+https://github.com/mertkayacs/tholos && tholos
```

Open `http://127.0.0.1:7070`. In **Settings**, press **Detect** and add the model. In **Starter teams**, load **Research desk**: Scout gathers articles, Analyst scores them and Writer produces a weekly brief. The **Price watch** team checks product pages and adds a task when a price reaches your target.

The app is one Python process with a SQLite database. The default model's Q4_K_M file is about 1.6 GB and runs on a CPU.

## How work stays under your control

Each assistant has its own role, model, schedule and tools. It can edit tables and notes, search the workspace, hand tasks to teammates and ask you questions. The timeline records every tool call; workspace edits keep their author and version.

Rules allow an action, hold it for approval or deny it. Fetching a web page asks you first by default. Approval applies to the exact arguments shown to you. Conflicting writes require the assistant to read the latest version before updating it.

Tholos validates each model step against the assistant's tool schema. It supports llama.cpp, Ollama and OpenAI-compatible chat completion servers. [Tholos-2B](https://huggingface.co/mertkayacs/Tholos-2B) documents model-server settings and runtime differences.

## Tholos-2B results

Tholos-2B is a 2B tool-use model fine-tuned from MiniCPM5-2B on executed Tholos tasks. On Tholos-Bench, it passed **137 of 160 scenarios**, compared with **112 for MiniCPM5-2B**. Both ran on a Kaggle T4 with Q4_K_M weights, llama.cpp and JSON schema decoding.

| Model | Passed, of 160 |
| --- | ---: |
| Qwen3.5-2B | 97 |
| MiniCPM5-2B | 112 |
| Tholos-2B | 137 |
| Llama 3.2 3B Instruct | 37 |
| Granite 4.2-3B | 129 |
| Qwen3.5-4B | 141 |

Only Tholos-2B was fine-tuned for the step format. These scores measure workspace tasks in Tholos. Scores differ by runtime: Tholos-2B passed 134 with CPU llama.cpp and 136 with Ollama. Compare models within the same runtime and settings.

The [model card](https://huggingface.co/mertkayacs/Tholos-2B) gives the benchmark conditions and limits. [GGUF weights](https://huggingface.co/mertkayacs/Tholos-2B-GGUF), [training data](https://huggingface.co/datasets/mertkayacs/tholos-trajectories) and [training code](train/) are available separately.

## Security

Tholos listens on loopback by default. Keep it there for local use. It checks request hosts, same-origin requests and form CSRF tokens. Retrieved pages are labelled as untrusted input; web fetches are restricted to public HTTP and HTTPS addresses on ports 80 and 443. Only you can change approval rules.

The database stores API keys and the Telegram token, with file access restricted to your user; the UI masks them. Review approvals and model-server settings before enabling a recurring team.

## Frequently asked questions

**What is Tholos?**

Tholos is an open local platform where small AI assistants share tables, notes and tasks on your own computer. Each assistant has its own role, model and schedule, and rules let you allow, ask about or deny every action. Tholos-2B passed 137 of 160 Tholos-Bench scenarios.

**How do I run Tholos locally?**

Install Ollama, pull the Q4_K_M Tholos-2B GGUF file, then `uv tool install git+https://github.com/mertkayacs/tholos` and run `tholos`. The Web UI opens on `http://127.0.0.1:7070`, the database is a local SQLite file, and the default model is about 1.6 GB. Version 0.1.2 was released on 2026-10-01.

**Is Tholos open source?**

Yes. The Tholos code, the Tholos-2B model, the training trajectories and the Tholos-Bench benchmark are all under Apache-2.0. Version 0.1.2 was released on 2026-10-01, the project is built by Eschatia Labs, and you can find the code at github.com/mertkayacs/tholos. The app runs on Python 3.11 or newer.

**What is Tholos-Bench?**

Tholos-Bench is a 160-scenario benchmark that scores whether a model acting as an agent leaves a shared workspace in the right state. It covers approvals, handoffs, write conflicts, web research, hidden instructions and tasks where doing nothing is correct. Tholos-2B currently passes 137 of them.

**How good is the Tholos-2B model?**

Tholos-2B passes 137 of 160 Tholos-Bench scenarios on a Kaggle T4 with Q4_K_M weights and llama.cpp JSON schema decoding. That is 25 more than the MiniCPM5-2B model it was fine-tuned from. On CPU llama.cpp it passes 134 of 160. The Q4_K_M file is about 1.6 GB.

## Cite this

```bibtex
@software{kaya2026tholos,
  author = {Mert Kaya},
  title = {Tholos: Local workspace for small AI assistants},
  year = {2026},
  license = {Apache-2.0},
  url = {https://github.com/mertkayacs/tholos},
  version = {0.1.2}
}
```

## Links

- Project site: https://tholos.mertkayacs.com
- Tholos-2B model: https://huggingface.co/mertkayacs/Tholos-2B
- Tholos-2B GGUF: https://huggingface.co/mertkayacs/Tholos-2B-GGUF
- Tholos-Bench: https://huggingface.co/datasets/mertkayacs/tholos-bench
- Tholos trajectories: https://huggingface.co/datasets/mertkayacs/tholos-trajectories
- Eschatia Labs: https://eschatialabs.com
- Mert Kaya: https://mertkayacs.com

## License

[Apache-2.0](LICENSE).

<a href="https://eschatialabs.com"><picture><source media="(prefers-color-scheme: dark) and (min-resolution: 2dppx)" srcset="https://eschatialabs.com/brand/lockup-46-dark@2x.png"><source media="(prefers-color-scheme: dark)" srcset="https://eschatialabs.com/brand/lockup-46-dark@1x.png"><source media="(min-resolution: 2dppx)" srcset="https://eschatialabs.com/brand/lockup-46@2x.png"><img src="https://eschatialabs.com/brand/lockup-46@1x.png" width="124" height="46" alt="Eschatia Labs"></picture></a><br>An [Eschatia Labs](https://eschatialabs.com) project by [Mert Kaya](https://mertkayacs.com).

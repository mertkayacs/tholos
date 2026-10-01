<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-ivory.svg">
    <img src="docs/logo-navy.svg" width="72" alt="">
  </picture>
</p>
<h1 align="center">Tholos</h1>
<p align="center">Always-on agents that live on your own computer.</p>

Tholos gives a small team of AI agents one shared workspace of tables, notes and a task board. The agents
wake on a schedule, hand work to each other and stop to ask you before anything sensitive. The whole app is
one Python process and one SQLite file, and its default model, Tholos-2B, is a 1.6 GB file that runs on an
ordinary CPU.

## Quick start

```sh
ollama pull hf.co/mertkayacs/Tholos-2B-GGUF:Q4_K_M
uv tool install git+https://github.com/mertkayacs/tholos
tholos
```

Open http://127.0.0.1:7070, press Detect in Settings, add the model it finds, then load the Research desk
team from Starter teams. Its Scout checks Hacker News, arXiv and Hugging Face Papers every two hours and fills
a leads table. The Analyst scores what comes in, and the Writer turns the best items into a brief on Friday
afternoon. The second starter team, Price watch, checks product pages each morning and adds a task for you
when a price falls to your target.

## How it works

Each agent has a role, a model, a schedule and a short list of tools, picked from thirteen. The tools let an agent
create, read and edit tables, read and write notes, search the workspace, hand a task to a teammate, fetch a web
page, ask you a question, remember a fact, schedule a follow-up and finish with a summary. Every call appears in
the run timeline. Every change to a table or note keeps its author and version. A write to a row or note that
changed since the agent last read it is refused until the agent reads the new value. A run cut off by a crash is
retried after the restart, and each write lands once.

Rules decide whether a call goes ahead, waits for your approval or is refused. A rule can cover one agent or
all of them, and optionally one target such as a host name. Fetching a web page asks you first by default. An
approval shows the exact action, and you can answer in the browser or on Telegram.

## Models

Tholos is built for small local models, so the app does the checking. Each model turn is one JSON step: a short
thought, a tool name and its arguments. Tholos validates the step against a schema built from the agent's tools,
and a step that fails gets one retry with the validation error attached. Any OpenAI-compatible chat completions
endpoint works, each agent picks its own model, and Tholos has run against llama.cpp's llama-server, Ollama and
hosted APIs.

With llama.cpp the JSON schema also constrains decoding, so a 2B model can only write a well-formed call to one
of the agent's tools. Ollama reorders schema keys, and Tholos-2B writes its thought before the tool call. So with
Ollama, Tholos requests JSON mode, sends `reasoning_effort: "none"` to keep thinking off, validates every step
against the schema and fills optional arguments the model left out with null. Give either server a 16,384-token
context, as our benchmark runs do. Ollama defaults to 4,096, and the Tholos-2B GGUF repo carries a `params` file
that raises it. Each model profile also carries a timeout, 120 seconds by default. On a slow machine, raise it
under Settings, Models, in the Timeout (seconds) field.

Tholos-2B is MiniCPM5-2B fine-tuned on Tholos work (pipeline and notebook in `train/`). The weights are on
Hugging Face as [Tholos-2B](https://huggingface.co/mertkayacs/Tholos-2B) and
[Tholos-2B-GGUF](https://huggingface.co/mertkayacs/Tholos-2B-GGUF), and the public part of the training data is
[tholos-trajectories](https://huggingface.co/datasets/mertkayacs/tholos-trajectories). On 8 shared vCPUs with 4
threads, its Q4_K_M file reads 84 tokens a second and writes 24 (llama-bench b11263, measured while other jobs
shared the machine). In the same pass, Granite 4.2-3B wrote 18 tokens a second and Qwen3.5-4B wrote 11.

## Tholos-Bench

160 scenarios in 14 categories, each scored on the final state of the workspace. They run from table edits and
handoffs to approvals, write conflicts, web research on local fixture pages, prompt injection and tasks where the
right move is to do nothing. The clock is pinned to a fixed UTC time, so prompts repeat exactly. Run it with
`tholos bench --base-url <server>/v1 --model <name>` against any OpenAI-compatible server.

The table sets Tholos-2B beside models of its own size and larger ones, with each file's size next to its score.
Injection counts the 12 scenarios where a page, a table cell or a note carries instructions the agent should ignore.

| Model, Q4_K_M | File | Passed, of 160 | Injection, of 12 |
|---|---:|---:|---:|
| Qwen3.5-2B | 1.28 GB | 97 | 4 |
| MiniCPM5-2B (base of Tholos-2B) | 1.56 GB | 112 | 8 |
| **Tholos-2B** | 1.56 GB | **137** | **10** |
| Llama 3.2 3B Instruct | 2.02 GB | 37 | 1 |
| Granite 4.2-3B | 2.24 GB | 129 | 7 |
| Qwen3.5-4B | 2.74 GB | 141 | 6 |

Tholos-2B passes 137, 25 more than MiniCPM5-2B, the model it was trained from, and more than both 3B models.
Qwen3.5-4B passes 141 with a file 1.75 times the size. On the 12 injection scenarios Tholos-2B passes 10, the
highest count in the table.

Every row ran on one Kaggle T4 with llama.cpp (CUDA build fdf5818), a 16,384-token context, JSON schema decoding,
temperature 0 and the pinned clock. Only Tholos-2B was fine-tuned for the step format. LFM2.5-2.6B (1.67 GB) has no
score: it returned empty replies for every step under `json_schema` on this build.

We measure the benchmark three ways, and one file scores differently on each. CPU llama.cpp with the JSON schema is
the reproducible path: on the base model, a second run gave identical transcripts for all 160 scenarios. Ollama with
the GGUF repo's template is the realistic default for most users. The Kaggle GPU table above is the cross-model
comparison. MiniCPM5-2B passes 117, 96 and 112 of 160 on the three, and Tholos-2B passes 134, 136 and 137, so compare
scores inside one path.

## Docker

```sh
docker run -d --name tholos -p 127.0.0.1:7070:7070 -v tholos:/data \
  --add-host=host.docker.internal:host-gateway ghcr.io/mertkayacs/tholos
docker logs tholos
```

The log shows the access token. The image runs as an unprivileged user and keeps its database in the `/data`
volume. To use a model server on the host, add it in Settings with a base URL such as
`http://host.docker.internal:11434/v1`. Ollama listens on 127.0.0.1 by default, so set `OLLAMA_HOST` to an
address the container can reach first.

## Security

Tholos listens on 127.0.0.1 and refuses requests addressed to any other host name. Bound anywhere else, it asks
for the access token it prints at start. Forms carry same-origin and CSRF checks. Pages the agents read reach
the model labelled as untrusted data. Fetches go only to public addresses over http and https on ports 80 and
443. An approval is tied to a hash of the exact arguments, and only you can change the rules. API keys and the
Telegram token sit in the database, which only your user can read, and appear masked in the UI.

## License

Apache-2.0. The name comes from the Tholos, the round house in the Athenian agora. Aristotle writes that the
council's chairman kept the keys of the temples where the money and documents of the state were lodged, along with
the state seal, and had to stay in the Round-house (Athenian Constitution 44.1).

# DaCode — The Ultimate Autonomous Engineering Rig

> A 5-layer coding-agent rig built from the blueprint *Ultimate Autonomous
> Engineering Rig* — combining **Pi**, **Google Harness²**, **pstack**, an
> **Obsidian vault memory**, and an **isolated git sandbox** into one CLI,
> with an optional **Jev** (TypeSafe System One) judgment layer on Compose.

DaCode does not just run a prompt once. It **proposes** several ways to run the
task, **probes** each one in its own git worktree, lets an **adversarial critic**
gate the result against real tests, and **writes the lesson back** to the vault
so the next run starts smarter. No ground-truth labels required.

## Contents

[The 5-layer stack](#the-5-layer-stack) · [Install](#install) ·
[Usage](#usage) · [Memory backends](#memory-backends) ·
[How a run works](#how-a-run-works) · [Isolation](#isolation) ·
[Tests](#tests) · [FAQ](#faq) · [Going further](#going-further) ·
[Design principles](#design-principles) ·
[Working on DaCode itself](#working-on-dacode-itself) ·
[Status / roadmap](#status--roadmap)

---

## The 5-layer stack

| # | Layer | Module | Job |
|---|-------|--------|-----|
| 1 | **Vault Memory** | `dacode/memory/` | retrieve prior lessons → seed the harness; ingest the winner back as a note |
| 2 | **Harness² Optimizer** | `dacode/harness/` | Propose → Probe → Compose recursion over *harness* variants |
| 3 | **Executor** | `dacode/executor/` | Pi coding agent (or a script / test double) does the change |
| 4 | **pstack Discipline** | `dacode/discipline/` | AST checks for weak tests + a real `pytest` run as the gate |
| 5 | **Git Sandbox** | `dacode/sandbox/` | one isolated `git worktree` per trial, checkpoint + rollback |
| + | **Jev Judge** *(optional)* | `dacode/judgment.py` | a TypeSafe **System One** decision model picks the Compose winner — `--judge jev` |

```
        ┌───────────────────────────────────────────────┐
        │ 1. Vault Memory   retrieve(task) → snippets    │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │ 2. Harness²   propose(n) → probe → compose     │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │ 5. Git Sandbox  worktree per trial (isolated)  │
        │ 3. Executor     Pi / script applies the change │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │ 4. pstack Critic   static scan + real tests    │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │ 1. Vault Memory   ingest(lesson) if it passed  │
        └───────────────────────────────────────────────┘
```

---

## Install

```bash
# from the repo root
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Runtime needs: `git`, [`pi`](docs/installing-pi.md) on `PATH`, and a provider key —
OpenRouter (`OPENROUTER_API_KEY`) or Z.ai (`GLM_API_KEY`, see
[AI providers](#ai-providers)).

Pi (`--executor pi`) is a separate Node.js program — install it before the first
run: **[docs/installing-pi.md](docs/installing-pi.md)** (install, authentication
under the sandbox, and the 127 / 402 triage table).

Optional: a TypeSafe key in `JEV_API_KEY` for `--judge jev` — the System One
decision model that picks the Compose winner (fast, typed answers with
probabilities; it never issues pass/fail verdicts).

---

## Usage

```bash
# show layer health
.venv/bin/dacode --repo . layers

# watch the rig work (live, read-only — the log is the only thing it touches)
.venv/bin/dacode watch                      # follow; `q` quits; --once renders one frame

# run a task through the full rig (Pi as executor, DeepSeek v4.1 Flash)
OPENROUTER_API_KEY=... .venv/bin/dacode run "add a slugify() helper with tests" --budget 2

# same run, plus a Jev (System One) judge picking the Compose winner
OPENROUTER_API_KEY=... JEV_API_KEY=... .venv/bin/dacode run "..." --budget 2 --judge jev

# deterministic run without an LLM (good for demos / CI)
.venv/bin/dacode run "..." --llm stub --executor script --script-command '...'

# memory: keyword + DaCode's own embeddings (see "Memory backends")
.venv/bin/dacode memory index                       # embed the vault (incremental)
.venv/bin/dacode memory backends                    # which backends are usable
.venv/bin/dacode memory search "harness2 propose probe compose"
.venv/bin/dacode memory search "isolate agent worktrees" --backend local

# score Jev's Compose picks against recorded gate-verified winners
.venv/bin/dacode judge-eval --limit 10

# query the local SQLite history (runs, variants, picks)
.venv/bin/dacode state stats
.venv/bin/dacode state search "docstring sandbox"
.venv/bin/dacode state failures --task slugify   # don't retry these blindly

# inspect / clean trial worktrees
.venv/bin/dacode sandbox list
.venv/bin/dacode sandbox clean
```

### Engine configuration

| Flag | Values | Default |
|------|--------|---------|
| `--llm` | `stub` \| `local` \| any provider preset (table below) | `openrouter` |
| `--model` | any model id | `deepseek/deepseek-v4.1-flash` |
| `--executor` | `pi` \| `script` | `pi` |
| `--executor-provider` | any provider preset — which key the Pi executor authenticates with | `openrouter` |
| `--judge` | `jev` (unset = off) | off |
| `--escalate-models` | comma-separated model ladder, e.g. `claude-sonnet-4-5,gpt-4o` | none |

Environment overrides: `DACODE_REPO`, `DACODE_VAULT`, `DACODE_LLM`,
`DACODE_MODEL`, `DACODE_EXECUTOR`, `DACODE_BUDGET`, `DACODE_VARIANTS`,
`DACODE_JUDGE`, `DACODE_ESCALATE_MODELS`, `DACODE_RUNLOG` (`off` disables
the recorder; default `~/.local/state/dacode/runlog.jsonl`),
`DACODE_MEMORY_DB` (`off` disables the state store; default
`~/.local/state/dacode/memory.db`), `DACODE_SEMANTIC_BACKEND` (`auto` by
default — see "Memory backends"), `DACODE_EMBED_BASE_URL`, `DACODE_EMBED_MODEL`,
`DACODE_EMBED_API_KEY`, `DACODE_EMBED_BATCH`, `DACODE_EMBED_TIMEOUT`,
`DACODE_EMBED_DIMS` (request a specific vector length), `DACODE_EMBED_DB`, and
`DACODE_SEMANTIC=off` as the global kill switch.

### AI providers

| `--llm` | Key env var | Default model |
|---------|-------------|---------------|
| `openrouter` | `OPENROUTER_API_KEY` | `deepseek/deepseek-v4.1-flash` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `xai` | `XAI_API_KEY` | `grok-4` |
| `zai` | `GLM_API_KEY` (Pi also accepts `ZAI_API_KEY`) | `glm-5.3-flash` |

`zai` speaks the OpenAI-compatible shape at Z.ai's **coding endpoint** and is
verified end to end (propose + executor). Splitting propose and executor across
providers works: `--llm zai --executor-provider zai`, or mix families with
`--fleet`.

Keys resolve in order: an explicit `--api-key` / `api_key=` kwarg → the
provider's env var, populated from the **target repo's** `.env` by
`cli.load_env`. Either name of the z.ai key works everywhere (`GLM_API_KEY`
or Pi's own `ZAI_API_KEY`); when the Pi executor runs provider `zai`, DaCode
aliases one to the other so the agent always sees a key. DaCode deliberately never reads `~/.hermes/.env` or Pi's auth
store (`~/.pi/agent/auth.json`) — secrets stay with the project they belong
to. Every provider speaks the OpenAI-compatible shape, so adding one is a
single entry in `dacode/llm/providers.py`.

---

## Memory backends

Retrieval always runs the keyword/TF pass, then blends in a semantic backend
(60% cosine similarity + 40% normalized keyword score). **DaCode owns its
memory by default** — no other agent, database server, or service required.

| Backend | What it uses | When |
|---------|--------------|------|
| `local` (default) | DaCode's own vector store (`~/.local/state/dacode/embeddings.db`) built from any OpenAI-compatible `/v1/embeddings` endpoint | picked by `auto` as soon as vectors exist |
| `hermes` | an external index queried over a subprocess (Hermes' semantic-vault-search by default) — one bundled example of an adapter | `auto` fallback, or explicitly |
| keyword only | TF over the vault | no backend configured, or `DACODE_SEMANTIC_BACKEND=off` |

```bash
# point DaCode at any embedding endpoint — a local model needs no API key
export DACODE_EMBED_BASE_URL=http://127.0.0.1:11434/v1   # e.g. Ollama
export DACODE_EMBED_MODEL=bge-m3
.venv/bin/dacode memory index         # embed the vault (incremental by mtime)
.venv/bin/dacode memory backends      # what is usable right now
.venv/bin/dacode memory search "isolate agent worktrees" --backend local
```

`memory index` skips unchanged notes and prunes deleted ones; changing the
embedding model requires `--rebuild`, because the store refuses to mix models
(two models in one index produce plausible-looking garbage). Without an
embedder configured nothing breaks — retrieval is simply keyword-only.

Recipes (any OpenAI-compatible ``/v1/embeddings`` works):

| Endpoint | `DACODE_EMBED_BASE_URL` | `DACODE_EMBED_MODEL` | `DACODE_EMBED_DIMS` |
|---|---|---|---|
| Ollama (local, free) | `http://127.0.0.1:11434/v1` | `bge-m3` | leave unset |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-embedding-2` | `768` (default is 3072) |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | `1536` (optional) |

Keys come from `DACODE_EMBED_API_KEY`, falling back to `OPENAI_API_KEY`,
`GEMINI_API_KEY`, then `GOOGLE_API_KEY` — loaded from the target repo's `.env`
like every other secret. Gemini's OpenAI-compatible endpoint batches correctly
(`input: [a, b, c]` → three vectors) and honours `dimensions`, so a 583-note
vault indexes in one pass.

### Plugging in another agent's memory

A backend is any object with `available()` and
`search(query, top=8) -> list[SemanticHit]`. No registration needed:

```bash
export DACODE_SEMANTIC_BACKEND="my_package.memory:make_backend"  # called with the vault path
export DACODE_SEMANTIC_BACKEND=hermes                            # the bundled adapter
```

`dacode/memory/backends/hermes.py` is the worked example: it shells out to an
external search script and maps its JSON onto `SemanticHit`.

## How a run works

1. **Initialization & Context** — `VaultMemory.retrieve(task)` returns the most
   relevant notes; their titles + snippets become `memory_snippets` on the base
   harness variant.
2. **Experimentation Loop** — for each round, the optimizer asks the LLM for
   `n` genuinely different harness variants, then probes each one:
   * a fresh `git worktree` is created on branch `dacode/trial/<task>-rN-vM`
   * the executor applies the change inside it
   * the critic scans the tests and runs `pytest`
   * passing trials are checkpointed (committed) in their worktree
3. **Compose** — the best variant is selected (an optional Jev System One
   judgment, then LLM-assisted, with a deterministic fallback to the measured
   ranking). The judge only picks among variants the gate already scored —
   the pass/fail verdict stays with the critic. When the gate did not pass
   and budget remains, the judge also **triages** the next action: retry the
   same variants, propose fresh ones, escalate the executor up a model ladder
   (`--escalate-models`), or stop and flag the task for human review.
4. **Memory Ingestion** — if the winner passed, a `type: lesson` note is written
   into the vault under `Projects/DaCode/Lessons/`.

Losing trial worktrees are cleaned up automatically; the winner's worktree is
kept so you can inspect the diff.

---

Security posture (what is guaranteed, what is still open) lives in
[`SECURITY.md`](SECURITY.md). Read it before pointing the rig at a repository you
do not own.

## Isolation

A worktree gives a trial a private branch, not a private machine. `--isolate bwrap`
is the default (a host without bubblewrap fails closed; `--isolate none` opts out).
puts it in namespaces (bubblewrap, no daemon, no root):

```bash
dacode run "<task>" --executor pi --isolate bwrap --isolate-network \
  --test-command "${VIRTUAL_ENV}/bin/python -m pytest -q -o addopts="
```

- Host filesystem mounted **read-only**, the user's home replaced by an empty
  tmpfs, the trial workspace bound read-write — the only place a trial can write.
- **Network off by default** (`--unshare-net`). The Pi executor needs it to reach a
  model, so it opts in; the test gate never gets it.
- Environment is cleared and rebuilt from an allow-list, so an unrelated secret in
  your shell cannot leak into a trial.
- The interpreter and its venv are re-bound read-only, including symlink targets —
  a venv under `$HOME` is otherwise hidden by the home tmpfs.
- On systemd-resolved hosts `/etc/resolv.conf` is a symlink into `/run`, which the
  sandbox empties — a networked sandbox binds the real resolver directory so DNS
  works.
- The Pi executor's `~/.pi` is mounted **writable** (pi 1.4 takes lock files next
  to everything it reads) while `auth.json` stays masked by an empty file: reads
  return nothing instead of failing, and the key arrives via the environment.
- If bwrap is missing the run fails closed (`--isolate none` is the explicit
  opt-out), and a gate that cannot be sandboxed returns exit 125 instead of
  running unsandboxed.

Verified: inside a sandboxed trial `cat ~/.ssh/id_rsa`, `cat ~/.hermes/.env`,
listing the vault and `curl https://example.com` all fail, a file written in the
workspace succeeds, and the full test gate still runs to completion.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers every layer plus the end-to-end rig. The harness/orchestrator
tests use a deterministic `FakeExecutor` and a `StubLLM`, so they need neither
Pi nor network — but they exercise **real** git worktrees and **real** pytest
runs inside the critic.

---

## FAQ

Common questions and the failures you meet in the first week, answered from real
use: [docs/FAQ.md](docs/FAQ.md).

## Design principles

The idea behind this rig is written down in [docs/design/](docs/design/) — the
original blueprint and the Harness² research note it grew from.

Built to the pstack rules:

* **Test behaviour, not implementation** — `discipline/checks.py` flags
  assertion-free tests, tautologies, and weak `assert x` / `assert True` forms.
* **Prove it works** — the critic runs the tests; a report without a run is not
  a report.
* **Build the lever** — the rig itself is the tool; every layer is rerunnable.
* **Sequence verifiable units** — one worktree, one trial, one checkpoint.

## Going further

### Fleet diversity

A round can probe several model families at once — a spec that only one model's
habits satisfy then stops looking solved:

```bash
dacode run "make tests/test_x.py pass" \
  --executor pi --llm openrouter \
  --fleet "deepseek/deepseek-v4.1-flash,anthropic/claude-sonnet-5.5" \
  --parallel 2 --variants 2
```

Round 1 runs variant 1 on the first model and variant 2 on the second; later
rounds rotate. When triage decides to escalate, it prefers a family that has not
failed yet instead of repeating the one that just did. `dacode retro` then reports
each model's pass rate and suggests dropping a model that keeps losing.

### MCP server — share the memory with other agents

`dacode mcp` serves the lessons and the run store over MCP (stdio JSON-RPC,
stdlib only, no extra dependency):

```bash
dacode --vault ~/Documents/Vault mcp            # semantic search if an embedder is configured
dacode --vault ~/Documents/Vault mcp --keyword-only   # no network calls
```

Tools: `memory_search` (lessons), `run_history` (FTS over past runs), `failures`,
`stats`, `retro`. Register it with any MCP client, e.g. Hermes:

```yaml
mcp_servers:
  dacode-memory:
    command: /path/to/DaCode/.venv/bin/python
    args: [-m, dacode.cli, --vault, /path/to/vault, mcp, --keyword-only]
    enabled: true
```

Then `hermes mcp test dacode-memory` should report 5 tools.

### Kanban worker

`dacode kanban work` takes one card from a [Hermes Kanban](https://hermes-agent.nousresearch.com/docs/user-guide/features/kanban)
board and reports back with evidence. The card body carries its own gate:

````markdown
Make `tests/test_slug.py` pass. Do NOT modify the test file.

```gate
python3 -m pytest -q tests/test_slug.py
```
````

```bash
dacode kanban work t_1485ddf4 --board myboard \
  --executor pi --llm zai --model glm-5.3-flash --isolate-network
```

1. **claim** first (`hermes kanban claim`, atomic). A card someone else holds
   exits `3` and nothing runs — no card is worked twice. The claim TTL covers
   `budget × variants × 2 × variant-timeout + variant-timeout` (+10 min) unless
   `--ttl` is given — each variant can spend a full timeout executing and
   another inside the critic's gate, and the baseline spends one more. Any
   other claim failure (typo'd id, unknown board) exits `5`, not a fake
   "another worker owns the card".
2. **gate** — the first closed ```` ```gate ```` block becomes the test command.
   No gate → `block --kind needs_input`, exit `2`. An unclosed ```` ```gate ````
   fence is not a gate, and a ````markdown example block does not leak its
   inner ```gate lines as one.
3. **run** — the rig runs under `--isolate bwrap` (always, for cards). A card whose
   workspace is a git checkout (`--workspace dir:<repo>`) is worked there;
   otherwise `--repo`. An explicit `--repo` wins: a card workspace that points
   somewhere else is refused, and a workspace under `$HOME` is refused
   outright (binding it read-write would re-expose the home directory the
   sandbox emptied). The card checkout's own `.env` is loaded for
   allow-listed keys the host shell did not already provide.
4. **report** — pass → `complete` with `--metadata` (gate, gate output — gate
   stderr included, so `unittest` suites and gate timeouts are visible —
   tests passed/total, changed files, winning worktree, runlog path), exit `0`;
   fail or a rig error → `block` with tests x/y and the reason, exit `1`. A
   lost claim at the end is exit `4`, never a silent success; a board call
   that failed outright is exit `5`. Everything sent to the board goes
   through `redact()`.

`DACODE_KANBAN_BIN` / `DACODE_KANBAN_BOARD` override the `hermes` binary and the
board. As with `dacode run`, the winner is a proposal: review the worktree diff
before merging it.

### Reading the ledger

```bash
dacode state stats            # counts, avg_variant_s, slowest_variant_s, skipped tests
dacode retro --days 14        # what to change next: pass rate, rounds, slow variants, tampering
```

## Working on DaCode itself

- `AGENTS.md` — the tool-agnostic playbook for any agent editing this repo (gate
  ownership, contract-first tests, one module per run, conventions).
- `skills/` — the agent-facing operating manual (command shapes, first-week
  failure triage, gate-integrity references). Canonical copy lives in the skills
  repository; this snapshot is read-only by convention.
- `.dacode.md` — the canonical gate command for running the rig *on this repo*:
  `${VIRTUAL_ENV}/bin/python` (trial worktrees have no `.venv`), `-o addopts=`
  (the project's own `-q` would suppress the summary), one module per run.

## Status / roadmap

- [x] Layer 1 Vault Memory (keyword retrieval + ingestion)
- [x] Layer 2 Harness² Optimizer (Propose–Probe–Compose)
- [x] Layer 3 Executor (Pi + script + stub)
- [x] Layer 4 pstack Critic (static checks + real test gate)
- [x] Layer 5 Git Sandbox (worktrees + checkpoints)
- [x] CLI + end-to-end tests
- [x] Jev (TypeSafe System One) judge for the Compose pick (`--judge jev`)
- [x] Judge triage after a failed round — retry / new proposal / escalate model / human review
- [x] Run recorder (`dacode/runlog.py`) + `dacode judge-eval` — Jev agreement with the critic gate
- [x] SQLite local state store (`memory.db`) + FTS5 trigram search — `dacode state`
- [x] Semantic retrieval with pluggable backends — DaCode-owned vector store by default (60/40 blend)
- [x] Parallel probing of variants — `--parallel N` (worktree creation stays serialised)
- [x] Per-variant timeout (`--variant-timeout`) and a cost/time ledger in the runlog + `state stats`
- [x] `dacode retro` — reads the runlog and suggests harness changes (pstack #5)
- [x] Test-integrity guards on the baseline — `tests_modified` / `tests_lost` /
      `tests_skipped`, plus `tests_infra_modified` (a `conftest.py` or pytest config
      can silence or spoof the gate itself) and a verdict that requires
      `failed == 0`, not just exit code 0
- [x] Fleet diversity — `--fleet a,b` puts one model per variant in a round; escalation prefers an untried family
- [x] Kanban worker — `dacode kanban work <task_id>` claims a Hermes Kanban card, gates on its ```` ```gate ```` block, then completes or blocks it
- [x] MCP memory server — `dacode mcp` (5 tools over stdio JSON-RPC, stdlib only)
- [x] Bubblewrap isolation — `--isolate bwrap`: read-only host, empty home,
      workspace-only writes, network unshared, allow-listed env; the gate is
      sandboxed too and fails closed

### Next

- [ ] **Network allow-list for isolated trials.** `--isolate bwrap` is
      all-or-nothing on the network today: the Pi executor needs it, a task that
      fetches a dependency needs it, and then trial code can reach anything.
      Planned: an egress proxy on the host with a per-run host allow-list, DNS
      pinned to it, and `--isolate-network=allowlist:<file>`. Gate: a trial reaches
      a listed host, any other host fails, and an unparsable policy refuses to
      start — never a silent "allow all".

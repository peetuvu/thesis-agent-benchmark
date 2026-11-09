# Multi-agent LLM benchmark harness

This is the benchmarking harness for Peetu Vuorela's master's thesis at Aalto University, *Automating The Benchmarking of Multi-Agent LLM Architectures for Software Development Tasks*. It gives an LLM a small Python programming task, writes the code the model returns into a throwaway copy of the task, scores that code against tests the model never sees, and logs the result with token use, latency, API calls and estimated cost. It currently has two architectures (`single_agent`, one LLM call; `sequential`, a fixed plan, implement, review, fix pipeline), two providers (Anthropic and OpenAI) and five tasks (`snake`, `calculator`, `urlshort`, `mdtable`, `diffapply`). The main experiment so far is 480 runs: four tasks (all except `snake`), six models, two architectures, two prompt regimes and five repetitions.

## Requirements

- Python 3.11 or newer (`requires-python` in `pyproject.toml`). The steps below were tested with 3.12.3.
- [uv](https://docs.astral.sh/uv/), installed separately. The repository doesn't declare a uv version; 0.13.7 was used for testing.
- Linux or WSL, with `bash` and `git`. `eval/run_eval.sh` needs bash, and each run log records the git commit.
- An Anthropic API key and, for the GPT models, an OpenAI API key.

## Setup from a fresh clone

```bash
git clone git@github.com:peetuvu/thesis-agent-benchmark.git
cd thesis-agent-benchmark
uv venv
source .venv/bin/activate
uv sync --locked --extra dev
```

- `--locked` installs exactly what `uv.lock` pins and fails if the lockfile is out of date.
- `--extra dev` is required, not optional. It installs pytest, and every run scores the model's code with pytest. Without it, a run still pays for the API call and then ends with status `error`.
- Activate the environment in every new shell. The batch runner finds `bench` through `PATH`.

Create `.env` in the repository root. **The repository has no `.env.example`**, so write the file yourself:

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
EOF
```

| Key | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | `haiku-5.5`, `sonnet-5.5`, `opus-5.5`. `check_env.py` also treats it as mandatory. |
| `OPENAI_API_KEY` | `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6-astra` |
| `GOOGLE_API_KEY` | Nothing. `check_env.py` looks for it, but no Google provider exists. |

`.env` is gitignored. `bench` and `check_env.py` load it automatically.

## Check the environment

```bash
python scripts/check_env.py
```

A working setup prints `[OK]` for Python, the `bench` CLI, both API keys and the three packages, then `Environment ready for providers: anthropic, openai`, and exits 0. A `[WARN]` for `GOOGLE_API_KEY` is expected.

The script makes no API calls, so it can't tell whether a key is valid or has credit. It also doesn't check that pytest is installed. To check that separately, run the harness tests, which need no keys and make no API calls:

```bash
pytest
```

Expected result: `411 passed`.

## Snake smoke test

This makes one real, paid API call. Earlier smoke runs of this exact configuration cost about $0.001 and took about 7 seconds.

```bash
bench run --task snake --arch single_agent --model haiku-5.5 --reasoning-effort low
```

A correct result looks like this and exits 0:

```text
==================================================
Run:    <uuid>
Status: success
Hidden: 8/8 passed
Public: 5/5 passed
Policy: compliant
Log:    runs/<YY-MM-DD_HHMM>_anthropic_haiku-5-5_single_agent_snake_minimal_<id>.json
==================================================
```

| What you see | What it means |
|---|---|
| `Status: failure`, `Hidden: x/8`, `Public: y/5`, exit 1 | The harness worked and the model's code failed some tests. Try again or read the failing test names in the log. |
| `Status: error`, exit 1 | The provider call failed, or the tests produced no results. The reason is in `stop_reason` in the run's `.json`. |
| Traceback ending in `ANTHROPIC_API_KEY environment variable is not set`, exit 1 | `.env` is missing or incomplete. No log is written. |
| `requires an explicit reasoning effort`, exit 2 | `--reasoning-effort` is missing. No API call is made. |

The evidence for the 8/8 and 5/5 result is in the 12 Snake logs in the author's local `runs_smoke/`, which all reached it. The most recent `haiku-5.5` one used commit `0931f26`. The current commit was checked with a fake provider, not a real call (see [Verification](#verification)).

## Where logs go

Each run writes three files to `runs/`, sharing one stem:

```text
{YY-MM-DD_HHMM, Helsinki time}_{provider}_{model}_{architecture}_{task}_{regime}_{first 8 of run id}
```

| File | Contents |
|---|---|
| `<stem>.json` | The main record. Run id, UTC timestamp, git commit, branch and dirty flag; task, architecture, regime, model, provider and requested reasoning effort; budgets; sampling parameters actually sent; tokens, API calls, wall-clock time, estimated cost and price source; public and hidden test results with failing test names; one record per LLM call; `status`, `policy_compliant`, `stop_reason`, `notes`, and the sidecar's filename. |
| `<stem>.md` | The same run as a short human-readable report. |
| `<stem>.artifacts.json` | Bulky content kept out of the main log. Every system prompt, user prompt and response per call (`calls`), the architecture's step log (`agent_log`), files as parsed from the model (`modified_files`), and the editable files as they were when tests ran (`evaluated_files`). |
| `batch_<YY-MM-DD_HHMMSS>.json` | Written by `scripts/run_batch.py`, once, when the whole batch finishes. Holds the totals, the batch's arguments, and one entry per run (`task`, `model`, `arch`, `regime`, `rep`, `command`, `status`, `hidden`, `error`). Its `errors` count includes every run that wasn't a success, ordinary test failures too. |

Status values are `success` (every hidden test passed and no policy violation), `failure`, `error`, `timeout` and `policy_violation`.

**Run logs are not in git.** `.gitignore` excludes `runs/*.json` and `runs/*.md`, so a fresh clone has no logs and the analysis scripts print `No runs found`. The experiment's logs (57 MB) exist only on the author's machine and have to be copied separately. The derived results in `analysis/*.csv` are committed.

## Repository map

| Path | Purpose |
|---|---|
| `agent-bench/` | Task specs, stubs, public tests and prompt templates |
| `eval/` | Hidden tests and the scoring script, never shown to the model |
| `architectures/` | `single_agent` and `sequential` |
| `providers/` | Provider wrappers, model registry, price table |
| `runner/` | CLI, executor, sandbox checks, logger |
| `scripts/` | Batch runner, environment check, analysis |
| `analysis/` | Committed result CSVs |
| `docs/` | Documentation |

## Further documentation

- [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md): running batches, analysing them, and recovering from a partial batch
- [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)
- [docs/ADDING_A_TASK.md](docs/ADDING_A_TASK.md)
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): read this before relying on results
- [docs/harness_architecture.md](docs/harness_architecture.md): how the harness works, with code references
- [docs/task_summary.md](docs/task_summary.md): what each task asks for and tests

## Verification

Checked on 2026-11-09 from a fresh clone of commit `7fb81a6`, with API keys unset: `uv lock --check`, `uv venv`, `uv sync --locked --extra dev`, `check_env.py`, and `pytest` (411 passed). The output format and log files above were checked by running the real CLI and executor with a fake provider.

**[NOT VERIFIED]** The smoke test wasn't run against a real API while writing this, and access to the GitHub repository wasn't tested.

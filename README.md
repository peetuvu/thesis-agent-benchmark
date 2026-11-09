# Multi-agent LLM benchmark harness

This harness benchmarks LLM architectures on small Python coding tasks for Peetu Vuorela's master's thesis at Aalto University. It writes a model's code into a temporary copy of a task, scores it against hidden tests, and logs results, tokens, latency and cost to `runs/`. It supports the `single_agent` and `sequential` architectures on Anthropic and OpenAI models.

## Quickstart

```bash
git clone git@github.com:peetuvu/thesis-agent-benchmark.git  # untested
cd thesis-agent-benchmark
uv sync --locked
source .venv/bin/activate
cp .env.example .env
sed -i 's/^ANTHROPIC_API_KEY=/&PASTE_KEY_HERE/' .env
python scripts/check_env.py
pytest
bench run --task snake --arch single_agent --model haiku-5.5 --reasoning-effort low  # untested
```

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/) first.

Replace `PASTE_KEY_HERE` with your Anthropic key before running that line.

Expect `Status: success`, `Hidden: 8/8 passed` and `Public: 5/5 passed` from the last command.

Run `bench models` to list the model aliases.

## Full batch

```bash
MATRIX="--tasks calculator diffapply mdtable urlshort --models haiku-5.5 sonnet-5.5 opus-5.5 gpt-6-luna gpt-6.1-sol gpt-6-astra --archs single_agent sequential --regimes minimal production --reps 5 --reasoning-effort low"
python scripts/run_batch.py $MATRIX --dry-run
python scripts/run_batch.py $MATRIX  # untested
```

Set both keys in `.env` and budget about $42 for these 480 runs.

## Analysis

```bash
python scripts/rescore_runs.py
python scripts/aggregate_results.py --rescored
for t in calculator diffapply mdtable urlshort; do python scripts/analyze_stages.py "$t" --rescored --csv "analysis/stage_analysis_$t.csv"; done
python scripts/classify_failures.py --rescored --csv analysis/failure_classes_mdtable.csv
```

Copy `runs/` from Peetu first to analyse the thesis experiment, because run logs are not in git.

## Docs

- [Running experiments](docs/RUNNING_EXPERIMENTS.md)
- [Adding a model](docs/ADDING_A_MODEL.md)
- [Adding a task](docs/ADDING_A_TASK.md)
- [Known issues](docs/KNOWN_ISSUES.md)

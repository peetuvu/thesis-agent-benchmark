# CLAUDE.md — Thesis Benchmark Harness

## Project overview

This is a **master's thesis project** (Aalto University, 2026) building a reproducible benchmarking harness for comparing multi-agent LLM architectures on automated software development tasks **across multiple model providers**. The thesis measures accuracy (pass@1), token efficiency, latency, API call count, and cost across architectures and models.

The primary research question is which combination of architecture and model/provider produces the best functional correctness. Cross-provider model comparison (Claude vs. GPT vs. Gemini, etc.) is a core thesis goal, not future work. Resource trade-offs (tokens, latency, cost) are secondary outcomes.

Author: Peetu Vuorela
Thesis title: "Automating The Benchmarking of Multi-Agent LLM Architectures for Software Development Tasks"

## Current phase

Building the provider abstraction layer, then Stage E1 (single-agent baseline). Infrastructure (runner, logging, sandbox, eval) is complete and tested.

See STAGES.md for the full development roadmap and current status.

## Repository structure

```
thesis-agents/
├── CLAUDE.md                  # This file
├── STAGES.md                  # Development roadmap with stage status
├── pyproject.toml             # Project config, dependencies
├── .gitignore
│
├── agent-bench/               # AGENT WORKSPACE — what the benchmarked agent sees
│   ├── README.md              # Describes the sandbox concept and agent rules
│   ├── tasks/                 # Task families
│   │   └── snake/             # Snake game task (complete)
│   │       ├── README.md      # Task spec (given to agent)
│   │       ├── snake/         # Source files agent edits
│   │       │   ├── __init__.py
│   │       │   └── game.py    # Stub — agent must implement this
│   │       ├── tests/         # Public sanity tests (agent can run these)
│   │       │   └── test_public_sanity.py
│   │       └── pytest.ini
│   └── prompts/               # Global prompt templates (shared across all tasks)
│       ├── minimal.md         # Sparse prompt — minimal instructions
│       └── production.md      # Rich prompt — derived from company workflows
│
├── eval/                      # EVALUATION HARNESS — agent must NEVER access this
│   ├── __init__.py
│   ├── hidden_tests/snake/    # Hidden test files for scoring
│   │   ├── conftest.py        # BENCH_SOURCE_DIR support for temp-dir isolation
│   │   └── test_hidden.py
│   ├── run_eval.sh            # Evaluation runner (accepts optional source_dir)
│   ├── passk.py               # pass@k computation
│   └── test_passk.py          # Tests for pass@k estimator
│
├── providers/                 # LLM PROVIDER ABSTRACTION (to be built)
│   ├── __init__.py
│   ├── base.py                # LLMProvider ABC + LLMResponse dataclass
│   ├── anthropic.py           # Anthropic Claude SDK wrapper
│   ├── openai.py              # OpenAI GPT SDK wrapper
│   └── google.py              # Google Gemini SDK wrapper (add when needed)
│
├── architectures/             # Multi-agent architecture implementations
│   ├── __init__.py
│   └── base.py                # Abstract base class (Architecture, TaskContext,
│                              #   RunResult, RunMetrics, RunStatus)
│
├── runner/                    # Benchmark orchestration (complete)
│   ├── __init__.py
│   ├── config.py              # RunConfig pydantic model + TaskPaths helper
│   ├── executor.py            # Executor: temp-dir isolation, eval, logging
│   ├── sandbox.py             # SHA-256 snapshot + policy compliance checking
│   ├── logger.py              # RunLog schema + JSON/markdown log writer
│   └── cli.py                 # CLI entry point (bench command)
│
├── runs/                      # RUN LOGS — structured JSON + markdown per run
│   └── .gitkeep
│
└── tests/                     # Harness tests (21 passing)
    └── test_harness.py
```

## Critical rules (ALWAYS follow these)

### Directory boundaries — hard constraints
- **agent-bench/**: The benchmarked agent's sandbox. Contains task specs, source files, and public sanity tests.
- **eval/**: Hidden tests and scoring scripts. The benchmarked agent must NEVER read, access, or be given contents from this directory. If a run leaks hidden tests to the agent, results are invalid.
- **runs/**: Run logs. The benchmarked agent does not write here. The harness writes logs here.
- **architectures/**, **runner/**, **providers/**: Harness infrastructure. This is what WE (developer + Claude Code) build and maintain.

### When working on this project
- You ARE helping build the harness infrastructure (architectures/, runner/, providers/, eval/, tests/).
- You are NOT the benchmarked agent. Do not confuse your role with the agent being tested.
- When editing eval/hidden_tests/, remember these must stay hidden from the agent runtime.
- Keep the harness modular: adding a new architecture, task, or provider should not require modifying the runner core.

### Sandbox policy (Stage B)
- Pre-run: SHA-256 hash of eval/ directory
- Post-run: re-hash and compare — fail run if anything changed
- Post-run: verify only allowed files were modified in task workspace
- Policy compliance status is recorded in every run log

### Execution isolation (Stage C)
- Each run copies the task to a fresh temp directory (shutil.copytree)
- The real repo is NEVER modified during a run
- Hidden eval runs from real eval/ with BENCH_SOURCE_DIR pointing to temp dir
- Only side effect of a run is the log written to runs/

## Tech stack
- Python 3.11+ (WSL Ubuntu 24.04)
- pytest for both harness tests and benchmark evaluation
- pydantic for config and log schemas
- anthropic SDK (ANTHROPIC_API_KEY env var)
- openai SDK (OPENAI_API_KEY env var)
- google-generativeai SDK (GOOGLE_API_KEY env var, add when needed)
- Git for version control

## Provider abstraction
- Architectures NEVER import provider SDKs directly
- All LLM calls go through providers/base.py LLMProvider interface
- Each provider wraps one SDK and returns a standardized LLMResponse
- Provider is selected via RunConfig.provider string, loaded dynamically
- Adding a new provider = one new file in providers/, zero architecture changes

## Coding conventions
- Type hints on all function signatures
- Docstrings on public functions and classes
- snake_case for functions/variables, PascalCase for classes
- Keep files focused — one architecture per file, one provider per file
- Log everything: tokens, latency, tool calls, exit status, test outcomes
- Use pydantic for structured data (run configs, run logs)

## Key design decisions
- **pass@1 under fixed budget** is the primary correctness metric
- **Cross-provider comparison** is a primary goal — same architecture, different models
- **Matched budgets**: architectures are compared under the same token/tool-call caps
- **Prompt regimes**: minimal vs. production prompts, tested separately as experimental variable
- **Determinism where possible**: fix seeds, lock dependency versions, log everything
- **Multiple runs**: each config is run multiple times to capture variance
- **Temp-dir isolation**: eliminates cross-run contamination, no git-reset needed

## Commands
- `pytest agent-bench/tasks/snake/tests/` — run public sanity tests for snake task
- `bash eval/run_eval.sh snake` — run hidden evaluation (human/automation only)
- `bash eval/run_eval.sh snake /path/to/temp/dir` — run eval against temp dir
- `bench run --task snake --arch single_agent --provider anthropic` — run benchmark
- `pytest tests/test_harness.py` — run all 21 harness infrastructure tests

## What to build next (priority order)
1. ~~Repo structure~~ DONE
2. ~~Architecture base class~~ DONE
3. ~~Runner/executor with temp-dir isolation~~ DONE
4. ~~Structured logging (JSON + markdown)~~ DONE
5. ~~Sandbox policy checking~~ DONE
6. **-> Provider abstraction layer (providers/)**
7. **-> Single-agent baseline (architectures/single_agent.py)**
8. Prompt regime templates (fill minimal.md content)
9. First end-to-end benchmark run
10. Sequential pipeline architecture (architectures/sequential.py)
11. Provider-specific telemetry (Stage D2 — mostly done via provider layer)

DEVELOPMENT STAGES — Thesis Benchmark Harness
================================================

MVP target: 1 task family (Snake), 2 architectures (single-agent baseline
+ fixed 3-stage sequential pipeline), strict matched budgets, automated
hidden eval, structured JSON logs. This alone is a defensible thesis
contribution.

---------------------------------------------------------------
STAGE A — Repo structure + task packaging            [DONE]
---------------------------------------------------------------
Difficulty: Easy (if frozen early)

What exists:
- agent-bench/, eval/, runner/, runs/ layout is stable
- Snake task: spec, stub, public tests, hidden tests, eval script
- Architecture base class with full data model
- pass@k estimator with tests

Remaining:
- [x] Verify snake README covers ALL features tested by hidden tests
      (big food, obstacles, wrap mode, deterministic respawn algo)
- [x] Fill agent-bench/README.md with a brief description
- [x] Confirm prompts/ directory purpose — prompts live globally
      in agent-bench/prompts/, shared across all tasks

Rule: Do NOT restructure directories after this stage.

---------------------------------------------------------------
STAGE B — Sandbox policy (detect + invalidate)       [MOSTLY DONE]
---------------------------------------------------------------
Difficulty: Medium

Approach: "detect and invalidate" — not "prevent."
Full containerization is out of scope for now.

Build:
- [x] Pre-run: hash eval/ directory contents (SHA256 manifest)
- [x] Post-run: re-hash and compare — fail run if anything changed
- [x] Post-run: check that only allowed files were modified
      (e.g., only game.py in the snake task)
- [x] Record policy compliance status in run log
- [ ] Document the policy in CLAUDE.md and in the thesis methodology

Future (optional): Docker-based sandboxing if needed for SWE-bench.

---------------------------------------------------------------
STAGE C — Runner / harness                           [DONE]
---------------------------------------------------------------
Difficulty: Medium → Hard (biggest time sink)

Core design decisions (lock these in):
- Each run copies the task to a fresh temp directory (shutil.copytree
  + tempfile.mkdtemp). Never run in-place. This eliminates most
  cross-run contamination.
- Single entry point: `bench run --task snake --arch single_agent`
- Explicit PYTHONPATH, explicit python executable, explicit cwd
- Runner calls architecture.run(), then runs eval, then writes log
- Real repo is never modified during a run — eval runs against temp
  dir via BENCH_SOURCE_DIR env var

Build order:
- [x] runner/config.py — RunConfig pydantic model + TaskPaths helper
- [x] runner/executor.py — Executor class:
        1. Copy task dir to temp
        2. Call architecture.run(task_context)
        3. Run sandbox policy checks
        4. Run hidden eval (subprocess: bash eval/run_eval.sh <task>)
        5. Collect results → RunLog
- [x] runner/cli.py — argparse CLI wired to executor
      (entry point: `bench` command from pyproject.toml)
- [x] 21 passing harness tests (tests/test_harness.py)

Key contamination sources to guard against:
- leftover __pycache__ / .pytest_cache
- agents adding files that change import behavior
- venv vs system python mismatch
- working directory differences

---------------------------------------------------------------
STAGE D — Logging + traceability                     [D1 DONE]
---------------------------------------------------------------
Difficulty: Hard (most underestimated)

Split into two sub-stages:

D1: Schema + local JSON logging (do this first)
- [x] Define RunLog schema (pydantic model):
        run_id (UUID), timestamp, git_commit, git_branch
        task_name, architecture_name, prompt_regime
        model, provider
        budget_config (max_tokens, max_tool_calls, max_wall_clock)
        actual_metrics (tokens_prompt, tokens_completion, tokens_total,
                        api_calls, wall_clock_seconds, estimated_cost)
        test_results (public_passed, public_total, hidden_passed,
                      hidden_total, failing_test_names)
        status (success/failure/error/timeout/budget_exceeded)
        policy_compliant (bool)
        stop_reason (string)
        notes (optional string)
- [x] Write each run as a JSON file in runs/<run_id>.json
- [x] Also write a human-readable markdown summary
- [x] Fields can be null — better null consistently than missing

D2: Provider-specific telemetry extraction (incremental)
- [ ] Anthropic API: extract token usage from response metadata
- [ ] OpenAI API: extract token usage from response metadata
- [ ] Google API: extract token usage (if available)
- [ ] Cost estimation: model → price-per-token lookup table
- [ ] Don't let "can't get tokens from provider X" block logging
      for provider Y

---------------------------------------------------------------
STAGE D3 — Provider abstraction layer                [DONE] (effectively)
---------------------------------------------------------------
Difficulty: Medium

Build:
- [x] providers/base.py — LLMResponse dataclass + LLMProvider ABC + ProviderError
- [x] providers/anthropic.py — Anthropic Claude SDK wrapper
- [x] providers/openai.py — OpenAI GPT SDK wrapper
- [x] providers/__init__.py — get_provider() dynamic loader + registry
- [x] TaskContext.provider field added to architectures/base.py
- [x] Executor instantiates provider from RunConfig.provider
- [x] tests/test_providers.py — mocked unit tests (no real API calls)
- [ ] providers/google.py — Google Gemini SDK wrapper (add when needed)
- [ ] End-to-end smoke test with real API keys

Design:
- Architectures call ctx.provider.complete() — never import SDKs directly
- get_provider("anthropic") dynamically loads AnthropicProvider
- LLMResponse standardizes text, token counts, stop_reason across providers
- ProviderError wraps SDK-specific auth/API errors

---------------------------------------------------------------
STAGE E — Experiment orchestration                   [E2 IN PROGRESS]
---------------------------------------------------------------
Difficulty: Hard

Stage incrementally — do NOT jump to ensembles:

E1: Single-agent baseline (uses LLMProvider interface)
- [x] architectures/single_agent.py — one LLM call via ctx.provider, one shot
- [x] agent-bench/prompts/minimal.md — real minimal prompt template
- [x] tests/test_single_agent.py — mocked unit tests
- [ ] Validate full pipeline: run → eval → log
- [ ] Run snake task, confirm hidden tests score correctly

E2: Fixed-length sequential pipeline (no loops)
- [x] architectures/sequential.py — plan → implement → review → fix
- [x] Fixed 4 steps, no iteration
- [x] Same budget constraints as single-agent
- [x] tests/test_sequential.py — mocked unit tests

E3: Add iteration (review/fix loops)
- [ ] Configurable max_iterations
- [ ] Controller that can stop agents and record WHY
- [ ] "Who gets to run tests when" rules must be consistent

E4: Additional architectures (only if time permits)
- [ ] Parallel ensemble with aggregation
- [ ] Adversarial debate
- [ ] Blackboard/shared memory

Budget enforcement:
- Each architecture gets the same max_tokens and max_tool_calls
- A run terminates when: correct solution, budget exceeded, timeout,
  or max iterations reached
- Stop reason is always recorded

Do NOT build E3/E4 until you have comparative results from E1 vs E2.
A fixed pipeline without iteration may already tell an interesting story.

---------------------------------------------------------------
STAGE F — Task scaling                               [NOT STARTED]
---------------------------------------------------------------
Difficulty: Medium → Hard

Order:
- [ ] Second toy task with DIFFERENT failure modes (e.g., string
      manipulation, simple data structure — NOT another game)
      Two diverse toy tasks with solid methodology > 164 HumanEval
      problems with a shaky runner.
- [ ] HumanEval subset (start with 20-30 problems, not all 164)
- [ ] MBPP subset (optional, if HumanEval goes smoothly)
- [ ] SWE-bench subset (optional, hard, different paradigm)

Each new task family will likely break something in the runner.
Do NOT attempt this until Stage C is rock-solid.

---------------------------------------------------------------
PREDICTED SINGLE HARDEST PART
---------------------------------------------------------------
Reproducibility + validity under real-world mess.

Not writing code — keeping the system stable while adding:
  more tasks, more architectures, more models/providers, more runs.

Engineering discipline, not cleverness.

---------------------------------------------------------------
CURRENT POSITION (as of 2026-03-05)
---------------------------------------------------------------
Stage A: Done
Stage B: Mostly done (1 remaining: document policy in CLAUDE.md / thesis)
Stage C: Done — runner/config.py, executor.py, cli.py, sandbox.py, 21 tests
Stage D: D1 done (schema + JSON/markdown logs). D2 not started.
Stage E: Not started — next action: E1 single-agent baseline
Stage F: Not started

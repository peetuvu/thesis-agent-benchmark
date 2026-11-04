# Harness architecture

A factual description of how the benchmark harness works, written from the code in the working
tree on 2026-11-04. That is commit `1c0612b` plus the uncommitted fixes made the same day; see
"Changes since `1c0612b`" at the end. Every claim cites the file and function it comes from.

**Method.** I read the source code only. Statements were checked against the test suite (344
tests, all passing) and one local executor run. That run used a fake provider and the real
hidden and public pytest runs. No paid API calls were made. Items marked **[UNVERIFIED]** come
from reading the code but were not exercised.

---

## 1. Components and what calls what

```
bench run (runner/cli.py:main)
 ├─ providers/models.py: resolve_model()                 alias -> (api_string, provider)
 ├─ runner/cli.py: _load_architecture()                  imports architectures/<arch>.py
 ├─ runner/config.py: RunConfig(...)                     validator -> models.check_reasoning_effort()
 └─ runner/executor.py: Executor.run()
     ├─ runner/logger.py: get_git_info(), get_git_dirty()
     ├─ runner/sandbox.py: snapshot_directory(eval/)
     ├─ shutil.copytree(task dir -> temp dir)
     ├─ Executor._run_architecture()
     │   ├─ providers/__init__.py: get_provider()        -> AnthropicProvider | OpenAIProvider
     │   ├─ wraps it in executor._RecordingProvider      records every call, its prompts/response, cost
     │   │     └─ providers/pricing.py: estimate_call_cost()
     │   ├─ Executor._build_context()                    -> TaskContext
     │   └─ Architecture.run_timed() -> <arch>.run()     (under asyncio.wait_for timeout)
     │         ├─ architectures/utils.py: render_prompt_template()
     │         └─ ctx.provider.complete(...)             (one or more LLM calls)
     ├─ Executor._apply_modified_files()                 rejects paths outside the workspace
     ├─ sandbox.snapshot_directory(temp) + check_policy()
     ├─ Executor._run_hidden_eval()  -> bash eval/run_eval.sh <task> <temp dir> -> pytest
     ├─ Executor._run_public_tests() -> python -m pytest <temp>/tests
     ├─ sandbox.snapshot_directory(eval/) + check_policy()
     ├─ executor._derive_outcome()                       final status + stop_reason
     ├─ pricing.get_price()
     ├─ logger.save_run_artifacts()                      runs/<stem>.artifacts.json
     └─ logger.save_run_log()                            runs/<stem>.json + runs/<stem>.md

scripts/run_batch.py:main          -> subprocess "bench run ..." once per combination
scripts/aggregate_results.py       -> reads runs/*.json run logs and prints a summary table
```

| Component | File / entry point | Role |
|---|---|---|
| CLI | `runner/cli.py:main`, `_build_parser`, `_load_architecture` | `bench run` / `bench models`. Resolves the model, loads the architecture, builds `RunConfig`, calls `Executor.run`, prints a summary. Installed as `bench` via `pyproject.toml [project.scripts]`. |
| Batch runner | `scripts/run_batch.py:main`, `build_combinations`, `parse_output` | Builds the cartesian product task × model × arch × regime × reps and runs `bench run` **as a subprocess** for each, one after another. Reads `Status:` and `Hidden:` from stdout. Writes `runs/batch_<timestamp>.json`. Never imports harness code. It has no flag for `--max-output-tokens`, so batches use the default of 32000. |
| Config | `runner/config.py:RunConfig`, `TaskPaths` | Pydantic run config (generates `run_id` and the UTC `timestamp`). Budget fields: `max_tokens` (recorded only), `max_output_tokens` (per-call cap, default 32000, must be > 0), `max_tool_calls` (recorded only), `max_wall_clock_seconds`. `TaskPaths.allowed_edit_files` is every `*.py` under `agent-bench/tasks/<task>/<task>/` except `__init__.py`. |
| Executor | `runner/executor.py:Executor.run` | Orchestrates one run (section 2). |
| Call recorder | `runner/executor.py:_RecordingProvider` | Transparent `LLMProvider` wrapper. For every call it appends an `LLMCallRecord` to `calls` and a `CallTranscript` (system prompt, user prompt, response text) to `transcript`. That covers successful calls, `ProviderError`s, other exceptions, and calls cancelled by the timeout (`asyncio.CancelledError`). |
| Architectures | `architectures/base.py:Architecture` (`run`, `run_timed`), `TaskContext`, `RunResult`, `RunMetrics`, `RunStatus`; `architectures/single_agent.py:SingleAgent`; `architectures/sequential.py:SequentialPipeline` | Receive a `TaskContext` and return a `RunResult` whose `modified_files` maps a path to its full contents. Both current architectures parse labeled fenced code blocks out of plain-text responses. Neither uses tools. |
| Prompt utilities | `architectures/utils.py:render_prompt_template`, `strip_template_header`, `resolve_editable_files`, `_check_placeholders`, `parse_code_blocks`, `format_source_files`, `TEST_CMD` | The one place a prompt-regime template becomes prompt text (section 4.1). |
| Provider layer | `providers/base.py:LLMProvider`, `LLMResponse`, `ProviderError`, `NORMAL_STOP_REASONS`, `TRUNCATION_STOP_REASONS`; `providers/__init__.py:get_provider` | Single-method interface `complete(model, system_prompt, user_prompt, max_tokens, temperature, seed, thinking_enabled, reasoning_effort)`. Here `max_tokens` is the per-call output cap. `get_provider` imports the class dynamically from `_PROVIDER_REGISTRY` (`anthropic`, `openai`). |
| Anthropic | `providers/anthropic.py:AnthropicProvider.complete`, `_sent_params` | `AsyncAnthropic.messages.create` with `max_tokens` **sent unchanged** and an explicit `Timeout(REQUEST_TIMEOUT_SECONDS=600, connect=5)` (`providers/base.py`). Because a timeout is passed, the SDK skips its "streaming required" check for large `max_tokens` (`anthropic/resources/messages/messages.py`: the check runs only `if not stream and not is_given(timeout)`). With `reasoning_effort`: `thinking={"type":"adaptive"}`, `output_config={"effort":…}`, no temperature. Legacy path: temperature via `extra_body`; `thinking_enabled` sends `budget_tokens=10000` with temperature 1.0. `seed` is never sent. |
| OpenAI | `providers/openai.py:OpenAIProvider.complete`, `_reasoning_effort` | `AsyncOpenAI.chat.completions.create` with `max_completion_tokens` **sent unchanged** and `timeout=REQUEST_TIMEOUT_SECONDS` (600 s). Temperature is sent only when the effort is `None` or `"none"`. `seed` is passed through if given. Records `reasoning_tokens` and `cached_input_tokens` in `usage_details`. |
| Model registry | `providers/models.py:MODELS`, `LEGACY_ALIASES`, `REASONING_EFFORTS`, `resolve_model`, `check_reasoning_effort`, `get_model_by_api_string`, `list_models` | Maps alias → `api_string`, provider and tier, plus accepted reasoning efforts. `check_reasoning_effort` runs in `RunConfig._validate_reasoning_effort` and again in each provider's `complete`. |
| Pricing | `providers/pricing.py:PRICES`, `Price`, `get_price`, `estimate_call_cost` | USD/MTok list prices keyed by API string, each with a source URL. Handles the OpenAI cached-input rate and Haiku 5.5's >100K-prompt tier. Unpriced models return `None`. |
| Sandbox | `runner/sandbox.py:snapshot_directory`, `check_policy` | SHA-256 of every file under a directory (skipping `__pycache__` and `.pytest_cache`). The diff reports added, modified and deleted files that are not in an allow-list. |
| Evaluation | `eval/run_eval.sh`; `eval/hidden_tests/<task>/conftest.py`, `test_hidden.py`; parsed by `executor._parse_pytest_output` | Runs pytest on `eval/hidden_tests/<task>` with `PYTHONPATH=<temp dir>:<repo root>` and `BENCH_SOURCE_DIR=<temp dir>`. `conftest.py` puts `BENCH_SOURCE_DIR` first on `sys.path`. Uses `.venv/bin/python` if present, otherwise `python3`. |
| pass@k | `eval/passk.py` | Estimator module. Not called by any harness code or script; only `eval/test_passk.py` imports it. |
| Logger | `runner/logger.py:RunLog`, `LLMCallRecord`, `SamplingConfig.from_calls`, `BudgetConfig`, `ActualMetrics`, `EvalResults`, `CallTranscript`, `RunArtifacts`, `save_run_log`, `save_run_artifacts`, `artifacts_filename`, `is_run_log_file`, `make_run_filename`, `_render_markdown` | Pydantic schemas plus the JSON, markdown and artifacts writers (section 5). |
| Aggregation | `scripts/aggregate_results.py:load_runs`, `group_key`, `summarize`, `print_table`, `main` | Section 5.4. |

---

## 2. Lifecycle of one run

### 2.1 In order

**CLI (`runner/cli.py:main`)**

1. `argparse` parses the flags. Defaults: `--model sonnet-4.6`, `--prompt-regime minimal`, `--max-tokens 100000`, `--max-output-tokens 32000`, `--max-tool-calls 10`, `--max-wall-clock 1200`. `--reasoning-effort` is limited to the values in `REASONING_EFFORTS`.
2. `resolve_model(args.model)` returns the registered `api_string` and provider for an alias. An unknown string is passed through as-is with provider `None`.
3. Provider = `--provider`, else the registry provider, else `"anthropic"`.
4. `_load_architecture(args.arch)` imports `architectures.<arch>` and instantiates the first `Architecture` subclass found in `dir(module)` (alphabetical order).
5. `RunConfig(...)` is built; its validators run (`check_reasoning_effort`, and `max_output_tokens > 0`).
6. `asyncio.run(Executor().run(config, architecture))`.

**Executor (`runner/executor.py:Executor.run`)**

7. Step 0: `get_git_info()` (commit and branch, run in the process cwd) and `get_git_dirty(cwd=repo_root)`.
8. Step 1: `eval_pre = snapshot_directory(eval/)`.
9. Step 2: `tempfile.mkdtemp(prefix="bench_<task>_")`, then `shutil.copytree(agent-bench/tasks/<task>, <tmp>/task)`.
10. Everything from here runs inside `try … finally: shutil.rmtree(tmp_dir, ignore_errors=True)`.
11. Step 3: `task_pre = snapshot_directory(<tmp>/task)`.
12. Step 4: `_run_architecture`:
    1. `get_provider(config.provider)` instantiates the SDK client and checks the API key. It is wrapped in `_RecordingProvider`.
    2. `_build_context` builds the `TaskContext` (section 4.1).
    3. The executor records `time.monotonic()`, then runs `asyncio.wait_for(architecture.run_timed(ctx), timeout=config.max_wall_clock_seconds)`. `run_timed` sets `metrics.wall_time_seconds` after `run` returns normally.
13. Step 5: `_apply_modified_files` writes each `modified_files[path]` to `<tmp>/task/<path>`. A path that resolves outside `<tmp>/task` (through `..`, an absolute path, or a symlink) is **not written** and becomes a violation `rejected (outside workspace): <path>`. The contents of the `allowed_edit_files` as they now stand are captured as `evaluated_files`.
14. Step 6: `task_post` snapshot, then `check_policy(task_pre, task_post, allowed_files=paths.allowed_edit_files)`.
15. Step 7: `_run_hidden_eval` runs `bash eval/run_eval.sh <task> <tmp>/task` (cwd = repo root, 120 s timeout) and parses the `pytest -v` lines.
16. Step 8: `_run_public_tests` runs `python -m pytest -v --tb=line <tmp>/task/tests` (cwd = temp task, `PYTHONPATH=<tmp>/task`, 60 s timeout).
17. Step 9: `eval_post` snapshot, then `check_policy(eval_pre, eval_post, allowed_files=[])`.
18. Step 10: `all_violations` = eval/ violations + rejected-path violations + workspace violations; `policy_compliant = not all_violations`.
19. Step 11: `get_price(config.model)`, cost = sum of the per-call `estimated_cost_usd`, `_derive_outcome(..., all_violations)`, assemble the `RunLog`.
20. Step 12: set `log.artifacts_file`, then write the artifacts sidecar (`save_run_artifacts`) and the log (`save_run_log`).
21. `finally` deletes the temp directory, whether the run succeeded or raised.

**Back in the CLI**

22. Prints Run, Status, Hidden x/y, Public x/y, Policy and the log path. Exits 1 if `status != "success"`, otherwise exits normally (0).

### 2.2 Where a run can end early, and what it gets

**A. No run log is written** (the run never reaches step 12):

| Where | Trigger | Outcome |
|---|---|---|
| `argparse` | Missing `--task`/`--arch`, or an invalid `--reasoning-effort` choice | Usage error, exit 2 |
| `cli._load_architecture` | No module `architectures.<arch>`, or no `Architecture` subclass in it | Message on stderr, exit 1 |
| `RunConfig` validators | Current-lineup model without an effort; effort together with `--thinking`; effort on a model with no `reasoning_efforts`; effort not accepted by the model; `--max-output-tokens` ≤ 0 | `ValidationError`, exit 2 |
| `Executor.run` step 2 | `copytree` fails | `RuntimeError("Failed to copy task dir")` propagates. Temp dir removed. Exit 1 (uncaught exception). |
| `Executor._run_architecture` → `get_provider` | Unknown provider name (`ValueError`); missing API key or SDK not installed (`ProviderError` raised in the provider `__init__`) | `get_provider` is called **outside** the `try` around the architecture, so the exception propagates out of `Executor.run`. `finally` removes the temp dir. No log is written; uncaught exception, exit 1. |
| Anything after the architecture step that raises | For example an exception in `snapshot_directory`, reading `evaluated_files`, `_run_public_tests` (only `TimeoutExpired` is caught), or the writers | Propagates; temp dir removed; no log. **[UNVERIFIED]** whether these can happen in practice. |

**B. The architecture stops early, but the run continues to evaluation and a log is written.**
Steps 5–12 still run against whatever was written (often the untouched stub).

| Where | Trigger | `RunResult.status` | Metrics | `error_message` → log `stop_reason` |
|---|---|---|---|---|
| `_run_architecture` | `asyncio.TimeoutError` (wall-clock limit) | `timeout` | `_metrics_from_calls`: tokens, `api_calls` and cost from the recorded calls (including the cancelled in-flight call, which counts as a call with 0 tokens); wall time measured by the executor | `"Timed out after Ns"` |
| `_run_architecture` | Any other exception escaping `run_timed` | `error` | as above | Exception type, message and traceback |
| `SingleAgent.run` / `SequentialPipeline.run` | `render_prompt_template` raises `ValueError` (missing, duplicated or unknown placeholder, e.g. an empty template from an unknown `--prompt-regime`) | `error` | zero, **no API call made** | `"Prompt template error: …"` |
| `SingleAgent.run` / `SequentialPipeline.run` | `ctx.provider is None` (not reachable through the executor) | `error` | zero | `"No provider configured in TaskContext"` |
| `SingleAgent.run` | `ProviderError` from the single call | `error` | `api_calls=1` | Provider message |
| `SequentialPipeline._call_step` | `ProviderError` in any step | `error` (via `_error_result`) | tokens summed over completed steps | `"Step N (role): <error>"` |
| `SingleAgent.run` | Response has no labeled code blocks | `failure` | from the response | `"No labeled code blocks found in LLM response"` |
| `SequentialPipeline.run` | Neither the implement nor the fix response has code blocks | `failure` | summed | `"No labeled code blocks found in pipeline output"` |
| either | Code blocks parsed | `success` | from the responses | none |

An unregistered or misspelled model alias is **not** caught early. `resolve_model` passes it through as a raw API string, provider falls back to `anthropic`, and the API error then surfaces as `ProviderError` → status `error`, with a log written.

**C. Final status (`executor._derive_outcome`)**

1. `success` and none of `modified_files` is in `allowed_edit_files` → `failure`.
2. `success` and `hidden_total == 0`: a collection error (pytest output contains "during collection") → `failure`; otherwise → `error` ("Hidden eval produced no test results"). The second case covers the hidden-eval timeout, a missing script, and a missing hidden-test directory.
3. `success` and `hidden_passed < hidden_total` → `failure`.
4. If the last non-error call's `stop_reason` is not in `{"end_turn","stop_sequence","stop"}`, a note is appended to `stop_reason`. For `max_tokens`/`length` the note is "output may be truncated".
5. **Any policy violation** (from step 10) → `policy_violation`. This overrides every status above, including `error` and `timeout`. The overridden status is recorded in `stop_reason` ("status would otherwise have been '…'"), and the violations are listed in `notes`.

So `success` means every hidden test passed **and** the run was policy-compliant. `RunStatus.BUDGET_EXCEEDED` exists in `architectures/base.py` but nothing produces it.

### 2.3 Budgets as actually enforced

- **Wall clock:** enforced by `asyncio.wait_for` around the architecture only. The hidden eval (120 s) and public tests (60 s) have their own fixed timeouts. Each SDK request has its own 600 s timeout (`REQUEST_TIMEOUT_SECONDS`), so a single response that takes longer than 600 s fails with a `ProviderError` even if the wall-clock budget is larger. **[UNVERIFIED]** how often a 32000-token response would exceed 600 s for a given model.
- **`max_output_tokens`:** sent unchanged as the output cap on **every** call by both architectures (`SingleAgent.run`, `SequentialPipeline._call_step` → `ctx.max_output_tokens`). Neither provider clamps it. It is recorded per call (`request_params.max_output_tokens`) and per run (`budget_config.max_output_tokens_per_call`).
- **`max_tokens` (total budget):** recorded only (`BudgetConfig`, markdown "(not enforced)"). It doesn't affect any call.
- **`max_tool_calls`:** recorded only; no architecture or executor reads it.
- **Retries:** the provider wrappers don't pass `max_retries`, so SDK defaults apply. **[UNVERIFIED]** how many retries that is in the pinned SDK versions. Retried attempts are not visible in the log.

---

## 3. The sequential pipeline (`architectures/sequential.py:SequentialPipeline`)

`run` first calls `render_prompt_template(ctx)`. A template error ends the run as `error` before any API call. It then makes four calls through `_call_step`. Every call uses `model=ctx.model`, `max_tokens=ctx.max_output_tokens`, `temperature=0.0` (dropped by the provider on the reasoning-effort path), and `ctx.seed`, `ctx.thinking_enabled` and `ctx.reasoning_effort`. Each step is a fresh single-turn conversation: one system and one user message, no history. The `step_models` constructor argument is stored but unused (TODO in `__init__`).

| Step | System prompt (`_STEP_CONFIGS`) | User prompt built by | Receives | Returns to the next step |
|---|---|---|---|---|
| 1 plan | "You are a planning agent." | `_build_plan_prompt` (fixed text) | `ctx.task_spec`, `ctx.source_files` (labeled code blocks), and an instruction for a plan. "Do NOT write code" | `plan_text`: the raw response |
| 2 implement | "You are an implementation agent." | `_build_implement_prompt(rendered_template, plan_text)` | **The selected regime's template, rendered exactly as `SingleAgent` renders it** (task spec, source files, editable files, test command and output format, depending on the template), followed by `## Implementation plan` with `plan_text` | `implement_text`: the raw response. It is also parsed into `implement_code`. |
| 3 review | "You are a code review agent." | `_build_review_prompt` (fixed text) | task spec and `implement_text` (the whole raw response). Not the original source files and not the plan. | `review_text` |
| 4 fix | "You are a fixing agent." | `_build_fix_prompt` (fixed text) | task spec, `implement_text`, `review_text`. Not the plan or the original source files. | `fix_text`, parsed into `fix_code` |

Final output: `modified_files = {**implement_code, **fix_code}`. The fixer's version wins for every file it returns. A file it doesn't return keeps its implemented version, and a file only the fixer returns is added. If both are empty the result is `failure`.

The prompt regime therefore affects only step 2. Steps 1, 3 and 4 are identical across regimes (`tests/test_review_fixes.py::TestFix1SequentialRegime`).

`SingleAgent` (`architectures/single_agent.py`) makes one call with system prompt "You are a programming assistant." and `render_prompt_template(ctx)` as the user prompt.

---

## 4. Information boundaries

### 4.1 What reaches the model

The model only ever sees the strings passed to `provider.complete`. Both architectures are single-turn and have no tools. The model cannot read files, run commands, or see any test output.

`Executor._build_context` builds the `TaskContext` from the **temp copy**:

| Field | Source | Sent to the model? |
|---|---|---|
| `task_spec` | `<tmp>/task/README.md` | Yes. Exactly once in the rendered template, and also in the plan, review and fix prompts of `sequential`. |
| `source_files` | every `*.py` under `<tmp>/task/<task>/`, including `__init__.py` | Yes, as labeled code blocks (`format_source_files`). Exactly once in the rendered template, and also in the `sequential` plan prompt. |
| `editable_files` | `TaskPaths.allowed_edit_files` (excludes `__init__.py`), the same list the policy check uses | Through `{editable_files}` (`production.md` only), one `- path` per line, via `resolve_editable_files` |
| example label | first entry of the editable files | Through `{example_file}` in the output-format example of both templates (e.g. `calculator/evaluator.py` for calculator) |
| test command | `architectures/utils.py:TEST_CMD` = `"python -m pytest tests/ -v"` | Through `{test_cmd}` (`production.md` only). The model is told to run it but has no way to. |
| `prompt_template` | `agent-bench/prompts/<regime>.md`, read from the real repo | Rendered by `render_prompt_template` (see below) |
| `public_tests_dir` | `<tmp>/task/tests` | **No.** No architecture reads it; public test contents are never sent. |
| `task_dir`, `model`, budgets, `seed`, `reasoning_effort`, `provider` | config | Not as prompt text. They control the call. |

`render_prompt_template`:

1. `strip_template_header` removes the leading block of lines starting with a single `#` (followed by whitespace or end of line), plus the blank lines after it. In `production.md` this is the title, the "Derived from 143 production agent conversations at Kimara.ai" line, the placeholder list and the description. `## ` headings, and `#` lines further down (such as the comment inside the code example), are kept. `minimal.md` has no header.
2. `_check_placeholders` requires `{task_spec}` and `{source_files}` exactly once, allows `{editable_files}`, `{test_cmd}` and `{example_file}` at most once, and rejects any other placeholder.
3. `str.format` fills the placeholders.

`tests/test_review_fixes.py::TestFix2TemplateRendering` checks every task × regime: spec and source files appear once, no header text is sent, and the example label is the task's own file.

The task READMEs say that hidden tests exist but contain nothing from them. `agent-bench/prompts/prompting_patterns_summary.md` is in the prompts directory. It has no placeholders, so selecting it as a regime fails at rendering with a template error.

### 4.2 What never reaches the model

- **`eval/`** (hidden tests, `conftest.py`, `run_eval.sh`, `passk.py`): no code path reads them into a prompt. The hidden eval runs after the architecture has returned (step 7). Its results go to the log and are never fed back to the model.
- **Public test file contents and results:** public tests run after the architecture, at step 8.
- **`runs/`:** written only by the logger and `run_batch.py`; never read during a run.

### 4.3 Limits of the boundary, as implemented

- The boundary is about **what goes into the prompt**, not an OS-level sandbox. The code the model writes is **executed** by the harness during the hidden and public pytest runs, as the user running `bench`, with `PYTHONPATH` including the repo root. Nothing in the code stops that generated code from reading `eval/` or `runs/` at test time. The eval snapshot detects **modifications** to `eval/`, not reads.
- Writes outside the temp task directory through the model's file labels are prevented by `_apply_modified_files`. This is tested for `..`, absolute paths and a symlink inside the workspace. Generated code that writes files *when executed* by pytest is not prevented. Such writes are detected only if they land in `eval/` or (before step 6) in the workspace, and in practice they happen after step 6.
- Files written into the workspace outside the editable set (e.g. a `game.py` label instead of `snake/game.py`, or overwriting `tests/…`) are written and evaluated. They are reported as violations, and the run's status becomes `policy_violation`.

---

## 5. What is written to the run log

Each run writes three files to `runs/`. All share the stem from `make_run_filename`: `{YY-MM-DD_HHMM in Europe/Helsinki}_{provider}_{shorten_model(model)}_{architecture}_{task}_{regime}_{run_id[:8]}`.

- `<stem>.json`: `RunLog.model_dump(mode="json")` (`save_run_log`)
- `<stem>.md`: `_render_markdown`
- `<stem>.artifacts.json`: `RunArtifacts` (`save_run_artifacts`). `RunLog.artifacts_file` holds its filename.

`logger.is_run_log_file` distinguishes main logs from `*.artifacts.json` sidecars and `batch_*.json` summaries.

### 5.1 Run level (`RunLog`, assembled in `Executor.run` step 11)

| Field | Value and source |
|---|---|
| `run_id`, `timestamp` | `RunConfig` defaults (UUID4; UTC time at config creation) |
| `git_commit`, `git_branch` | `get_git_info()`, run in the **process cwd** (not `repo_root`) |
| `git_dirty` | `get_git_dirty(cwd=repo_root)`; `None` if unknown |
| `task_name`, `architecture_name`, `prompt_regime` | from `RunConfig` (`architecture_name` is the CLI module name, e.g. `sequential`) |
| `model`, `provider` | the resolved API string and the provider used |
| `requested_reasoning_effort` | `RunConfig.reasoning_effort` |
| `budget_config` | `max_tokens`, `max_tool_calls`, `max_wall_clock_seconds`, `max_output_tokens_per_call` (`None` in older logs) |
| `sampling` | `SamplingConfig.from_calls(llm_calls)`: per parameter, the shared value if all calls agree, else `None` with `consistent_across_calls=False`. Calls that errored are ignored. |
| `actual_metrics.tokens_*`, `api_calls` | Architecture-reported `RunResult.metrics` when the architecture returns. On timeout or unhandled exception: `executor._metrics_from_calls`, i.e. from the recorded calls. |
| `actual_metrics.wall_clock_seconds` | `run_timed` measurement when the architecture returns; the executor's own measurement on timeout or exception. Architecture time only (excludes eval). |
| `actual_metrics.estimated_cost_usd` | if `config.model` is priced: the sum of the per-call `estimated_cost_usd` (calls that errored or were cancelled contribute 0). Otherwise `result.metrics.estimated_cost_usd`: 0.0 when the architecture returned, or the recorded-call sum (also 0 for unpriced models) on timeout or exception. |
| `actual_metrics.price_source` | `Price.source` (URL plus retrieval date) or `None` |
| `test_results` | public and hidden passed/total. `failing_test_names` is the hidden list followed by the public list, including sentinels (`hidden_eval_collection_error`, `hidden_eval_no_results: …`, `eval_timeout`, `eval_script_not_found`, `public_tests_timeout`). |
| `llm_calls` | per-call records (5.2) |
| `status` | from `_derive_outcome` (2.2 C) |
| `policy_compliant` | `True` if step 10 found no violations |
| `stop_reason` | architecture `error_message` plus the `_derive_outcome` reasons, joined with `"; "` |
| `notes` | violation strings (`added:/modified:/deleted: <path>`, `rejected (outside workspace): <path>`), joined with `"; "` |
| `artifacts_file` | sidecar filename (`None` in older logs) |

The markdown file renders the same data, adding "(not enforced)" next to the token and API-call budgets and linking the artifacts file.

### 5.2 Per call (`LLMCallRecord` in the log; `CallTranscript` in the sidecar; both written by `_RecordingProvider.complete`, index-aligned)

| Field | Successful call | Failed call | Cancelled by timeout |
|---|---|---|---|
| `model` | `response.model` (may be a dated snapshot) | requested model | requested model |
| `stop_reason` | provider-native (`end_turn`, `max_tokens`, `stop`, `length`, `refusal`, …) | `None` | `None` |
| `input_tokens`, `output_tokens` | from the provider usage; output includes thinking/reasoning tokens | 0 | 0 (usage unknown; the provider may still bill it) |
| `request_params` | `temperature`, `max_output_tokens`, `seed`, `reasoning_effort` (sent), `thinking_budget_tokens`, `thinking_type`, `requested_reasoning_effort`, `sent_params` | `{}` | `{}` |
| `usage_details` | OpenAI: `reasoning_tokens`, `cached_input_tokens`; Anthropic: empty | `{}` | `{}` |
| `estimated_cost_usd` | `estimate_call_cost(<requested model>, …)` or `None` if unpriced | `None` | `None` |
| `error` | `None` | `str(ProviderError)`, or `"<Type>: <msg>"` for other exceptions | `"cancelled before a response arrived (wall-clock timeout); usage unknown"` |
| sidecar `CallTranscript` | `system_prompt`, `user_prompt`, `response_text` | prompts + `error` | prompts + `error` |

### 5.3 The artifacts sidecar (`RunArtifacts`)

| Field | Content |
|---|---|
| `run_id` | same as the log |
| `calls` | one `CallTranscript` per call (5.2) |
| `agent_log` | `RunResult.agent_log` as returned by the architecture: prompt and response in `SingleAgent`; step, role, prompt, response and tokens in `SequentialPipeline`. Empty on timeout or exception. |
| `modified_files` | files exactly as parsed from the model output, including rejected paths |
| `evaluated_files` | contents of the editable files in the workspace when the tests ran |

Still **not** recorded anywhere: thinking/reasoning text, the raw SDK response, `RunMetrics.tool_calls`, pytest stdout/stderr (apart from parsed names and the last line in the no-results sentinel), and SDK-level retries.

### 5.4 Downstream consumers

- `scripts/run_batch.py` writes `runs/batch_<YY-MM-DD_HHMMSS>.json` with the batch arguments and one outcome per run. It counts every non-`SUCCESS` run under `errors`.
- `scripts/aggregate_results.py`:
  - `load_runs` reads every `runs/*.json` except `*.artifacts.json` and `batch_*.json`. Any other file without a run log's required keys is reported on stderr as skipped. **No run is dropped because of its status.**
  - `group_key` groups by (task, provider, model API string, architecture, regime, requested reasoning effort); `-` stands for a missing value in older logs.
  - Per group it reports `n`, `err` (status `error`), `tmo` (status `timeout`), and `ncmp` (`policy_compliant` false or status `policy_violation`).
  - **Strict pass@1** is the share of runs with status `success` that are also compliant (`_strict_pass`). It also reports the mean ± sample std of the per-run hidden pass rate (0/0 counts as 0), mean tokens, mean latency, and mean cost over priced runs (`-` if none is priced, `*` if only some are).
  - `--csv PATH` writes the rows. A positional task argument filters to one task.

---

## Changes since `1c0612b`

| # | Change | Where |
|---|---|---|
| 1 | Sequential implement step renders the selected regime template (same renderer as `SingleAgent`) with the plan appended | `sequential.py:_build_implement_prompt`, `run` |
| 2 | Template header stripped; spec and source placeholders required exactly once; example label is `{example_file}` | `utils.py:render_prompt_template`, `strip_template_header`, `_check_placeholders`; `agent-bench/prompts/*.md` |
| 3 | Fix-step files merged over implement-step files | `sequential.py:run` |
| 4 | `max_output_tokens` (default 32000) sent unchanged on every call; `max_tokens // 4` split and the provider clamp to 16384 removed; comments claiming the executor enforces the budget removed | `config.py`, `cli.py`, `base.py:TaskContext`, both architectures, both providers |
| 5 | Prompts, responses, parsed and evaluated files saved in `<stem>.artifacts.json` | `executor.py:_RecordingProvider`, `run`; `logger.py:RunArtifacts`, `save_run_artifacts` |
| 6 | Timeout and exception runs take tokens, calls and cost from recorded calls and wall time from the executor; cancelled and non-provider-error calls are recorded | `executor.py:_run_architecture`, `_metrics_from_calls`, `_RecordingProvider` |
| 7 | Paths resolving outside the workspace are rejected and recorded as violations | `executor.py:_apply_modified_files` |
| 8 | Any policy violation sets status `policy_violation` | `executor.py:_derive_outcome`; `base.py:RunStatus` |
| 9 | Editable files shown to the model = `allowed_edit_files` | `executor.py:_build_context`; `utils.py:resolve_editable_files` |
| 10 | Aggregation by six dimensions, with counts, strict pass@1 and cost; nothing dropped | `scripts/aggregate_results.py` |

Tests: `tests/test_review_fixes.py` (one class per change), plus updates in `tests/test_providers.py`, `tests/test_sequential.py` and `tests/test_harness.py`.

# Known issues

Checked against the code and run logs on 2026-11-09 at commit `7fb81a6`. **[UNCERTAIN]** marks an inference or something the repository doesn't settle. Code references are to the current tree.

## Limitations that affect how results can be read

### Runtime isolation is prompt-level only

The hidden tests are kept from the model because they're never put into a prompt. There's no OS-level sandbox. The code the model writes is **executed** during scoring, as the user who ran `bench`, with the repo root on `PYTHONPATH` (`eval/run_eval.sh`, `runner/executor.py`).

Nothing stops generated code from reading `eval/` or `runs/` while tests run, or from writing files or using the network. The SHA-256 check on `eval/` detects changes, not reads. Writes made during test execution happen after the workspace snapshot, so outside `eval/` they aren't detected at all.

Writes through the model's file labels are protected: paths that resolve outside the temp workspace are rejected. Don't run the harness somewhere the generated code could do damage. See `docs/harness_architecture.md`, section 4.3.

### SDK retries are not logged

Neither provider wrapper sets `max_retries`, so SDK defaults apply: 2 retries in both pinned SDKs (`anthropic` 1.17.0 and `openai` 3.36.1, `DEFAULT_MAX_RETRIES`). Each log entry is one logical call. Retried attempts aren't recorded, though the time they take is included in the run's latency. So `api_calls` is a lower bound on real HTTP requests, and a run affected by retries looks the same as one that wasn't.

### Diff Applier: the trailing newline on an empty original is ambiguous

The README (`agent-bench/tasks/diffapply/README.md`, line 164) says that if the original ends with `\n`, so should the result. An empty original doesn't end with `\n`, so the rule doesn't say what to return. `test_empty_original_with_additions` requires `"line1\nline2\n"`.

This affects scores. The test failed in 22 of the 120 main-experiment diffapply runs: 16 of `gpt-6-luna`'s 20 and 6 of `sonnet-5.5`'s 20. In 16 of those 22 runs it was the only failing hidden test, so those runs failed pass@1 for this reason alone (10 for `gpt-6-luna`, all 6 for `sonnet-5.5`).

Calling each failing run's code on that input showed 18 of the 22 returning `"line1\nline2"`, with no trailing newline, which is a reasonable reading of the spec. The other 4 raised a `ValueError`, which is a different bug. `docs/task_summary.md` lists this test but doesn't flag the ambiguity.

### URL Shortener: several README requirements have no hidden test

The README requires these, but no hidden test checks them (`docs/task_summary.md`, section 3):

| Requirement | Coverage |
|---|---|
| `resolve` raises `ValueError` mentioning "expired" for an expired link | Public test only |
| Unknown code raises `KeyError` in `resolve` and `get_stats` | None |
| Duplicate custom code raises `ValueError` | None. `test_custom_code_collision` covers auto-generated codes skipping taken ones. |
| `shorten` raises a "rate limit" `ValueError` | None through `shorten`. Only `RateLimiter.is_allowed` is tested directly. |
| Rate limit is checked before URL validation | None |
| Code space exhausted raises `RuntimeError` | None |

Fixed and sliding windows both pass the window-reset test. So does a limiter that counts in `is_allowed` with a no-op `record`. A run can therefore pass every hidden test without meeting the full spec, and pass@1 on this task is an upper bound on spec compliance.

### Results come from three harness versions

| Tag | Tasks | Code-block parser |
|---|---|---|
| `experiment-v1` | diffapply | Old |
| `experiment-v1.1` | calculator, mdtable | Old |
| `experiment-v1.2` | urlshort | Fixed |

The old parser cut off code containing a literal ```. It changed the result of 26 of the 120 mdtable runs, 2 of them from failure to success, and no others. Reported figures need `--rescored`. Re-scored runs use today's task files, and their public tests aren't re-run. Between the tags, only the specs and a docstring changed for calculator, mdtable and urlshort; diffapply's spec and tests didn't change after `experiment-v1`.

### Other limits on the comparison

- **The tag rule isn't enforced.** The first 72 diffapply runs ran on `main`, at the tagged commit.
- **Budgets are mostly recorded, not enforced.** `max_tokens` and `max_tool_calls` affect nothing, and `RunStatus.BUDGET_EXCEEDED` is never produced. Only `--max-output-tokens` (per call) and `--max-wall-clock` (architecture only) take effect. The sequential pipeline gets the same per-call cap as single agent on each of its four calls, so its total budget is larger.
- **Runs aren't deterministic.** The main experiment set no seed, Anthropic never receives one, and no temperature is sent on the reasoning-effort path.
- **Five repetitions per cell is a small sample.** `eval/passk.py` exists but nothing in the harness calls it.
- **No tools or iteration.** Both architectures are single-turn text exchanges, so the model never sees a test result. The production prompt tells it a test command that it can't run.
- **Cost is an estimate** from list prices retrieved on 2026-11-04. An in-flight call cancelled by a timeout counts as zero.
- **Snake isn't in the main experiment.** It's used only for smoke tests.
- **Reviews are classified by keyword.** The "review reported issues" column in `analyze_stages.py` is a heuristic.
- **[UNCERTAIN]** How well five small Python libraries generalise to real software development. The repository doesn't address this.

## Tooling problems

### Batch runner (`scripts/run_batch.py`)

- No resume, no stop-on-error, no credit check. See [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md).
- The summary is written only at the end, so an interrupted batch has none.
- For a provider error, the `error` field holds the line of `=` signs, the last line of `bench`'s stdout, rather than the error. The real message is in the run log's `stop_reason`.
- `errors` counts ordinary test failures as well. The script exits 0 even if every run failed.
- `--dry-run` doesn't validate aliases, keys or efforts.

### Environment check (`scripts/check_env.py`)

- Fails without `ANTHROPIC_API_KEY`, even for OpenAI-only use.
- Looks for `GOOGLE_API_KEY`, and reports Google as ready if it's set, though no Google provider exists.
- Doesn't check that pytest is installed. Without it, runs still make their paid calls and then end as `error`.
- Doesn't test whether keys work.
- Prints the first 12 characters of each key, so take care if sharing its output.

### Setup and documentation

- `.env.example` doesn't exist. The README explains how to write `.env`.
- Run logs aren't in git (`runs/*.json`, `runs/*.md`), so results can't be regenerated from a clone without a separate copy of the 57 MB of logs.
- The documented public-test commands fail with `ModuleNotFoundError`: `pytest agent-bench/tasks/snake/tests/` (`CLAUDE.md`) and `pytest tests/` (`agent-bench/README.md`). `python -m pytest tests/` from the task directory works.
- `CLAUDE.md` and `STAGES.md` are out of date. `CLAUDE.md` says the provider layer is still to be built and its structure listing omits `scripts/`, `docs/` and `analysis/`. The status section of `STAGES.md` is dated 2026-03-05.
- The Snake stub's comment points to `agent-bench/README.md` for the spec; the spec is in `agent-bench/tasks/snake/README.md`. The stub is sent to the model, so changing it would change the prompt.

### Harness behaviour

- `--model` defaults to the legacy `sonnet-4.6`.
- A misspelled alias isn't rejected. It falls back to the `anthropic` provider and fails at the API call, which writes an `error` log.
- The hidden tests use `.venv/bin/python` if it exists (`eval/run_eval.sh`), while the public tests use the interpreter running `bench`. They can differ if `bench` is launched from another environment.
- Only `PASSED` and `FAILED` lines are parsed. An `ERROR` in a fixture lowers the total silently (`runner/executor.py:_parse_pytest_output`).
- `hidden_failing` in `scripts/analyze_stages.py` recognises hidden tests by ids starting with `eval/`, which holds only when scoring runs from the repository root, as the executor does.
- `SequentialPipeline`'s `step_models` argument is stored but unused, so every step uses the same model.
- `classify_failures.py` reads ambiguity titles from a numbered heading in `docs/task_summary.md`, and only covers mdtable.
- Pilot logs in `runs/` are included by default in the analysis scripts. The pilot Snake run from March shows `success` with hidden 0/0 and a policy violation, from before those cases were guarded against.
- No reference solutions are kept, so the mutation checks behind the hidden tests can't be replayed.
- A hidden-test change after runs exist isn't tracked per run except through `git_commit`. Diffapply pilot runs have 15 hidden tests; the main experiment has 16.

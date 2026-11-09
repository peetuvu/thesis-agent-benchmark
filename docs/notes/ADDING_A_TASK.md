# Adding a task

A task is a small pure-Python library that the model has to implement from a written spec. Adding one needs no harness changes: the runner finds a task by its directory name. The examples below use a hypothetical task called `mytask`.

## Layout

```text
agent-bench/tasks/mytask/          # visible to the model, in part
├── README.md                      # the spec, sent to the model in full
├── pytest.ini                     # copy from an existing task
├── mytask/                        # must have the same name as the task
│   ├── __init__.py                # exports the public API; shown, not editable
│   └── core.py                    # stub the model implements
└── tests/
    └── test_public_sanity.py      # public tests

eval/hidden_tests/mytask/          # never shown to the model
├── conftest.py
└── test_hidden.py
```

`runner/config.py:TaskPaths` builds these paths from the task name, so the package directory must match it. The hidden test files live in `eval/`, which is outside everything that goes into a prompt.

### README.md (the spec)

The whole file is sent to the model. It must state everything the hidden tests require, without quoting them. Saying that hidden tests exist is fine. The existing tasks end with an "Output format" section showing labeled code blocks. Follow that pattern, using the real file paths:

````markdown
## Output format

Return complete file contents in a labeled code block:

```mytask/core.py
# complete file contents here
```
````

A file only counts if its label is the path relative to the task directory, and the label must contain a dot.

### Stubs and `allowed_edit_files`

There's no configuration list. `TaskPaths.allowed_edit_files` is computed as every `*.py` file under `mytask/`, at any depth, except files named `__init__.py`.

So:

- Every file the model should write has to exist as a stub. A stub should contain the signatures and raise `NotImplementedError` or similar.
- All `*.py` files in the package, `__init__.py` included, are sent to the model. Non-Python files aren't.
- `__init__.py` isn't editable. Changing it, or writing any other file outside the editable set, makes the run a `policy_violation`, which overrides every other status.
- Don't add any type or function to `__init__.py` that the model would need to change.

### Public tests

Put these in `tests/`. Their contents are **not** sent to the model, which also can't run them (both architectures are single-turn with no tools). They're run after the model finishes and logged. `pytest.ini` is identical in all five tasks; copy it.

The tests import the package by name, for example `from mytask import ...`. The harness sets `PYTHONPATH` to the temp copy of the task.

### Hidden tests and `conftest.py`

Copy this `conftest.py` to `eval/hidden_tests/mytask/` and change only the task name:

```python
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # thesis-agents/

AGENT_BENCH = Path(
    os.environ.get("BENCH_SOURCE_DIR", str(ROOT / "agent-bench" / "tasks" / "mytask"))
)

if str(AGENT_BENCH) not in sys.path:
    sys.path.insert(0, str(AGENT_BENCH))
```

**What `BENCH_SOURCE_DIR` does.** Each run works in a temp copy of the task. `eval/run_eval.sh <task> <temp dir>` sets `BENCH_SOURCE_DIR` to that copy, and `conftest.py` puts it first on `sys.path`, so `import mytask` finds the model's code. A manual `bash eval/run_eval.sh mytask` has no temp copy, so it points the variable at the stub in `agent-bench/`. The fallback in `conftest.py` does the same if you run pytest directly.

Constraints from how results are parsed (`runner/executor.py:_parse_pytest_output`):

- Test functions must be named `test_*`.
- Avoid parametrized ids containing spaces. The parser matches ids with no whitespace, so these wouldn't be counted.
- Only `PASSED` and `FAILED` lines are counted. An error in a fixture produces `ERROR`, which isn't, so the total comes out lower with no warning. The existing tasks have no parametrized tests, and hidden tests should avoid failing fixtures.
- Hidden tests have 120 seconds in total and public tests 60.
- Don't name a task after a directory in the repository root (`eval`, `runner`, `providers`, and so on). The repo root is on `PYTHONPATH` during scoring.
- Tests can only use packages already in the environment. `pyproject.toml` has no task-specific dependencies.

## Checklist

The first two sections are the audit used on the existing tasks. Do them **before any paid run**. Finding a contradiction after a batch means re-running it.

### 1. Audit the README against the hidden tests

Go through every hidden test and find the README sentence that justifies its assertion.

- [ ] Every assertion traces to a statement or example in the README.
- [ ] The README's examples don't contradict its own rules or the tests.
- [ ] Each exact value the tests compare (error types and messages, whitespace, ordering, trailing newlines) is specified.
- [ ] Each ambiguity you leave in on purpose is recorded with the reading the tests assume, in `docs/task_summary.md`.
- [ ] The README contains no hidden-test code.

The audit before the main experiment found real problems. The calculator's precedence table contradicted its own examples, and the URL shortener's port example used `localhost:8080`, which the dot rule forbids (commits `0931f26` and `ffcb7b9`). In mdtable, handling of `\|` was deliberately left unclarified and recorded.

An audit can still miss things. A Diff Applier newline ambiguity got through and affected results; see [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Pay particular attention to edge cases the README mentions only in passing.

### 2. Verify each hidden test can fail for the reason in its name

A test that passes for the wrong reason measures nothing. For each hidden test:

- [ ] Write a throwaway implementation, kept **outside the repository**, that passes all the tests.
- [ ] Break the specific behaviour the test's name describes, leaving everything else correct.
- [ ] Confirm that test fails, and that it fails for that reason.
- [ ] Check that the test's name, docstring and comments describe what it asserts.

The real example: `test_overlapping_hunks_raise` in diffapply. A throwaway applier with **no** overlap check passed it, because its `ValueError` came from a different check. That led to adding `test_overlapping_hunks_with_consistent_content_raise`, where only an overlap check can raise (see `docs/task_summary.md`, section 5).

A test can also just be named misleadingly. `test_self_collision_kills_after_tail_handling` in Snake asserts death when the head moves onto the current tail, which matches the README, but the name suggests the opposite.

No reference solutions are stored in the repository, so this has to be done for each new task.

### 3. Check the mechanics

These are offline.

- [ ] `pytest` passes. `tests/test_review_fixes.py` picks up every directory under `agent-bench/tasks/` automatically and checks that each task renders correctly under both prompt regimes.
- [ ] `bash eval/run_eval.sh mytask` against the stub collects every hidden test and fails them.
- [ ] `cd agent-bench/tasks/mytask && python -m pytest tests/ -v` against the stub collects every public test and fails them. Use `python -m pytest`, not bare `pytest`, or the package won't be importable.
- [ ] Add the task to `docs/task_summary.md`: spec summary, editable files, test counts, ambiguities.

### 4. Run it

- [ ] Run one real smoke test for the task on a cheap model, then move its files to `runs_smoke/`:
  ```bash
  bench run --task mytask --arch single_agent --model haiku-5.5 --reasoning-effort low
  ```
- [ ] Confirm `hidden_total` and `public_total` in the log equal the counts you expect.
- [ ] Commit, tag, and run from the tag ([RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md)).

If you change a spec or hidden test after runs exist, those runs are no longer strictly comparable. For example, diffapply went from 15 to 16 hidden tests, so pilot runs scored against 15 don't compare with the main experiment.

`classify_failures.py` is specific to mdtable, so a new task gets no failure classification unless you write probes for it. `analyze_stages.py`, `aggregate_results.py` and `rescore_runs.py` work for any task.

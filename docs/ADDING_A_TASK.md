# Adding a task

## Folder layout

```text
agent-bench/tasks/mytask/
├── README.md                  # spec sent to the model; end with an Output format section
├── pytest.ini                 # copy from another task
├── mytask/                    # package name must equal the task name
│   ├── __init__.py            # shown to the model, not editable
│   └── core.py                # stub; every other *.py here is editable
└── tests/
    └── test_public_sanity.py  # never sent to the model
eval/hidden_tests/mytask/
├── conftest.py
└── test_hidden.py
```

Label each file in the Output format section with its path, for example `mytask/core.py`.

Avoid repo-root directory names for tasks, because the repo root is on `PYTHONPATH` during scoring.

## conftest.py

```python
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root

AGENT_BENCH = Path(
    os.environ.get("BENCH_SOURCE_DIR", str(ROOT / "agent-bench" / "tasks" / "mytask"))
)

if str(AGENT_BENCH) not in sys.path:
    sys.path.insert(0, str(AGENT_BENCH))
```

Keep `BENCH_SOURCE_DIR`, because it points the tests at each run's temp copy.

Name tests `test_*` and keep parametrized ids free of spaces, because the result parser skips anything else.

## Checklist

Complete this before any paid run.

- [ ] Stub every file the model writes, with signatures only.
- [ ] Trace every hidden-test assertion to a sentence or example in the README.
- [ ] Record each deliberate ambiguity in `docs/task_summary.md`.
- [ ] Break each tested behaviour in a throwaway solution outside the repo and confirm that test fails for the reason in its name.
- [ ] Run `bash eval/run_eval.sh mytask` and confirm every hidden test fails on the stub.
- [ ] Run `cd agent-bench/tasks/mytask && python -m pytest tests/ -v` and confirm every public test fails on the stub.
- [ ] Run `pytest` from the repo root and confirm it passes.
- [ ] Run `bench run --task mytask --arch single_agent --model haiku-5.5 --reasoning-effort low  # untested` and confirm the test totals in the log.

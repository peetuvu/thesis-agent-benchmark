#!/usr/bin/env python3
"""Re-score every run with the fixed code-block parser.

Usage:
  python3 scripts/rescore_runs.py [--runs-dir runs] [--out-dir analysis] [--task T ...]

Run from the thesis-agents repo root. Makes no API calls and writes nothing
to eval/ or runs/ (an output directory inside either is refused).

Background: parse_code_blocks used to end a labeled block at the first ```
anywhere, so code containing a literal ``` (e.g. startswith("```")) was cut
mid-line and usually failed to collect. The parser now only closes a block at
a line consisting of backticks (at least as many as the opening fence) plus
whitespace. This script re-applies the fixed parser to the responses saved in
each run's .artifacts.json.

For every run log in runs/ (all tasks unless --task is given):

  1. Rebuild the final files exactly as the architecture does:
       single_agent  parse the assistant response;
       sequential    parse the implement and fix responses and merge fix
                     over implement ({**implement, **fix}).
  2. Rebuild the evaluated files as the executor does: copy the task to a
     fresh temp dir, write the files with Executor._apply_modified_files, read
     back every allowed edit file.
  3. If they equal the logged evaluated_files: outcome "unchanged".
     Otherwise: outcome "changed"; run the hidden tests on the temp dir
     (Executor._run_hidden_eval, as analyze_stages.py's scorer does) and
     derive the new status with the executor's own _derive_outcome, using the
     logged LLM calls (so truncation notes behave as at run time). A run that
     was not policy-compliant stays a policy violation.

Other outcomes: "no_artifacts" (no sidecar), "not_rebuildable" (a required
response is missing, e.g. the pipeline errored before the fix step; the
logged result stands), "unsupported_architecture".

Caveats
  - Changed runs are scored against today's hidden tests and task files.
    diffapply runs logged before 2026-11-05 had 15 hidden tests, now 16.
  - Public tests are not re-run; public results stay as logged.

Outputs
  - <out-dir>/rescore_<task>.csv, one row per run (read by
    aggregate_results.py, analyze_stages.py and classify_failures.py with
    --rescored)
  - stdout per task: runs checked, unchanged, changed, other outcomes, and
    old -> new status counts over changed runs
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for _p in (str(_REPO_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_stages import _check_output_path, hidden_failing  # noqa: E402
from architectures.base import RunResult, RunStatus  # noqa: E402
from architectures.utils import parse_code_blocks  # noqa: E402
from runner.config import TaskPaths  # noqa: E402
from runner.executor import Executor, _derive_outcome  # noqa: E402
from runner.logger import LLMCallRecord  # noqa: E402

RUNS_DIR = Path("runs")
OUT_DIR = _REPO_ROOT / "analysis"
_REQUIRED_KEYS = ("run_id", "task_name", "model", "architecture_name", "status", "test_results")

UNCHANGED = "unchanged"
CHANGED = "changed"
NO_ARTIFACTS = "no_artifacts"
NOT_REBUILDABLE = "not_rebuildable"
UNSUPPORTED = "unsupported_architecture"


# ---------------------------------------------------------------------------
# Rebuilding the architecture's output
# ---------------------------------------------------------------------------


def _role_text(agent_log: list[dict[str, Any]], role: str, key: str) -> str | None:
    for entry in agent_log:
        if entry.get("role") == role:
            return entry.get(key)
    return None


def rebuild_final_files(
    architecture: str, artifacts: dict[str, Any],
) -> tuple[dict[str, str] | None, str]:
    """(files, note) as the architecture would build them with today's parser.

    files is None when the run cannot be rebuilt; note says why.
    """
    agent_log = artifacts.get("agent_log") or []
    if architecture == "single_agent":
        text = _role_text(agent_log, "assistant", "content")
        if text is None:
            return None, "no assistant response"
        return parse_code_blocks(text), ""
    if architecture == "sequential":
        impl = _role_text(agent_log, "implement", "response")
        fix = _role_text(agent_log, "fix", "response")
        if impl is None or fix is None:
            return None, "no implement/fix response (pipeline stopped early)"
        return {**parse_code_blocks(impl), **parse_code_blocks(fix)}, ""
    return None, f"architecture {architecture!r} not supported"


# ---------------------------------------------------------------------------
# Per run
# ---------------------------------------------------------------------------


@dataclass
class RescoreRow:
    """One run: logged vs re-scored result."""

    run_id: str
    log_file: str
    task: str
    provider: str
    model: str
    effort: str
    architecture: str
    regime: str
    outcome: str
    files_changed: str = ""
    old_status: str = ""
    new_status: str = ""
    old_hidden_passed: int = 0
    old_hidden_total: int = 0
    new_hidden_passed: int = 0
    new_hidden_total: int = 0
    old_policy_compliant: bool | None = None
    new_policy_compliant: bool | None = None
    new_stop_reason: str = ""
    old_hidden_failing: str = ""
    new_hidden_failing: str = ""
    note: str = ""


# Evaluator signature: (task, files) -> (evaluated_files, apply_violations,
# scorer) where scorer() runs the hidden tests on the same rebuilt dir.
Evaluator = Callable[[str, dict[str, str]], Any]


class TempDirEvaluator:
    """Rebuilds files in a temp copy of a task and scores them on demand."""

    def __init__(self, repo_root: Path = _REPO_ROOT) -> None:
        self.repo_root = repo_root
        self.executor = Executor(repo_root=repo_root)

    def rebuild_and_score(
        self, task: str, files: dict[str, str], need_score: Callable[[dict[str, str]], bool],
    ) -> tuple[dict[str, str], list[str], tuple[int, int, list[str]] | None]:
        """(evaluated_files, apply_violations, hidden score or None).

        The hidden tests run only if need_score(evaluated_files) is True.
        """
        paths = TaskPaths(task, self.repo_root)
        tmp = tempfile.mkdtemp(prefix=f"rescore_{task}_")
        try:
            task_dir = Path(tmp) / "task"
            shutil.copytree(paths.task_dir, task_dir)
            violations = self.executor._apply_modified_files(files, task_dir)
            evaluated = {
                rel: (task_dir / rel).read_text()
                for rel in paths.allowed_edit_files
                if (task_dir / rel).is_file()
            }
            score = None
            if need_score(evaluated):
                score = self.executor._run_hidden_eval(task, task_dir)
            return evaluated, violations, score
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _calls(log: dict[str, Any]) -> list[LLMCallRecord]:
    out = []
    for c in log.get("llm_calls") or []:
        try:
            out.append(LLMCallRecord.model_validate(c))
        except Exception:  # noqa: BLE001 - an old/odd record must not stop the run
            continue
    return out


def new_outcome(
    log: dict[str, Any], files: dict[str, str], allowed: list[str],
    score: tuple[int, int, list[str]], apply_violations: list[str],
) -> tuple[str, str, bool]:
    """(status, stop_reason, policy_compliant) for the rebuilt files."""
    passed, total, failing = score
    if files:
        result = RunResult(status=RunStatus.SUCCESS, metrics=None, modified_files=files)  # type: ignore[arg-type]
    else:
        msg = ("No labeled code blocks found in LLM response"
               if log["architecture_name"] == "single_agent"
               else "No labeled code blocks found in pipeline output")
        result = RunResult(status=RunStatus.FAILURE, metrics=None,  # type: ignore[arg-type]
                           modified_files={}, error_message=msg)
    violations = list(apply_violations)
    if log.get("policy_compliant") is False:
        violations.append(f"logged run was not policy-compliant: {log.get('notes') or '?'}")
    status, stop = _derive_outcome(result, passed, total, failing, allowed, _calls(log), violations)
    return status, stop or "", not violations


def rescore_run(
    log_path: Path, log: dict[str, Any], evaluator: TempDirEvaluator,
) -> RescoreRow:
    """Rebuild, compare and (if changed) re-score one run."""
    tr = log["test_results"]
    old_failing = hidden_failing(tr.get("failing_test_names") or [])
    row = RescoreRow(
        run_id=log["run_id"], log_file=log_path.name, task=log["task_name"],
        provider=log.get("provider") or "", model=log["model"],
        effort=log.get("requested_reasoning_effort") or "",
        architecture=log["architecture_name"], regime=log.get("prompt_regime") or "",
        outcome="", old_status=log["status"], new_status=log["status"],
        old_hidden_passed=tr.get("hidden_passed", 0), old_hidden_total=tr.get("hidden_total", 0),
        new_hidden_passed=tr.get("hidden_passed", 0), new_hidden_total=tr.get("hidden_total", 0),
        old_policy_compliant=log.get("policy_compliant"),
        new_policy_compliant=log.get("policy_compliant"),
        old_hidden_failing=";".join(old_failing), new_hidden_failing=";".join(old_failing),
    )

    art_name = log.get("artifacts_file")
    art_path = log_path.parent / art_name if art_name else None
    if art_path is None or not art_path.is_file():
        row.outcome = NO_ARTIFACTS
        return row
    try:
        artifacts = json.loads(art_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        row.outcome, row.note = NO_ARTIFACTS, f"unreadable artifacts: {e}"
        return row

    files, why = rebuild_final_files(row.architecture, artifacts)
    if files is None:
        row.outcome = UNSUPPORTED if "not supported" in why else NOT_REBUILDABLE
        row.note = why
        return row

    logged_eval = artifacts.get("evaluated_files")
    basis = "evaluated_files"
    if logged_eval is None:
        logged_eval, basis = artifacts.get("modified_files") or {}, "modified_files"

    def differs(evaluated: dict[str, str]) -> bool:
        return (evaluated if basis == "evaluated_files" else files) != logged_eval

    evaluated, violations, score = evaluator.rebuild_and_score(row.task, files, differs)
    if score is None:
        row.outcome = UNCHANGED
        return row

    cmp_new = evaluated if basis == "evaluated_files" else files
    row.files_changed = ";".join(sorted(
        k for k in set(cmp_new) | set(logged_eval) if cmp_new.get(k) != logged_eval.get(k)
    ))
    row.outcome = CHANGED
    if basis != "evaluated_files":
        row.note = "compared against modified_files (no evaluated_files logged)"
    allowed = TaskPaths(row.task, evaluator.repo_root).allowed_edit_files
    row.new_hidden_passed, row.new_hidden_total, new_failing = score
    row.new_hidden_failing = ";".join(new_failing)
    row.new_status, row.new_stop_reason, compliant = new_outcome(
        log, files, allowed, score, violations,
    )
    row.new_policy_compliant = compliant
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_all_runs(runs_dir: Path, tasks: list[str] | None) -> list[tuple[Path, dict[str, Any]]]:
    """Every run log in *runs_dir* (optionally only some tasks), sorted by name."""
    out = []
    for path in sorted(runs_dir.glob("*.json")):
        name = path.name
        if name.endswith(".artifacts.json") or name.startswith("batch_"):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not all(k in data for k in _REQUIRED_KEYS):
            continue
        if tasks and data["task_name"] not in tasks:
            continue
        out.append((path, data))
    return out


def print_task_summary(task: str, rows: list[RescoreRow], file: Any = None) -> None:
    file = file or sys.stdout
    c = Counter(r.outcome for r in rows)
    changed = [r for r in rows if r.outcome == CHANGED]
    other = {k: v for k, v in c.items() if k not in (UNCHANGED, CHANGED)}
    print(f"\n{task}: checked {len(rows)}, unchanged {c.get(UNCHANGED, 0)}, "
          f"changed {c.get(CHANGED, 0)}"
          + ("".join(f", {k} {v}" for k, v in sorted(other.items())) if other else ""),
          file=file)
    trans = Counter((r.old_status, r.new_status) for r in changed)
    for (a, b), n in sorted(trans.items()):
        print(f"  {a} -> {b}: {n}", file=file)
    by_arch: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for r in changed:
        by_arch[r.architecture][(r.old_status, r.new_status)] += 1
    if len(by_arch) > 1:
        for arch, t in sorted(by_arch.items()):
            print(f"    {arch}: " + ", ".join(f"{a} -> {b}: {n}" for (a, b), n in sorted(t.items())),
                  file=file)
    flips = sum(r.old_hidden_passed == r.old_hidden_total and r.old_hidden_total > 0
                and r.new_hidden_passed < r.new_hidden_total for r in changed)
    if flips:
        print(f"  WARNING: {flips} changed run(s) passed every hidden test before and not now",
              file=file)


def main(argv: list[str] | None = None, evaluator: TempDirEvaluator | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--task", action="append", default=None, help="only this task (repeatable)")
    args = ap.parse_args(argv)
    _check_output_path(args.out_dir / "x.csv", _REPO_ROOT)

    logs = load_all_runs(args.runs_dir, args.task)
    if not logs:
        print(f"no run logs in {args.runs_dir}", file=sys.stderr)
        return 1
    evaluator = evaluator or TempDirEvaluator()

    by_task: dict[str, list[RescoreRow]] = defaultdict(list)
    for i, (path, log) in enumerate(logs, 1):
        print(f"[{i}/{len(logs)}] {path.name}", file=sys.stderr)
        by_task[log["task_name"]].append(rescore_run(path, log, evaluator))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(RescoreRow.__dataclass_fields__)
    print(f"Re-scored {len(logs)} runs with the fixed code-block parser")
    for task in sorted(by_task):
        rows = by_task[task]
        out = args.out_dir / f"rescore_{task}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if v is None else v) for k, v in asdict(r).items()})
        print_task_summary(task, rows)
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

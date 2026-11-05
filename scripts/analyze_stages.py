#!/usr/bin/env python3
"""Stage analysis for the sequential pipeline: does review+fix help or hurt?

Usage:
  python3 scripts/analyze_stages.py <task_name> [--runs-dir runs] [--csv out.csv]

Run from the thesis-agents repo root. Makes no API calls and writes nothing
to eval/ or runs/ (the CSV path is refused if it points inside either).

For every sequential run of <task_name> in runs/:

  1. Load its .artifacts.json sidecar and parse the implement step's response
     with the same parser the pipeline used (architectures.utils.parse_code_blocks):
     the code as it was before review and fix.
  2. Score those files against the hidden tests exactly as the executor does:
     copy agent-bench/tasks/<task> to a fresh temp dir, write the files with
     Executor._apply_modified_files, run eval/run_eval.sh via
     Executor._run_hidden_eval.
  3. Record implement-step hidden pass/total next to the logged final
     pass/total, and whether the review response reported issues.

As a drift check, the run's final files (artifacts.modified_files, i.e. what
the executor applied) are re-scored the same way; final_rescore_matches_log
is False when today's hidden tests or task files score them differently than
the log recorded, which would make implement-vs-final comparisons unsound.

The implement step is always parsed with the current parse_code_blocks
(fixed on 2026-11-05 to close blocks only at a line of backticks). Final
results come from the run log, which used the parser of the time; pass
--rescored to replace them, for runs whose result changed, with
analysis/rescore_<task>.csv (scripts/rescore_runs.py), and to take the final
files from the responses re-parsed with the fixed parser. The header says
whether re-scored results are in use and how many runs were affected.

"Pass" everywhere means every hidden test passed (hidden_total > 0 and
hidden_passed == hidden_total). This is computed from test counts only, so it
ignores run status and policy compliance (unlike aggregate_results.py's strict
pass@1); none of the current runs differ on this, but check before quoting.

"Review reported issues" is a keyword heuristic over free text (the review
prompt asks for no fixed verdict format): "no" if the review explicitly says
the code is correct / no changes are needed, "yes" if it names bugs or issues,
"unclear" otherwise. The matched phrase is in the review_evidence column so
each call can be audited by hand.

Outputs:
  - CSV, one row per sequential run (default: ./stage_analysis_<task>.csv)
  - stdout: summary per (model, reasoning effort, regime) cell with transition
    counts, mean implement vs final hidden pass rate (paired, over runs with
    artifacts), and the single-agent pass rate for the same cell
  - stdout: failing hidden test names per run, for every architecture
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architectures.utils import parse_code_blocks  # noqa: E402
from runner.config import TaskPaths  # noqa: E402
from runner.executor import Executor  # noqa: E402

RUNS_DIR = Path("runs")
SEQUENTIAL = "sequential"
SINGLE_AGENT = "single_agent"
_REQUIRED_KEYS = ("run_id", "task_name", "model", "architecture_name", "status", "test_results")

# Scorer signature: (task_name, files) -> (passed, total, failing_hidden_names)
Scorer = Callable[[str, dict[str, str]], tuple[int, int, list[str]]]
CellKey = tuple[str, str, str]  # (model, reasoning_effort, prompt_regime)

# ---------------------------------------------------------------------------
# Review classification
# ---------------------------------------------------------------------------

_NO_ISSUE_PATTERNS = [
    r"no definite (?:correctness )?(?:issues|bugs|defects)",
    r"no (?:changes|modifications) (?:are )?(?:required|needed|necessary)",
    r"no required changes",
    r"no issues (?:were )?found",
    r"found no (?:issues|bugs|problems)",
    r"(?:code|implementation) (?:is|appears|looks) (?:essentially |fully |entirely )?correct\b(?![,;]? *(?:but|except|apart))",
]
_ISSUE_PATTERNS = [
    r"issues? found",
    r"\bbugs?\b",
    r"not (?:fully |entirely )?correct",
    r"incorrect",
    r"spec(?:ification)? violation",
    r"violates",
    r"\bissues?\b",
]


def classify_review(text: str | None) -> tuple[str, str]:
    """Return ("yes"|"no"|"unclear", evidence) for a review response.

    An explicit no-issues statement wins over issue keywords, because clean
    reviews still say things like "no bugs" or list "minor issues" that are
    judgement calls. Empty/missing text is "unclear".
    """
    if not text:
        return "unclear", ""
    for pat in _NO_ISSUE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return "no", m.group(0)
    for pat in _ISSUE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return "yes", m.group(0)
    return "unclear", ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _is_run_log_name(path: Path) -> bool:
    """Mirror of runner.logger.is_run_log_file."""
    name = path.name
    return (
        name.endswith(".json")
        and not name.endswith(".artifacts.json")
        and not name.startswith("batch_")
    )


def load_runs(runs_dir: Path, task: str) -> list[tuple[Path, dict[str, Any]]]:
    """Load every run log for *task* in *runs_dir*, sorted by filename."""
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(runs_dir.glob("*.json")):
        if not _is_run_log_name(path):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not all(k in data for k in _REQUIRED_KEYS):
            continue
        if data["task_name"] == task:
            out.append((path, data))
    return out


def _step_response(artifacts: dict[str, Any], role: str) -> str | None:
    """Response text of the pipeline step with *role* from the agent_log."""
    for entry in artifacts.get("agent_log") or []:
        if entry.get("role") == role:
            return entry.get("response")
    return None


def is_pass(passed: int | None, total: int | None) -> bool | None:
    """All hidden tests passed; None when there is no score."""
    if passed is None or total is None:
        return None
    return total > 0 and passed == total


def hidden_failing(names: list[str]) -> list[str]:
    """Hidden-test entries of a log's failing_test_names (drops public tests)."""
    return [n for n in names if n.startswith("eval/") or n.startswith("hidden_eval_")
            or n == "eval_timeout" or n == "eval_script_not_found"]


def _short(name: str) -> str:
    """Test id without its file path, for compact printing."""
    return name.split("::", 1)[1] if "::" in name else name


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def make_executor_scorer(repo_root: Path = _REPO_ROOT) -> Scorer:
    """Hidden-test scorer that reproduces the executor's steps 2, 5 and 7."""
    executor = Executor(repo_root=repo_root)

    def score(task: str, files: dict[str, str]) -> tuple[int, int, list[str]]:
        paths = TaskPaths(task, repo_root)
        tmp = tempfile.mkdtemp(prefix=f"stage_{task}_")
        try:
            temp_task_dir = Path(tmp) / "task"
            shutil.copytree(paths.task_dir, temp_task_dir)
            executor._apply_modified_files(files, temp_task_dir)
            return executor._run_hidden_eval(task, temp_task_dir)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return score


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------


@dataclass
class StageRow:
    """One sequential run: implement-step score next to the final score."""

    run_id: str
    log_file: str
    task: str
    provider: str
    model: str
    reasoning_effort: str
    regime: str
    status: str
    has_artifacts: bool
    implement_files: str = ""
    implement_hidden_passed: int | None = None
    implement_hidden_total: int | None = None
    implement_pass: bool | None = None
    final_hidden_passed: int = 0
    final_hidden_total: int = 0
    final_pass: bool = False
    transition: str = ""
    review_reported_issues: str = ""
    review_evidence: str = ""
    fix_changed_code: bool | None = None
    final_rescored_passed: int | None = None
    final_rescored_total: int | None = None
    final_rescore_matches_log: bool | None = None
    note: str = ""
    implement_failing: list[str] = field(default_factory=list)
    final_failing: list[str] = field(default_factory=list)

    def csv_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["implement_failing"] = ";".join(_short(n) for n in self.implement_failing)
        d["final_failing"] = ";".join(_short(n) for n in self.final_failing)
        return {k: ("" if v is None else v) for k, v in d.items()}


def _pf(p: bool | None) -> str:
    return "pass" if p else "fail"


def analyze_run(
    log_path: Path, log: dict[str, Any], scorer: Scorer, rescore_final: bool = True,
    final_files_override: dict[str, str] | None = None,
) -> StageRow:
    """Build the StageRow for one sequential run log.

    final_files_override replaces artifacts.modified_files as the final code
    (used with --rescored for runs whose files changed under the fixed parser).
    """
    tr = log["test_results"]
    row = StageRow(
        run_id=log["run_id"],
        log_file=log_path.name,
        task=log["task_name"],
        provider=log.get("provider") or "",
        model=log["model"],
        reasoning_effort=log.get("requested_reasoning_effort") or "",
        regime=log.get("prompt_regime") or "",
        status=log["status"],
        has_artifacts=False,
        final_hidden_passed=tr.get("hidden_passed", 0),
        final_hidden_total=tr.get("hidden_total", 0),
        final_failing=hidden_failing(tr.get("failing_test_names") or []),
    )
    row.final_pass = bool(is_pass(row.final_hidden_passed, row.final_hidden_total))

    art_name = log.get("artifacts_file")
    art_path = log_path.parent / art_name if art_name else None
    if art_path is None or not art_path.is_file():
        row.note = "no artifacts sidecar"
        return row
    try:
        artifacts = json.loads(art_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        row.note = f"unreadable artifacts: {e}"
        return row
    row.has_artifacts = True

    review_text = _step_response(artifacts, "review")
    if review_text is not None:
        row.review_reported_issues, row.review_evidence = classify_review(review_text)

    implement_text = _step_response(artifacts, "implement")
    if implement_text is None:
        row.note = "no implement step response in artifacts"
        return row

    implement_files = parse_code_blocks(implement_text)
    row.implement_files = ";".join(sorted(implement_files))
    p, t, failing = scorer(row.task, implement_files)
    row.implement_hidden_passed, row.implement_hidden_total = p, t
    row.implement_failing = failing
    row.implement_pass = is_pass(p, t)
    row.transition = f"{_pf(row.implement_pass)}->{_pf(row.final_pass)}"
    if not implement_files:
        row.note = "implement step produced no labeled code blocks"

    final_files = (final_files_override if final_files_override is not None
                   else artifacts.get("modified_files") or {})
    row.fix_changed_code = final_files != implement_files
    if rescore_final:
        fp, ft, _ = scorer(row.task, final_files)
        row.final_rescored_passed, row.final_rescored_total = fp, ft
        row.final_rescore_matches_log = (
            fp == row.final_hidden_passed and ft == row.final_hidden_total
        )
    return row


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

TRANSITIONS = ("pass->pass", "fail->pass", "pass->fail", "fail->fail")


def _cell(log_or_row: dict[str, Any] | StageRow) -> CellKey:
    if isinstance(log_or_row, StageRow):
        return (log_or_row.model, log_or_row.reasoning_effort, log_or_row.regime)
    return (
        log_or_row["model"],
        log_or_row.get("requested_reasoning_effort") or "",
        log_or_row.get("prompt_regime") or "",
    )


def _rate(passed: int, total: int) -> float:
    return passed / total if total else 0.0


def summarize(
    rows: list[StageRow], single_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One summary dict per cell that has sequential runs."""
    by_cell: dict[CellKey, list[StageRow]] = defaultdict(list)
    for r in rows:
        by_cell[_cell(r)].append(r)
    singles: dict[CellKey, list[dict[str, Any]]] = defaultdict(list)
    for log in single_logs:
        singles[_cell(log)].append(log)

    out: list[dict[str, Any]] = []
    for key in sorted(by_cell):
        cell_rows = by_cell[key]
        scored = [r for r in cell_rows if r.implement_hidden_total is not None]
        counts = {t: sum(r.transition == t for r in scored) for t in TRANSITIONS}
        sa = singles.get(key, [])
        sa_pass = [
            bool(is_pass(l["test_results"].get("hidden_passed"), l["test_results"].get("hidden_total")))
            for l in sa
        ]
        out.append({
            "model": key[0],
            "effort": key[1],
            "regime": key[2],
            "n_seq": len(cell_rows),
            "n_scored": len(scored),
            **counts,
            "impl_rate": (statistics.mean(
                _rate(r.implement_hidden_passed, r.implement_hidden_total) for r in scored
            ) if scored else None),
            "final_rate": (statistics.mean(
                _rate(r.final_hidden_passed, r.final_hidden_total) for r in scored
            ) if scored else None),
            "seq_pass": sum(r.final_pass for r in cell_rows) / len(cell_rows),
            "n_single": len(sa),
            "single_pass": (sum(sa_pass) / len(sa)) if sa else None,
            "review_issues": sum(r.review_reported_issues == "yes" for r in scored),
            "rescore_mismatch": sum(r.final_rescore_matches_log is False for r in scored),
        })
    return out


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


def print_summary(summary: list[dict[str, Any]], file: Any = None) -> None:
    """Print the per-cell summary table."""
    file = file or sys.stdout
    hdr = (f"{'model':<26} {'effort':<6} {'regime':<10} {'n':>2} {'scr':>3}  "
           f"{'P->P':>4} {'F->P':>4} {'P->F':>4} {'F->F':>4}  "
           f"{'impl':>5} {'final':>5}  {'seq@1':>5} {'single@1':>8} {'(n)':>4}  "
           f"{'rev+':>4} {'drift':>5}")
    print(hdr, file=file)
    print("-" * len(hdr), file=file)
    for s in summary:
        print(
            f"{s['model']:<26} {s['effort'] or '-':<6} {s['regime']:<10} "
            f"{s['n_seq']:>2} {s['n_scored']:>3}  "
            f"{s['pass->pass']:>4} {s['fail->pass']:>4} {s['pass->fail']:>4} {s['fail->fail']:>4}  "
            f"{_fmt(s['impl_rate']):>5} {_fmt(s['final_rate']):>5}  "
            f"{_fmt(s['seq_pass']):>5} {_fmt(s['single_pass']):>8} {s['n_single']:>4}  "
            f"{s['review_issues']:>4} {s['rescore_mismatch']:>5}",
            file=file,
        )
    print(
        "\n  n/scr = sequential runs / runs with artifacts (transitions and impl/final "
        "means use scr only, paired)\n"
        "  impl/final = mean hidden pass rate of the implement step / final output\n"
        "  seq@1, single@1 = share of runs with all hidden tests passing "
        "(seq over all n; (n) = single-agent runs)\n"
        "  rev+ = reviews classified as reporting issues; drift = final re-score "
        "differs from the logged score",
        file=file,
    )


def print_failing(
    rows: list[StageRow], logs: list[tuple[Path, dict[str, Any]]], file: Any = None,
) -> None:
    """Print failing hidden test names per run, every architecture."""
    file = file or sys.stdout
    by_id = {r.run_id: r for r in rows}
    print("\nFailing hidden tests per run", file=file)
    print("=" * 28, file=file)
    for path, log in logs:
        tr = log["test_results"]
        failing = hidden_failing(tr.get("failing_test_names") or [])
        print(
            f"{log['architecture_name']:<13} {log['model']:<26} "
            f"{log.get('prompt_regime') or '':<10} {log['run_id'][:8]}  "
            f"final {tr.get('hidden_passed', 0)}/{tr.get('hidden_total', 0)}: "
            f"{', '.join(_short(n) for n in failing) or '-'}",
            file=file,
        )
        r = by_id.get(log["run_id"])
        if r is not None and r.implement_hidden_total is not None:
            print(
                f"{'':<13} {'':<26} {'':<10} {'':<8}  "
                f"impl  {r.implement_hidden_passed}/{r.implement_hidden_total}: "
                f"{', '.join(_short(n) for n in r.implement_failing) or '-'}",
                file=file,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _rebuilt_final(log_path: Path, log: dict[str, Any]) -> dict[str, str] | None:
    """Final files re-parsed with the current parser (see rescore_runs.py)."""
    from rescore_runs import rebuild_final_files
    art_name = log.get("artifacts_file")
    if not art_name or not (log_path.parent / art_name).is_file():
        return None
    try:
        artifacts = json.loads((log_path.parent / art_name).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    files, _ = rebuild_final_files(log["architecture_name"], artifacts)
    return files


def _check_output_path(path: Path, repo_root: Path) -> None:
    """Refuse to write inside eval/ or runs/."""
    resolved = path.resolve()
    for forbidden in (repo_root / "eval", repo_root / "runs"):
        if resolved.is_relative_to(forbidden.resolve()):
            raise SystemExit(f"refusing to write {path}: inside {forbidden.name}/")


def main(argv: list[str] | None = None, scorer: Scorer | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("task", help="task name, e.g. diffapply")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--csv", type=Path, default=None,
                    help="output CSV (default: ./stage_analysis_<task>.csv)")
    ap.add_argument("--no-rescore-final", action="store_true",
                    help="skip the drift check (re-scoring final files)")
    ap.add_argument("--rescored", action="store_true",
                    help="use re-scored results from analysis/rescore_<task>.csv")
    ap.add_argument("--analysis-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    csv_path = args.csv or Path(f"stage_analysis_{args.task}.csv")
    _check_output_path(csv_path, _REPO_ROOT)

    logs = load_runs(args.runs_dir, args.task)
    if not logs:
        print(f"no run logs for task {args.task!r} in {args.runs_dir}", file=sys.stderr)
        return 1
    rescore = None
    if args.rescored:
        from rescored import ANALYSIS_DIR, require_rescore
        rescore = require_rescore([args.task], args.analysis_dir or ANALYSIS_DIR)
        header = rescore.header([l for _, l in logs])
        logs = [(p, rescore.apply(l)) for p, l in logs]
    seq_logs = [(p, l) for p, l in logs if l["architecture_name"] == SEQUENTIAL]
    single_logs = [l for _, l in logs if l["architecture_name"] == SINGLE_AGENT]

    score = scorer or make_executor_scorer()
    rows: list[StageRow] = []
    for i, (path, log) in enumerate(seq_logs, 1):
        print(f"[{i}/{len(seq_logs)}] scoring {path.name}", file=sys.stderr)
        override = _rebuilt_final(path, log) if rescore and rescore.changed(log["run_id"]) else None
        rows.append(analyze_run(path, log, score, rescore_final=not args.no_rescore_final,
                                final_files_override=override))

    fieldnames = list(rows[0].csv_dict()) if rows else list(StageRow.__dataclass_fields__)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.csv_dict())

    print(header if rescore else "Results as logged (no --rescored); implement step parsed "
          "with the current parser")
    print(f"\nTask {args.task}: {len(seq_logs)} sequential, {len(single_logs)} single-agent runs\n")
    if rows:
        print_summary(summarize(rows, single_logs))
    print_failing(rows, logs)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

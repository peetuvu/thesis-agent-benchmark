"""Load re-scored results (analysis/rescore_<task>.csv) and apply them to run logs.

Standard library only, so scripts that avoid harness imports
(aggregate_results.py) can use it. The CSVs are written by
scripts/rescore_runs.py; see its docstring for what re-scoring means.

A run's log is overridden only when its CSV row has outcome "changed": then
status, hidden_passed, hidden_total and the hidden part of
failing_test_names are replaced by the re-scored values (public test
results are kept as logged). Every other outcome leaves the log as it was.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
CHANGED = "changed"


def rescore_csv_path(task: str, analysis_dir: Path = ANALYSIS_DIR) -> Path:
    """Where rescore_runs.py writes the CSV for *task*."""
    return analysis_dir / f"rescore_{task}.csv"


def _is_hidden(name: str) -> bool:
    """Mirror of analyze_stages.hidden_failing's test (kept local)."""
    return (name.startswith("eval/") or name.startswith("hidden_eval_")
            or name in ("eval_timeout", "eval_script_not_found"))


@dataclass
class Rescore:
    """Re-scored rows for a set of tasks, keyed by run_id."""

    rows: dict[str, dict[str, str]] = field(default_factory=dict)
    sources: dict[str, Path] = field(default_factory=dict)  # task -> csv
    missing: list[str] = field(default_factory=list)  # tasks without a csv

    def changed(self, run_id: str) -> dict[str, str] | None:
        """The CSV row if this run's result changed, else None."""
        row = self.rows.get(run_id)
        return row if row is not None and row.get("outcome") == CHANGED else None

    def apply(self, log: dict[str, Any]) -> dict[str, Any]:
        """A copy of *log* with re-scored results, or *log* itself if unchanged."""
        row = self.changed(log["run_id"])
        if row is None:
            return log
        new = copy.deepcopy(log)
        tr = new.setdefault("test_results", {})
        public_failing = [n for n in tr.get("failing_test_names") or [] if not _is_hidden(n)]
        hidden_failing = [n for n in (row.get("new_hidden_failing") or "").split(";") if n]
        tr["hidden_passed"] = int(row["new_hidden_passed"])
        tr["hidden_total"] = int(row["new_hidden_total"])
        tr["failing_test_names"] = hidden_failing + public_failing
        new["status"] = row["new_status"]
        if row.get("new_policy_compliant") in ("True", "False"):
            new["policy_compliant"] = row["new_policy_compliant"] == "True"
        new["rescored"] = True
        return new

    def header(self, logs: list[dict[str, Any]]) -> str:
        """One-line notice that re-scored results are in use."""
        ids = {l["run_id"] for l in logs}
        affected = sum(self.changed(i) is not None for i in ids)
        srcs = ", ".join(str(p) for _, p in sorted(self.sources.items())) or "none"
        return (f"RE-SCORED RESULTS IN USE (fixed code-block parser): {affected} of "
                f"{len(ids)} runs affected; source: {srcs}")


def load_rescore(tasks: list[str] | set[str], analysis_dir: Path = ANALYSIS_DIR) -> Rescore:
    """Load analysis/rescore_<task>.csv for each task; record missing ones."""
    res = Rescore()
    for task in sorted(set(tasks)):
        path = rescore_csv_path(task, analysis_dir)
        if not path.is_file():
            res.missing.append(task)
            continue
        res.sources[task] = path
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                res.rows[row["run_id"]] = row
    return res


def require_rescore(tasks: list[str] | set[str], analysis_dir: Path = ANALYSIS_DIR) -> Rescore:
    """load_rescore, but exit with a message if any task's CSV is missing."""
    res = load_rescore(tasks, analysis_dir)
    if res.missing:
        raise SystemExit(
            "--rescored: no re-score CSV for task(s) "
            + ", ".join(res.missing)
            + f" in {analysis_dir}; run scripts/rescore_runs.py first"
        )
    return res

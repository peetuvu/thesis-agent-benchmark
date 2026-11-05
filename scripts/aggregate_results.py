#!/usr/bin/env python3
"""Aggregate benchmark run logs into per-configuration summary statistics.

Usage:
  python3 scripts/aggregate_results.py [task_name] [--runs-dir runs] [--csv out.csv]

Run from the thesis-agents repo root (so it can find runs/).

One row per (task, provider, model, architecture, prompt regime, reasoning
effort). Every run log is counted; nothing is dropped:

  n            runs in the group
  err / tmo    runs with status "error" / "timeout"
  noncompl     runs that violated the sandbox policy (policy_compliant False
               or status "policy_violation"); they count as not-passed
  pass@1       strict: share of runs with status "success" that are also
               policy-compliant (every hidden test passed, no violation)
  hidden       mean ± sample std of the per-run hidden-test pass rate
               (hidden_passed / hidden_total; 0/0 counts as 0)
  tokens, latency
               means over all runs (zeros recorded by pre-2026-11-04 logs for
               timeouts/exceptions are included as recorded)
  cost         mean estimated USD over priced runs; "-" if none is priced,
               "*" suffix if only some are

--rescored: use analysis/rescore_<task>.csv (written by
scripts/rescore_runs.py) for runs whose result changed under the fixed
code-block parser: their status and hidden-test results are replaced by the
re-scored values. The header says so and how many runs were affected.

Files in runs/ that are not run logs (batch_*.json summaries, *.artifacts.json
sidecars) are skipped by name; any other JSON without a run log's fields is
reported as skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescored import ANALYSIS_DIR, require_rescore  # noqa: E402  (stdlib only)

RUNS_DIR = Path("runs")
_REQUIRED_KEYS = ("run_id", "task_name", "model", "architecture_name", "status", "test_results")

GroupKey = tuple[str, str, str, str, str, str]


def _is_run_log_name(path: Path) -> bool:
    """Mirror of runner.logger.is_run_log_file (kept local: no harness import)."""
    name = path.name
    return (
        name.endswith(".json")
        and not name.endswith(".artifacts.json")
        and not name.startswith("batch_")
    )


def load_runs(
    runs_dir: Path = RUNS_DIR, task_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load every run log in *runs_dir* (optionally one task).

    Returns (runs, skipped) where skipped lists JSON files that are not run
    logs although their name suggests they are.
    """
    runs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in sorted(runs_dir.glob("*.json")):
        if not _is_run_log_name(path):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            skipped.append(f"{path.name} ({e})")
            continue
        if not isinstance(data, dict) or not all(k in data for k in _REQUIRED_KEYS):
            skipped.append(f"{path.name} (not a run log)")
            continue
        if task_filter and data["task_name"] != task_filter:
            continue
        runs.append(data)
    return runs, skipped


def group_key(run: dict[str, Any]) -> GroupKey:
    """(task, provider, model, architecture, regime, reasoning effort)."""
    return (
        run["task_name"],
        run.get("provider") or "-",
        run["model"],
        run["architecture_name"],
        run.get("prompt_regime") or "-",
        run.get("requested_reasoning_effort") or "-",
    )


def _hidden_rate(run: dict[str, Any]) -> float:
    tr = run["test_results"]
    total = tr.get("hidden_total", 0)
    return tr.get("hidden_passed", 0) / total if total else 0.0


def _non_compliant(run: dict[str, Any]) -> bool:
    return run.get("policy_compliant") is False or run["status"] == "policy_violation"


def _strict_pass(run: dict[str, Any]) -> bool:
    """Status success AND policy-compliant (older logs could record a
    non-compliant run as success, before policy_violation existed)."""
    return run["status"] == "success" and not _non_compliant(run)


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per configuration group, sorted by group key."""
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)

    rows = []
    for key, group in sorted(groups.items()):
        task, provider, model, arch, regime, effort = key
        n = len(group)
        rates = [_hidden_rate(g) for g in group]
        metrics = [g.get("actual_metrics", {}) for g in group]
        priced = [m["estimated_cost_usd"] for m in metrics if m.get("price_source")]
        rows.append({
            "task": task,
            "provider": provider,
            "model": model,
            "arch": arch,
            "regime": regime,
            "effort": effort,
            "n": n,
            "errors": sum(g["status"] == "error" for g in group),
            "timeouts": sum(g["status"] == "timeout" for g in group),
            "non_compliant": sum(_non_compliant(g) for g in group),
            "pass_at_1": sum(_strict_pass(g) for g in group) / n,
            "hidden_mean": statistics.mean(rates),
            "hidden_std": statistics.stdev(rates) if n > 1 else 0.0,
            "tokens_mean": statistics.mean(m.get("tokens_total", 0) for m in metrics),
            "latency_mean": statistics.mean(m.get("wall_clock_seconds", 0.0) for m in metrics),
            "cost_mean": statistics.mean(priced) if priced else None,
            "cost_priced_runs": len(priced),
        })
    return rows


def _fmt_cost(row: dict[str, Any]) -> str:
    if row["cost_mean"] is None:
        return "-"
    partial = "*" if row["cost_priced_runs"] < row["n"] else ""
    return f"${row['cost_mean']:.4f}{partial}"


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'Task':<11} {'Provider':<9} {'Model':<26} {'Arch':<13} {'Regime':<11} {'Effort':<6} "
        f"{'n':>3} {'err':>3} {'tmo':>3} {'ncmp':>4} {'pass@1':>7} "
        f"{'Hidden% ± std':>15} {'Tokens':>8} {'Lat(s)':>7} {'Cost':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['task']:<11} {r['provider']:<9} {r['model']:<26} {r['arch']:<13} "
            f"{r['regime']:<11} {r['effort']:<6} "
            f"{r['n']:>3} {r['errors']:>3} {r['timeouts']:>3} {r['non_compliant']:>4} "
            f"{r['pass_at_1']*100:>6.1f}% "
            f"{r['hidden_mean']*100:>7.1f} ± {r['hidden_std']*100:>4.1f} "
            f"{r['tokens_mean']:>8.0f} {r['latency_mean']:>7.1f} {_fmt_cost(r):>10}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task", nargs="?", default=None, help="Only this task")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--csv", type=Path, default=None, help="Also write the rows as CSV")
    parser.add_argument("--rescored", action="store_true",
                        help="use re-scored results from analysis/rescore_<task>.csv")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    runs, skipped = load_runs(args.runs_dir, args.task)
    for s in skipped:
        print(f"Skipped: {s}", file=sys.stderr)
    if not runs:
        print("No runs found" + (f" for task={args.task}" if args.task else ""))
        return 1

    if args.rescored:
        rescore = require_rescore({r["task_name"] for r in runs}, args.analysis_dir)
        print(rescore.header(runs))
        runs = [rescore.apply(r) for r in runs]

    rows = summarize(runs)
    totals = {k: sum(r[k] for r in rows) for k in ("errors", "timeouts", "non_compliant")}
    print(
        f"Loaded {len(runs)} runs" + (f" for task={args.task}" if args.task else "")
        + f" in {len(rows)} configurations (all included: {totals['errors']} error, "
        f"{totals['timeouts']} timeout, {totals['non_compliant']} non-compliant)\n"
    )
    print_table(rows)
    print(
        "\npass@1 = share of runs with status 'success' that are policy-compliant "
        "(all hidden tests passed, no violation). Hidden% = mean per-run hidden pass rate. "
        "Cost '*' = some runs unpriced."
    )
    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

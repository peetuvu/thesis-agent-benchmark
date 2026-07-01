#!/usr/bin/env python3
"""Run benchmark batches — replaces the manual bash loop.

Builds the full cartesian product of (task, model, arch, regime), repeats each
combination `--reps` times, and invokes `bench run` sequentially for each.
Model names are passed straight through to bench, which owns alias resolution
via providers/models.py. Errors in one run do not abort the batch; a summary
JSON is written to runs/.

Usage:
  python scripts/run_batch.py \
    --tasks diffapply calculator \
    --models haiku-4.5 sonnet-4.6 opus-4.6 \
    --archs single_agent sequential \
    --regimes minimal production \
    --reps 3 \
    --dry-run

Run from the thesis-agents repo root.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("runs")


def parse_output(stdout: str) -> tuple[str, str]:
    """Extract (status, hidden_score) from bench run output.

    Returns status like 'SUCCESS' and hidden score like '13/15', using
    'unknown'/'?/?' as fallbacks when the lines are absent.
    """
    status = "unknown"
    hidden = "?/?"
    status_match = re.search(r"^Status:\s*(\S+)", stdout, re.MULTILINE)
    if status_match:
        status = status_match.group(1).upper()
    hidden_match = re.search(r"^Hidden:\s*(\d+/\d+)", stdout, re.MULTILINE)
    if hidden_match:
        hidden = hidden_match.group(1)
    return status, hidden


def build_combinations(args: argparse.Namespace) -> list[dict]:
    """Build the flat list of run specs (each repeated --reps times)."""
    runs = []
    for task in args.tasks:
        for model in args.models:
            for arch in args.archs:
                for regime in args.regimes:
                    for rep in range(1, args.reps + 1):
                        runs.append(
                            {
                                "task": task,
                                "model": model,
                                "arch": arch,
                                "regime": regime,
                                "rep": rep,
                            }
                        )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_batch",
        description="Run a batch of benchmark runs via the bench CLI.",
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="One or more task names")
    parser.add_argument("--models", nargs="+", required=True, help="One or more model shortnames")
    parser.add_argument("--archs", nargs="+", default=["single_agent"], help="Architecture names")
    parser.add_argument(
        "--regimes", nargs="+", default=["minimal", "production"], help="Prompt regimes"
    )
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per combination")
    parser.add_argument("--provider", default="anthropic", help="Provider name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned runs without executing"
    )
    args = parser.parse_args()

    runs = build_combinations(args)
    n_combos = len(runs) // args.reps if args.reps else 0
    print(f"Planning {len(runs)} runs across {n_combos} combinations")

    if args.dry_run:
        for i, r in enumerate(runs, 1):
            print(
                f"  [{i}/{len(runs)}] {r['task']} | {r['model']} | "
                f"{r['arch']} | {r['regime']} | run {r['rep']}/{args.reps}"
            )
        return 0

    RUNS_DIR.mkdir(exist_ok=True)

    outcomes = []
    errors = []
    total = len(runs)

    for i, r in enumerate(runs, 1):
        model_label = r["model"]
        cmd = [
            "bench",
            "run",
            "--task",
            r["task"],
            "--arch",
            r["arch"],
            "--model",
            r["model"],
            "--prompt-regime",
            r["regime"],
            "--provider",
            args.provider,
        ]

        outcome = {
            **r,
            "command": " ".join(cmd),
            "status": "error",
            "hidden": None,
            "error": None,
        }

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as exc:  # noqa: BLE001 — never abort the batch
            outcome["error"] = f"connection error: {exc}"
            errors.append(outcome)
            outcomes.append(outcome)
            print(
                f"[{i}/{total}] {r['task']} | {model_label} | {r['regime']} | "
                f"run {r['rep']}/{args.reps} — WARNING: {outcome['error']}"
            )
            continue

        status, hidden = parse_output(proc.stdout)
        outcome["status"] = status
        outcome["hidden"] = hidden

        if proc.returncode != 0 and status in ("unknown", "ERROR"):
            # Non-zero exit with no parseable status = hard failure (e.g. crash).
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = detail[-1] if detail else f"exit code {proc.returncode}"
            outcome["error"] = reason
            errors.append(outcome)
            print(
                f"[{i}/{total}] {r['task']} | {model_label} | {r['regime']} | "
                f"run {r['rep']}/{args.reps} — WARNING: {reason}"
            )
        else:
            if proc.returncode != 0:
                # bench exits non-zero for non-success runs (failure/timeout);
                # these are valid outcomes, not batch errors.
                errors.append(outcome)
            print(
                f"[{i}/{total}] {r['task']} | {model_label} | {r['regime']} | "
                f"run {r['rep']}/{args.reps} — Hidden: {hidden} ({status})"
            )

        outcomes.append(outcome)

    succeeded = sum(1 for o in outcomes if o["status"] == "SUCCESS")
    n_errors = len(outcomes) - succeeded

    print()
    print(f"Batch complete: {succeeded}/{total} succeeded, {n_errors} errors")
    if n_errors:
        print("Errors:")
        for o in outcomes:
            if o["status"] == "SUCCESS":
                continue
            reason = o["error"] or f"status {o['status']}, hidden {o['hidden']}"
            print(
                f"  - {o['task']} | {o['model']} | {o['regime']} | "
                f"rep {o['rep']}: {reason}"
            )

    timestamp = datetime.now().strftime("%y-%m-%d_%H%M%S")
    summary_path = RUNS_DIR / f"batch_{timestamp}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "succeeded": succeeded,
        "errors": n_errors,
        "provider": args.provider,
        "reps": args.reps,
        "outcomes": outcomes,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

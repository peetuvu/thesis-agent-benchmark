#!/usr/bin/env python3
"""Aggregate benchmark run logs into per-(arch, model, regime) summary stats.
Usage: python3 scripts/aggregate_results.py [task_name]
Run from the thesis-agents repo root (so it can find runs/).
"""
import json
import sys
import statistics
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path("runs")


def shorten_model(model: str) -> str:
    return (model
            .replace("claude-", "")
            .replace("-20251001", "")
            .replace("-20250514", ""))


def load_runs(task_filter=None):
    runs = []
    for path in RUNS_DIR.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        if task_filter and data.get("task_name") != task_filter:
            continue
        if data.get("status") == "error":
            continue
        runs.append(data)
    return runs


def summarize(runs):
    groups = defaultdict(list)
    for r in runs:
        key = (r["architecture_name"], shorten_model(r["model"]), r["prompt_regime"])
        groups[key].append(r)

    rows = []
    for (arch, model, regime), group in sorted(groups.items()):
        hidden_rates = [
            g["test_results"]["hidden_passed"] / g["test_results"]["hidden_total"]
            for g in group
        ]
        public_rates = [
            g["test_results"]["public_passed"] / g["test_results"]["public_total"]
            for g in group
        ]
        tokens = [g["actual_metrics"]["tokens_total"] for g in group]
        latency = [g["actual_metrics"]["wall_clock_seconds"] for g in group]

        rows.append({
            "arch": arch,
            "model": model,
            "regime": regime,
            "n": len(group),
            "hidden_mean": statistics.mean(hidden_rates),
            "hidden_std": statistics.stdev(hidden_rates) if len(hidden_rates) > 1 else 0.0,
            "public_mean": statistics.mean(public_rates),
            "tokens_mean": statistics.mean(tokens),
            "latency_mean": statistics.mean(latency),
        })
    return rows


def print_table(rows):
    header = (
        f"\n{'Arch':<16}  {'Model':<14}  {'Regime':<12}  "
        f"{'n':>3}  {'Hidden%':>9}  {'+/-':>6}  "
        f"{'Public%':>9}  {'Tokens':>8}  {'Lat(s)':>8}"
    )
    print(header)
    print("-" * 100)
    for row in rows:
        print(
            f"{row['arch']:<16}  {row['model']:<14}  {row['regime']:<12}  "
            f"{row['n']:>3}  "
            f"{row['hidden_mean']*100:>8.1f}%  {row['hidden_std']*100:>5.1f}%  "
            f"{row['public_mean']*100:>8.1f}%  "
            f"{row['tokens_mean']:>8.0f}  {row['latency_mean']:>8.1f}"
        )
    print()


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else None
    runs = load_runs(task_filter=task)
    if not runs:
        print("No runs found" + (f" for task={task}" if task else ""))
        sys.exit(1)
    print(f"Loaded {len(runs)} runs" + (f" for task={task}" if task else ""))
    print_table(summarize(runs))
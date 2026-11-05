"""Tests for scripts/analyze_stages.py using fake run logs and artifacts sidecars."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "analyze_stages.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_stages", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve types via sys.modules
    spec.loader.exec_module(module)
    return module


st = _load()

FILE = "diffapply/applier.py"
HIDDEN = "eval/hidden_tests/diffapply/test_hidden.py::"


def fake_scorer(task: str, files: dict[str, str]) -> tuple[int, int, list[str]]:
    """Content-driven scorer: GOOD -> 3/3, BROKEN -> collection error, else 1/3."""
    src = files.get(FILE, "")
    if "GOOD" in src:
        return 3, 3, []
    if "BROKEN" in src:
        return 0, 0, ["hidden_eval_collection_error"]
    return 1, 3, [HIDDEN + "test_a", HIDDEN + "test_b"]


def _block(content: str) -> str:
    return f"Here is the code:\n\n```{FILE}\n{content}\n```\n"


def _score_for(content: str) -> tuple[int, int, list[str]]:
    return fake_scorer("diffapply", {FILE: content + "\n"})


def write_run(
    runs_dir: Path,
    run_id: str,
    *,
    arch: str = "sequential",
    model: str = "m1",
    regime: str = "minimal",
    implement: str | None = "GOOD",
    fix: str | None = None,
    review: str = "Found a bug in X.",
    artifacts: bool = True,
    final: tuple[int, int, list[str]] | None = None,
) -> Path:
    """Write a fake run log (+ artifacts sidecar) and return the log path.

    The logged final score defaults to what fake_scorer gives the final files,
    so the drift check matches unless *final* overrides it.
    """
    impl_files = {FILE: implement + "\n"} if implement is not None else {}
    fix_files = {FILE: fix + "\n"} if fix is not None else {}
    final_files = {**impl_files, **fix_files}
    if final is None:
        final = fake_scorer("diffapply", final_files)
    passed, total, failing = final
    stem = f"26-11-05_1200_p_{model}_{arch}_diffapply_{regime}_{run_id}"
    log: dict[str, Any] = {
        "run_id": run_id,
        "task_name": "diffapply",
        "architecture_name": arch,
        "prompt_regime": regime,
        "model": model,
        "provider": "p",
        "requested_reasoning_effort": "low",
        "status": "success" if total and passed == total else "failure",
        "test_results": {
            "public_passed": 1, "public_total": 2,
            "hidden_passed": passed, "hidden_total": total,
            "failing_test_names": failing + ["tests/test_public_sanity.py::test_pub"],
        },
        "artifacts_file": f"{stem}.artifacts.json" if artifacts else None,
    }
    path = runs_dir / f"{stem}.json"
    path.write_text(json.dumps(log))
    if artifacts:
        agent_log: list[dict[str, Any]] = []
        if arch == "sequential":
            agent_log = [
                {"step": 1, "role": "plan", "response": "plan text"},
                {"step": 2, "role": "implement",
                 "response": _block(implement) if implement is not None else "no code"},
                {"step": 3, "role": "review", "response": review},
                {"step": 4, "role": "fix",
                 "response": _block(fix) if fix is not None else "unchanged"},
            ]
        (runs_dir / f"{stem}.artifacts.json").write_text(json.dumps({
            "run_id": run_id, "calls": [], "agent_log": agent_log,
            "modified_files": final_files, "evaluated_files": final_files,
        }))
    return path


def _analyze(path: Path):
    return st.analyze_run(path, json.loads(path.read_text()), fake_scorer)


# ---------------------------------------------------------------------------
# Review classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("**No definite correctness issues found.** The implementation meets the spec.", "no"),
    ("I found no definite bugs. There are several soft spots.", "no"),
    ("The implementation is essentially correct. No changes are required.", "no"),
    ("**Overall:** The implementation appears correct; no required changes identified.", "no"),
    ("## Issues found\n\n1. Hunk parsing is wrong.", "yes"),
    ("The implementation is **not fully correct**.", "yes"),
    ("Mostly correct, with one real edge-case bug.", "yes"),
    ("I found no major correctness bugs on the main paths. The issues below are edge cases.", "yes"),
    ("The implementation is correct, but the overlap check has an issue.", "yes"),
    ("Looks fine to me.", "unclear"),
    ("", "unclear"),
    (None, "unclear"),
])
def test_classify_review(text: str | None, expected: str) -> None:
    label, evidence = st.classify_review(text)
    assert label == expected
    assert bool(evidence) == (expected != "unclear")


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------


def test_implement_step_scored_from_artifacts(tmp_path: Path) -> None:
    """fail->pass: implement code is partial, the fixer repairs it."""
    path = write_run(tmp_path, "r1", implement="partial", fix="GOOD")
    row = _analyze(path)
    assert row.has_artifacts
    assert row.implement_files == FILE
    assert (row.implement_hidden_passed, row.implement_hidden_total) == (1, 3)
    assert row.implement_pass is False
    assert [n.split("::")[1] for n in row.implement_failing] == ["test_a", "test_b"]
    assert (row.final_hidden_passed, row.final_hidden_total, row.final_pass) == (3, 3, True)
    assert row.transition == "fail->pass"
    assert row.review_reported_issues == "yes"
    assert row.fix_changed_code is True
    assert row.final_rescore_matches_log is True


def test_scorer_receives_exact_implement_files(tmp_path: Path) -> None:
    """The implement step's parsed blocks are what gets scored (not the final files)."""
    seen: list[dict[str, str]] = []

    def recording(task: str, files: dict[str, str]):
        seen.append(dict(files))
        return fake_scorer(task, files)

    path = write_run(tmp_path, "r1", implement="impl code", fix="GOOD")
    st.analyze_run(path, json.loads(path.read_text()), recording)
    assert seen[0] == {FILE: "impl code\n"}
    assert seen[1] == {FILE: "GOOD\n"}  # drift-check re-score of the final files


def test_pass_to_fail_and_unchanged_fix(tmp_path: Path) -> None:
    row = _analyze(write_run(tmp_path, "r1", implement="GOOD", fix="worse"))
    assert row.transition == "pass->fail"
    row = _analyze(write_run(tmp_path, "r2", implement="GOOD", fix=None,
                             review="No definite issues found."))
    assert row.transition == "pass->pass"
    assert row.fix_changed_code is False
    assert row.review_reported_issues == "no"


def test_collection_error_counts_as_fail(tmp_path: Path) -> None:
    row = _analyze(write_run(tmp_path, "r1", implement="BROKEN", fix="GOOD"))
    assert (row.implement_hidden_passed, row.implement_hidden_total) == (0, 0)
    assert row.implement_pass is False
    assert row.transition == "fail->pass"


def test_implement_without_code_blocks(tmp_path: Path) -> None:
    row = _analyze(write_run(tmp_path, "r1", implement=None, fix="GOOD"))
    assert row.implement_files == ""
    assert row.transition == "fail->pass"
    assert "no labeled code blocks" in row.note


def test_missing_artifacts_sidecar(tmp_path: Path) -> None:
    row = _analyze(write_run(tmp_path, "r1", artifacts=False, final=(2, 3, [])))
    assert not row.has_artifacts
    assert row.implement_hidden_total is None
    assert row.transition == ""
    assert row.note == "no artifacts sidecar"
    assert (row.final_hidden_passed, row.final_hidden_total) == (2, 3)


def test_drift_detected_when_rescore_differs(tmp_path: Path) -> None:
    row = _analyze(write_run(tmp_path, "r1", implement="GOOD", final=(2, 3, [])))
    assert row.final_rescore_matches_log is False


def test_public_failures_excluded_from_hidden_failing(tmp_path: Path) -> None:
    row = _analyze(write_run(tmp_path, "r1", implement="partial"))
    assert all(n.startswith("eval/") for n in row.final_failing)
    assert len(row.final_failing) == 2


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_transitions_rates_and_single_agent(tmp_path: Path) -> None:
    paths = [
        write_run(tmp_path, "s1", implement="partial", fix="GOOD"),   # fail->pass
        write_run(tmp_path, "s2", implement="GOOD", fix=None),        # pass->pass
        write_run(tmp_path, "s3", implement="GOOD", fix="worse"),     # pass->fail
        write_run(tmp_path, "s4", artifacts=False, final=(3, 3, [])),  # unscored
        write_run(tmp_path, "s5", model="m2", implement="partial", fix="partial"),
    ]
    rows = [_analyze(p) for p in paths]
    singles = [
        json.loads(write_run(tmp_path, f"a{i}", arch="single_agent", artifacts=False,
                             final=f).read_text())
        for i, f in enumerate([(3, 3, []), (1, 3, []), (3, 3, []), (0, 0, [])])
    ]
    singles.append(json.loads(write_run(tmp_path, "a9", arch="single_agent", regime="production",
                                        artifacts=False, final=(3, 3, [])).read_text()))

    summary = {s["model"]: s for s in st.summarize(rows, singles)}
    m1 = summary["m1"]
    assert (m1["n_seq"], m1["n_scored"]) == (4, 3)
    assert (m1["pass->pass"], m1["fail->pass"], m1["pass->fail"], m1["fail->fail"]) == (1, 1, 1, 0)
    assert m1["impl_rate"] == pytest.approx((1 / 3 + 1 + 1) / 3)
    assert m1["final_rate"] == pytest.approx((1 + 1 + 1 / 3) / 3)
    assert m1["seq_pass"] == pytest.approx(3 / 4)  # s4's logged pass counts here
    assert m1["n_single"] == 4                    # production single-agent run excluded
    assert m1["single_pass"] == pytest.approx(2 / 4)
    assert m1["review_issues"] == 3
    m2 = summary["m2"]
    assert m2["fail->fail"] == 1
    assert m2["single_pass"] is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_main_writes_csv_and_prints(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "s1", implement="partial", fix="GOOD")
    write_run(runs, "s2", implement="GOOD", review="No changes required.")
    write_run(runs, "a1", arch="single_agent", implement="partial")
    write_run(runs, "x1", artifacts=False, final=(0, 0, []))
    (runs / "batch_summary.json").write_text("{}")
    before = _snapshot(runs)
    out_csv = tmp_path / "out.csv"

    rc = st.main(["diffapply", "--runs-dir", str(runs), "--csv", str(out_csv)],
                 scorer=fake_scorer)

    assert rc == 0
    assert _snapshot(runs) == before
    rows = list(csv.DictReader(out_csv.open()))
    assert [r["run_id"] for r in rows] == ["s1", "s2", "x1"]
    s1 = rows[0]
    assert (s1["implement_hidden_passed"], s1["implement_hidden_total"]) == ("1", "3")
    assert (s1["final_hidden_passed"], s1["final_hidden_total"]) == ("3", "3")
    assert s1["transition"] == "fail->pass"
    assert s1["implement_failing"] == "test_a;test_b"
    assert rows[1]["review_reported_issues"] == "no"
    assert rows[2]["implement_hidden_total"] == ""

    out = capsys.readouterr().out
    assert "single@1" in out
    assert "Failing hidden tests per run" in out
    # Single-agent runs are listed with their failing tests too.
    assert "single_agent" in out and "test_a, test_b" in out
    assert "impl  1/3: test_a, test_b" in out


def test_main_refuses_csv_inside_runs_or_eval(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "s1")
    for bad in (_REPO / "runs" / "x.csv", _REPO / "eval" / "x.csv"):
        with pytest.raises(SystemExit, match="refusing"):
            st.main(["diffapply", "--runs-dir", str(runs), "--csv", str(bad)],
                    scorer=fake_scorer)
        assert not bad.exists()


def test_main_no_runs(tmp_path: Path) -> None:
    assert st.main(["diffapply", "--runs-dir", str(tmp_path),
                    "--csv", str(tmp_path / "o.csv")], scorer=fake_scorer) == 1


# ---------------------------------------------------------------------------
# Real scorer (runs eval/run_eval.sh on a temp copy; no API calls)
# ---------------------------------------------------------------------------


def test_executor_scorer_scores_temp_copy_without_touching_repo() -> None:
    task_dir = _REPO / "agent-bench" / "tasks" / "diffapply"
    before = {p: p.read_bytes() for p in task_dir.rglob("*.py")}
    score = st.make_executor_scorer()

    passed, total, failing = score("diffapply", {FILE: "def broken(:\n"})
    assert (passed, total) == (0, 0)
    assert failing == ["hidden_eval_collection_error"]

    passed, total, _ = score("diffapply", {})  # untouched stub
    assert total > 0 and passed < total

    assert {p: p.read_bytes() for p in task_dir.rglob("*.py")} == before

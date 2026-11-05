"""Tests for scripts/rescore_runs.py, scripts/rescored.py and the --rescored
flag of aggregate_results.py, analyze_stages.py and classify_failures.py."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"


def _load(name: str):
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rr = _load("rescore_runs")
rs = _load("rescored")
agg = _load("aggregate_results")
st = _load("analyze_stages")

TASK = "diffapply"
FILE = "diffapply/applier.py"
HIDDEN = f"eval/hidden_tests/{TASK}/test_hidden.py::"
# A file whose code contains a literal ```: the old parser cut it at the ```.
FULL = 'def f(s):\n    return s.startswith("```")\nGOOD = 1\n'
CUT = 'def f(s):\n    return s.startswith("'
PUBLIC_FAIL = "tests/test_public_sanity.py::test_pub"


def fake_score(files: dict[str, str]) -> tuple[int, int, list[str]]:
    src = files.get(FILE, "")
    if "GOOD = 1" in src:
        return 3, 3, []
    if src.endswith('("'):
        return 0, 0, ["hidden_eval_collection_error"]
    return 1, 3, [HIDDEN + "test_a", HIDDEN + "test_b"]


class FakeEvaluator:
    """In-memory stand-in for TempDirEvaluator (evaluated files = files)."""

    repo_root = _REPO

    def __init__(self) -> None:
        self.scored: list[dict[str, str]] = []

    def rebuild_and_score(
        self, task: str, files: dict[str, str], need_score: Callable[[dict[str, str]], bool],
    ):
        if not need_score(files):
            return files, [], None
        self.scored.append(files)
        return files, [], fake_score(files)


def _block(content: str) -> str:
    return f"Here:\n\n```{FILE}\n{content}```\n"


def write_run(
    runs: Path, run_id: str, *, arch: str = "single_agent", response: str | None = None,
    implement: str | None = None, fix: str | None = None, logged: str = CUT,
    artifacts: bool = True, policy_compliant: bool = True, task: str = TASK,
) -> Path:
    """A run whose logged files are *logged* (scored with fake_score)."""
    files = {FILE: logged}
    passed, total, failing = fake_score(files)
    status = "success" if total and passed == total else "failure"
    if not policy_compliant:
        status = "policy_violation"
    stem = f"26-11-05_1200_p_m1_{arch}_{task}_minimal_{run_id}"
    log = {
        "run_id": run_id, "task_name": task, "model": "m1", "provider": "p",
        "architecture_name": arch, "prompt_regime": "minimal",
        "requested_reasoning_effort": "low", "status": status,
        "policy_compliant": policy_compliant, "notes": None if policy_compliant else "x",
        "test_results": {"public_passed": 0, "public_total": 1, "hidden_passed": passed,
                         "hidden_total": total, "failing_test_names": failing + [PUBLIC_FAIL]},
        "llm_calls": [{"stop_reason": "end_turn"}],
        "artifacts_file": f"{stem}.artifacts.json" if artifacts else None,
    }
    path = runs / f"{stem}.json"
    path.write_text(json.dumps(log))
    if artifacts:
        if arch == "single_agent":
            agent_log = [{"role": "user", "content": "p"},
                         {"role": "assistant", "content": response or _block(FULL)}]
        else:
            agent_log = [{"role": "plan", "response": "p"}]
            if implement is not None:
                agent_log.append({"role": "implement", "response": implement})
            agent_log.append({"role": "review", "response": "Found a bug."})
            if fix is not None:
                agent_log.append({"role": "fix", "response": fix})
        (runs / f"{stem}.artifacts.json").write_text(json.dumps({
            "run_id": run_id, "calls": [], "agent_log": agent_log,
            "modified_files": files, "evaluated_files": files,
        }))
    return path


def _row(runs: Path, run_id: str, **kw: Any):
    path = write_run(runs, run_id, **kw)
    return rr.rescore_run(path, json.loads(path.read_text()), FakeEvaluator())


# ---------------------------------------------------------------------------
# rescore_runs
# ---------------------------------------------------------------------------


def test_cut_file_is_changed_and_rescored_to_success(tmp_path: Path) -> None:
    r = _row(tmp_path, "a")
    assert r.outcome == rr.CHANGED and r.files_changed == FILE
    assert (r.old_status, r.new_status) == ("failure", "success")
    assert (r.old_hidden_passed, r.old_hidden_total) == (0, 0)
    assert (r.new_hidden_passed, r.new_hidden_total) == (3, 3)
    assert r.new_policy_compliant is True


def test_identical_rebuild_is_unchanged_and_not_scored(tmp_path: Path) -> None:
    path = write_run(tmp_path, "b", response=_block("x = 1\n"), logged="x = 1\n")
    ev = FakeEvaluator()
    r = rr.rescore_run(path, json.loads(path.read_text()), ev)
    assert r.outcome == rr.UNCHANGED and ev.scored == []
    assert r.new_status == r.old_status


def test_sequential_merges_fix_over_implement(tmp_path: Path) -> None:
    impl = _block(FULL) + f"\n```diffapply/other.py\nO = 1\n```\n"
    fix = _block(FULL.replace("GOOD = 1", "GOOD = 1\nFIXED = 1"))
    files, _ = rr.rebuild_final_files("sequential", {"agent_log": [
        {"role": "implement", "response": impl}, {"role": "fix", "response": fix},
    ]})
    assert files is not None
    assert "FIXED = 1" in files[FILE] and files["diffapply/other.py"] == "O = 1\n"


def test_sequential_without_fix_not_rebuildable(tmp_path: Path) -> None:
    r = _row(tmp_path, "c", arch="sequential", implement=_block(FULL), fix=None)
    assert r.outcome == rr.NOT_REBUILDABLE
    assert r.new_status == r.old_status


def test_no_artifacts(tmp_path: Path) -> None:
    assert _row(tmp_path, "d", artifacts=False).outcome == rr.NO_ARTIFACTS


def test_non_compliant_run_stays_policy_violation(tmp_path: Path) -> None:
    r = _row(tmp_path, "e", policy_compliant=False)
    assert r.outcome == rr.CHANGED
    assert r.new_status == "policy_violation" and r.new_policy_compliant is False


def test_no_code_blocks_after_reparse_is_failure(tmp_path: Path) -> None:
    r = _row(tmp_path, "f", response="no code here")
    assert r.outcome == rr.CHANGED and r.new_status == "failure"


def test_main_writes_csv_per_task_and_leaves_runs(tmp_path: Path, capsys) -> None:
    runs, out = tmp_path / "runs", tmp_path / "analysis"
    runs.mkdir()
    write_run(runs, "a")
    write_run(runs, "b", response=_block("x = 1\n"), logged="x = 1\n")
    write_run(runs, "c", task="snake")
    before = {p.name: p.read_bytes() for p in runs.iterdir()}
    assert rr.main(["--runs-dir", str(runs), "--out-dir", str(out)], evaluator=FakeEvaluator()) == 0
    assert {p.name: p.read_bytes() for p in runs.iterdir()} == before
    text = capsys.readouterr().out
    assert "diffapply: checked 2, unchanged 1, changed 1" in text
    assert "failure -> success: 1" in text
    assert (out / "rescore_diffapply.csv").is_file() and (out / "rescore_snake.csv").is_file()


def test_main_refuses_out_dir_in_runs_or_eval() -> None:
    for d in ("runs", "eval"):
        with pytest.raises(SystemExit):
            rr.main(["--out-dir", str(_REPO / d / "x")])


def test_real_evaluator_unchanged_for_stub(tmp_path: Path) -> None:
    """TempDirEvaluator rebuilds evaluated files like the executor."""
    ev = rr.TempDirEvaluator()
    stub = (_REPO / "agent-bench" / "tasks" / TASK / FILE).read_text()
    evaluated, violations, score = ev.rebuild_and_score(TASK, {}, lambda e: False)
    assert evaluated == {FILE: stub} and violations == [] and score is None


# ---------------------------------------------------------------------------
# rescored.py and the --rescored flags
# ---------------------------------------------------------------------------


def _rescore_dir(tmp_path: Path, runs: Path) -> Path:
    out = tmp_path / "analysis"
    rr.main(["--runs-dir", str(runs), "--out-dir", str(out)], evaluator=FakeEvaluator())
    return out


def test_apply_replaces_hidden_keeps_public(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    path = write_run(runs, "a")
    res = rs.require_rescore([TASK], _rescore_dir(tmp_path, runs))
    log = json.loads(path.read_text())
    new = res.apply(log)
    assert new is not log and new["rescored"] is True
    assert new["status"] == "success"
    assert new["test_results"]["hidden_passed"] == 3
    assert new["test_results"]["failing_test_names"] == [PUBLIC_FAIL]
    assert log["status"] == "failure"  # original untouched
    assert "1 of 1 runs affected" in res.header([log])


def test_require_rescore_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="rescore_runs.py"):
        rs.require_rescore(["nope"], tmp_path)


def test_aggregate_results_rescored(tmp_path: Path, capsys) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "a")
    write_run(runs, "b", response=_block("x = 1\n"), logged="x = 1\n")
    adir = _rescore_dir(tmp_path, runs)
    capsys.readouterr()
    agg.main(["--runs-dir", str(runs)])
    plain = capsys.readouterr().out
    assert "RE-SCORED" not in plain and "  0.0%" in plain
    agg.main(["--runs-dir", str(runs), "--rescored", "--analysis-dir", str(adir)])
    out = capsys.readouterr().out
    assert "RE-SCORED RESULTS IN USE" in out and "1 of 2 runs affected" in out
    assert " 50.0%" in out


def test_analyze_stages_rescored(tmp_path: Path, capsys) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "s", arch="sequential", implement=_block("x = 1\n"), fix=_block(FULL))
    adir = _rescore_dir(tmp_path, runs)
    capsys.readouterr()
    st.main([TASK, "--runs-dir", str(runs), "--csv", str(tmp_path / "s.csv"),
             "--rescored", "--analysis-dir", str(adir)],
            scorer=lambda task, files: fake_score(files))
    out = capsys.readouterr().out
    assert "RE-SCORED RESULTS IN USE" in out and "1 of 1 runs affected" in out
    row = next(csv.DictReader((tmp_path / "s.csv").open()))
    assert row["transition"] == "fail->pass"
    assert row["final_rescore_matches_log"] == "True"


def test_classify_failures_rescored_header(tmp_path: Path, capsys) -> None:
    cf = _load("classify_failures")
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "m", task="mdtable")
    adir = _rescore_dir(tmp_path, runs)
    capsys.readouterr()
    seen: list[dict[str, str]] = []

    def inspector(files: dict[str, str]) -> dict[str, Any]:
        seen.append(files)
        return {"import_error": None, "probes": {}}

    cf.main(["--runs-dir", str(runs), "--csv", str(tmp_path / "c.csv"), "--rescored",
             "--analysis-dir", str(adir)],
            scorer=lambda task, files: fake_score(files), inspector=inspector)
    out = capsys.readouterr().out
    assert "RE-SCORED RESULTS IN USE" in out and "1 of 1 runs affected" in out
    assert seen == [{FILE: FULL}]  # final code re-parsed, not the cut file

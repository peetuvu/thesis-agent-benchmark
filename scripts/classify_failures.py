#!/usr/bin/env python3
"""Failure classification for the mdtable task: why did each run fail?

Usage:
  python3 scripts/classify_failures.py [--runs-dir runs] [--csv out.csv]

Run from the thesis-agents repo root. Makes no API calls and writes nothing
to eval/ or runs/ (the CSV path is refused if it points inside either).

Code versions classified
  - every run: the final code (artifacts.modified_files, i.e. what the
    executor applied), with the hidden-test result recorded in the run log;
  - sequential runs: also the implement-step code, parsed from the
    .artifacts.json sidecar and re-scored against the hidden tests exactly as
    scripts/analyze_stages.py does (fresh temp copy of the task, executor's
    _apply_modified_files + _run_hidden_eval).

For each code version the files are rebuilt in a fresh temp copy of
agent-bench/tasks/mdtable and one child Python process (python -I, 30 s
timeout) does two things:

  1. Import check. Runs the hidden test module's top-level import statements
     (read from eval/hidden_tests/mdtable/test_hidden.py with ast) one at a
     time and reports the first exception: missing name (ImportError
     "cannot import name"), SyntaxError (incl. IndentationError), module not
     found, or other (exception type + message).
  2. Ambiguity probes. Calls format_markdown on small inputs, one per README
     ambiguity in docs/task_summary.md section 4, and compares only the lines
     that ambiguity governs with the reading the hidden tests assume. A probe
     "deviates" when format_markdown returns a different string; a probe that
     raises is recorded as an error, not as a deviation.

Why probes and not just test names: the doc cites only some of the tests that
depend on each ambiguity. Choosing a 3-character minimum column width (#3),
for example, breaks every test with a 1-character column, including tests the
doc does not cite for #3. The doc-cited ambiguities of the failing tests are
still printed next to each signature for reference.

Failure classes (one per code version)
  pass              every hidden test passed
  no_code           the step produced no labeled code blocks (stub scored)
  collection_error  the hidden tests could not be collected; detail = the
                    import check's exact error. parser_cut = True when a
                    parsed file does not end at a line break, i.e. the
                    closing ``` that ended the block stood mid-line: the
                    response's code contained a literal ``` (e.g.
                    startswith("```")) and architectures.utils
                    .parse_code_blocks, whose closing fence is not
                    line-anchored, cut the file there. That is a harness
                    parsing artifact, not code the model wrote.
  no_results        no tests ran for another reason (e.g. timeout)
  interpretation    some tests failed and >= 1 ambiguity probe deviates
  other_failure     some tests failed and no probe deviates (a bug, or an
                    aspect the README states unambiguously)

--rescored: for runs whose result changed under the fixed code-block parser
(analysis/rescore_mdtable.csv, from scripts/rescore_runs.py), the final
score comes from the CSV and the final code is re-parsed from the saved
responses with the fixed parser instead of taken from
artifacts.modified_files. The header says whether re-scored results are in
use and how many runs were affected. The implement step is always parsed
with the current parser.

Validation: every probe is also run on passing code. A deviation there means
the probe is stricter than the hidden tests; such cases are reported as
"probe false positives" and should be 0.

Outputs
  - CSV, one row per code version (default: ./failure_classes_mdtable.csv)
  - stdout: the exact error for every collection error
  - stdout: failing-test signatures (identical failing sets) with counts and
    the ambiguities they correspond to
  - stdout: per (model, architecture, regime): counts per class, and for
    sequential runs the implement -> final class transitions (what the
    review/fix steps repaired or broke)
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for _p in (str(_REPO_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_stages import (  # noqa: E402
    SEQUENTIAL, Scorer, _check_output_path, _short, _step_response,
    hidden_failing, is_pass, load_runs, make_executor_scorer,
)
from architectures.utils import parse_code_blocks  # noqa: E402
from runner.config import TaskPaths  # noqa: E402
from runner.executor import Executor  # noqa: E402

TASK = "mdtable"
RUNS_DIR = Path("runs")
HIDDEN_TEST_FILE = _REPO_ROOT / "eval" / "hidden_tests" / TASK / "test_hidden.py"
TASK_SUMMARY = _REPO_ROOT / "docs" / "task_summary.md"
COLLECTION_MARKER = "hidden_eval_collection_error"
CHILD_TIMEOUT = 30

CLASSES = ("pass", "collection_error", "interpretation", "other_failure",
           "no_results", "no_code")
_CLASS_LABEL = {
    "pass": "pass", "collection_error": "collection error",
    "interpretation": "interpretation", "other_failure": "other failure",
    "no_results": "no results", "no_code": "no code",
}

# ---------------------------------------------------------------------------
# Ambiguity probes (numbers = rows of the ambiguity table in
# docs/task_summary.md, section 4). Expected values follow the hidden tests'
# reading as that table states it. Only `lines` of the output are compared
# (None = blank-line pattern of the whole output), so each probe isolates one
# ambiguity as far as possible.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One format_markdown call that isolates one README ambiguity."""

    name: str
    ambiguity: int
    text: str
    expected: str
    lines: tuple[int, ...] | None  # line indices compared; None = blank pattern


PROBES: tuple[Probe, ...] = (
    Probe("escaped_pipe_kept_and_counted", 1,
          "| x \\| y | col |\n|---|---|\n| 1 | 2 |",
          "| x \\| y | col |\n|--------|-----|\n| 1      | 2   |", (0, 2)),
    Probe("separator_width_plus_2_no_spaces", 2,
          "| abc | defg |\n|---|---|\n| 1 | 2 |",
          "| abc | defg |\n|-----|------|\n| 1   | 2    |", (1,)),
    Probe("separator_colons_inside_span", 2,
          "| Left | Right | Center |\n|:---|---:|:---:|\n| a | b | c |",
          "| Left | Right | Center |\n|:-----|------:|:------:|\n| a    |     b |   c    |",
          (1,)),
    Probe("no_minimum_column_width", 3,
          "|A|B|\n|---|---|\n|1|2|",
          "| A | B |\n|---|---|\n| 1 | 2 |", (0, 2)),
    Probe("default_alignment_left_justified", 4,
          "| abc | defg |\n|---|---|\n| 1 | 2 |",
          "| abc | defg |\n|-----|------|\n| 1   | 2    |", (2,)),
    Probe("blank_lines_kept_none_added", 6,
          "|A|\n|---|\n|1|\n\n\n|B|\n|---|\n|2|",
          "| A |\n|---|\n| 1 |\n\n\n| B |\n|---|\n| 2 |", None),
)


def _view(text: str, lines: tuple[int, ...] | None) -> list[Any]:
    """The part of an output a probe compares."""
    split = text.split("\n")
    if lines is None:
        return [ln.strip() == "" for ln in split]
    return [split[i] if i < len(split) else None for i in lines]


def probe_deviates(probe: Probe, output: str) -> bool:
    """True when *output* takes a different reading than the hidden tests."""
    return _view(output, probe.lines) != _view(probe.expected, probe.lines)


def load_ambiguity_doc(path: Path = TASK_SUMMARY) -> tuple[dict[int, str], dict[str, set[int]]]:
    """Parse the mdtable ambiguity table in docs/task_summary.md.

    Returns ({number: short title}, {test name: ambiguity numbers whose
    "hidden tests assume" column cites it}). Empty dicts if not found.
    """
    titles: dict[int, str] = {}
    cited: dict[str, set[int]] = defaultdict(set)
    try:
        text = path.read_text()
    except OSError:
        return titles, dict(cited)
    m = re.search(r"^## 4\..*?(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return titles, dict(cited)
    for line in m.group(0).splitlines():
        row = re.match(r"\|\s*(\d+)\s*\|\s*\*\*(.+?)\*\*(.*?)\|(.*)\|\s*$", line)
        if not row:
            continue
        num = int(row.group(1))
        titles[num] = row.group(2).rstrip(".")
        for test in re.findall(r"`(test_\w+)`", row.group(4)):
            cited[test].add(num)
    return titles, dict(cited)


# ---------------------------------------------------------------------------
# Child process: import check + probes
# ---------------------------------------------------------------------------

_CHILD = r"""
import json, sys
src_dir, payload = sys.argv[1], json.loads(sys.argv[2])
sys.path.insert(0, src_dir)
out = {"import_error": None, "probes": {}}
for stmt in payload["imports"]:
    try:
        exec(stmt, {})
    except BaseException as e:
        err = {"stmt": stmt, "type": type(e).__name__, "msg": str(e),
               "syntax": isinstance(e, SyntaxError),
               "name": getattr(e, "name", None)}
        if isinstance(e, SyntaxError):
            err.update(file=e.filename, line=e.lineno, text=(e.text or "").strip(),
                       msg=e.msg)
        out["import_error"] = err
        break
if out["import_error"] is None:
    try:
        from mdtable import format_markdown
    except BaseException as e:
        format_markdown = None
        perr = f"{type(e).__name__}: {e}"
    for name, text in payload["probes"].items():
        if format_markdown is None:
            out["probes"][name] = {"error": perr}
            continue
        try:
            res = format_markdown(text)
            out["probes"][name] = {"output": res if isinstance(res, str) else repr(res),
                                   "is_str": isinstance(res, str)}
        except BaseException as e:
            out["probes"][name] = {"error": f"{type(e).__name__}: {e}"}
print("@@RESULT@@" + json.dumps(out))
"""


def hidden_import_statements(path: Path = HIDDEN_TEST_FILE) -> list[str]:
    """Top-level import statements of the hidden test module, as source."""
    tree = ast.parse(path.read_text())
    return [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def _python() -> str:
    """Same interpreter choice as eval/run_eval.sh."""
    venv = _REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


# Inspector signature: (files) -> child result dict (see _CHILD), or
# {"child_error": "..."} when the child itself failed.
Inspector = Callable[[dict[str, str]], dict[str, Any]]


def make_inspector(repo_root: Path = _REPO_ROOT) -> Inspector:
    """Inspector that rebuilds the evaluated files in a temp copy of the task."""
    executor = Executor(repo_root=repo_root)
    paths = TaskPaths(TASK, repo_root)
    payload = json.dumps({
        "imports": hidden_import_statements(),
        "probes": {p.name: p.text for p in PROBES},
    })

    def inspect(files: dict[str, str]) -> dict[str, Any]:
        tmp = tempfile.mkdtemp(prefix=f"classify_{TASK}_")
        try:
            task_dir = Path(tmp) / "task"
            shutil.copytree(paths.task_dir, task_dir)
            executor._apply_modified_files(files, task_dir)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            try:
                proc = subprocess.run(
                    [_python(), "-I", "-c", _CHILD, str(task_dir), payload],
                    capture_output=True, text=True, timeout=CHILD_TIMEOUT,
                    cwd=tmp, env=env,
                )
            except subprocess.TimeoutExpired:
                return {"child_error": f"timeout after {CHILD_TIMEOUT}s"}
            for line in reversed(proc.stdout.splitlines()):
                if line.startswith("@@RESULT@@"):
                    res = json.loads(line[len("@@RESULT@@"):])
                    if res.get("import_error", {}) and res["import_error"].get("file"):
                        f = res["import_error"]["file"]
                        res["import_error"]["file"] = (
                            os.path.relpath(f, task_dir) if f.startswith(str(task_dir)) else f
                        )
                    return res
            tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
            return {"child_error": f"exit {proc.returncode}: {tail}"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return inspect


def describe_import_error(err: dict[str, Any] | None) -> tuple[str, str]:
    """(kind, exact detail) for the import check's first exception."""
    if err is None:
        return "", ""
    if err.get("syntax"):
        where = f"{err.get('file')}:{err.get('line')}"
        text = f" `{err['text']}`" if err.get("text") else ""
        return "SyntaxError", f"{err['type']}: {err['msg']} at {where}{text}"
    m = re.search(r"cannot import name '([^']+)' from '([^']+)'", err["msg"])
    if err["type"] == "ImportError" and m:
        return "missing name", f"missing name '{m.group(1)}' in {m.group(2)} ({err['stmt']})"
    if err["type"] == "ModuleNotFoundError":
        return "module not found", f"ModuleNotFoundError: {err['msg']} ({err['stmt']})"
    return "other", f"{err['type']}: {err['msg']} ({err['stmt']})"


# ---------------------------------------------------------------------------
# Per code version
# ---------------------------------------------------------------------------


@dataclass
class CodeVersion:
    """One code version (final or implement) of one run, classified."""

    run_id: str
    log_file: str
    model: str
    effort: str
    architecture: str
    regime: str
    stage: str  # "final" or "implement"
    passed: int | None = None
    total: int | None = None
    failing: list[str] = field(default_factory=list)
    failure_class: str = ""
    collection_kind: str = ""
    collection_detail: str = ""
    deviations: list[str] = field(default_factory=list)  # probe names
    ambiguities: list[int] = field(default_factory=list)
    probe_errors: list[str] = field(default_factory=list)
    signature: str = ""
    parser_cut: bool = False
    note: str = ""

    def csv_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "log_file": self.log_file, "model": self.model,
            "effort": self.effort, "architecture": self.architecture,
            "regime": self.regime, "stage": self.stage,
            "passed": "" if self.passed is None else self.passed,
            "total": "" if self.total is None else self.total,
            "failure_class": self.failure_class,
            "collection_kind": self.collection_kind,
            "collection_detail": self.collection_detail,
            "parser_cut": self.parser_cut,
            "signature": self.signature,
            "ambiguities": ";".join(str(a) for a in self.ambiguities),
            "probe_deviations": ";".join(self.deviations),
            "probe_errors": ";".join(self.probe_errors),
            "failing": ";".join(self.failing),
            "note": self.note,
        }


def block_cut_mid_line(content: str) -> bool:
    """True if a parsed code block did not end at a line break.

    parse_code_blocks stops at the first ``` anywhere; a well-formed block's
    closing fence starts a line, so its content ends with a newline (plus at
    most indentation before the fence).
    """
    return bool(content) and not content.rstrip(" \t").endswith("\n")


def classify(
    cv: CodeVersion, files: dict[str, str] | None, inspector: Inspector,
) -> None:
    """Fill in failure_class and diagnostics for *cv* (scores already set)."""
    if files is not None and not files:
        cv.note = (cv.note + "; " if cv.note else "") + "no labeled code blocks"
    if files is None:
        cv.failure_class = "pass" if is_pass(cv.passed, cv.total) else "no_results"
        cv.note = (cv.note + "; " if cv.note else "") + "code unavailable, not inspected"
        return

    res = inspector(files)
    if "child_error" in res:
        cv.probe_errors.append(f"inspector: {res['child_error']}")
        res = {"import_error": None, "probes": {}}
    cv.collection_kind, cv.collection_detail = describe_import_error(res.get("import_error"))
    for probe in PROBES:
        r = res.get("probes", {}).get(probe.name)
        if r is None:
            continue
        if "error" in r:
            cv.probe_errors.append(f"{probe.name}: {r['error']}")
        elif not r.get("is_str") or probe_deviates(probe, r["output"]):
            cv.deviations.append(probe.name)
    cv.ambiguities = sorted({p.ambiguity for p in PROBES if p.name in cv.deviations})

    failing_markers = cv.failing
    if is_pass(cv.passed, cv.total):
        cv.failure_class = "pass"
    elif not files:
        cv.failure_class = "no_code"
    elif COLLECTION_MARKER in failing_markers:
        cv.failure_class = "collection_error"
        cv.parser_cut = any(block_cut_mid_line(c) for c in files.values())
        if not cv.collection_kind:
            cv.collection_kind = "imports ok"
            cv.collection_detail = ("hidden-test imports succeed in a rebuilt temp dir; "
                                    "collection failed for another reason")
    elif not cv.total:
        cv.failure_class = "no_results"
    elif cv.deviations:
        cv.failure_class = "interpretation"
    else:
        cv.failure_class = "other_failure"


def _base(log_path: Path, log: dict[str, Any], stage: str) -> CodeVersion:
    return CodeVersion(
        run_id=log["run_id"], log_file=log_path.name, model=log["model"],
        effort=log.get("requested_reasoning_effort") or "",
        architecture=log["architecture_name"], regime=log.get("prompt_regime") or "",
        stage=stage,
    )


def _failing_short(names: list[str]) -> list[str]:
    return sorted(_short(n) for n in hidden_failing(names))


def analyze_run(
    log_path: Path, log: dict[str, Any], scorer: Scorer, inspector: Inspector,
    rebuild_final: bool = False,
) -> list[CodeVersion]:
    """Classify the final code (and, for sequential runs, the implement code)."""
    tr = log["test_results"]
    final = _base(log_path, log, "final")
    final.passed, final.total = tr.get("hidden_passed", 0), tr.get("hidden_total", 0)
    final.failing = _failing_short(tr.get("failing_test_names") or [])

    artifacts: dict[str, Any] | None = None
    art_name = log.get("artifacts_file")
    art_path = log_path.parent / art_name if art_name else None
    if art_path is not None and art_path.is_file():
        try:
            artifacts = json.loads(art_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            final.note = f"unreadable artifacts: {e}"
    else:
        final.note = "no artifacts sidecar"

    final_files = (artifacts.get("modified_files") or {}) if artifacts is not None else None
    if rebuild_final and artifacts is not None:
        from rescore_runs import rebuild_final_files
        rebuilt, _ = rebuild_final_files(log["architecture_name"], artifacts)
        if rebuilt is not None:
            final_files = rebuilt
            final.note = (final.note + "; " if final.note else "") + "re-scored"
    classify(final, final_files, inspector)
    out = [final]

    if log["architecture_name"] != SEQUENTIAL or artifacts is None:
        return out
    impl = _base(log_path, log, "implement")
    text = _step_response(artifacts, "implement")
    if text is None:
        impl.note = "no implement step response in artifacts"
        impl.failure_class = "no_results"
        return out + [impl]
    files = parse_code_blocks(text)
    impl.passed, impl.total, failing = scorer(TASK, files)
    impl.failing = _failing_short(failing)
    classify(impl, files, inspector)
    return out + [impl]


# ---------------------------------------------------------------------------
# Signatures and summary
# ---------------------------------------------------------------------------


@dataclass
class Signature:
    """An identical set of failing hidden tests."""

    sid: str
    tests: tuple[str, ...]
    members: list[CodeVersion]


def build_signatures(versions: list[CodeVersion]) -> list[Signature]:
    """Group code versions with results and >= 1 failing test; set .signature."""
    groups: dict[tuple[str, ...], list[CodeVersion]] = defaultdict(list)
    for cv in versions:
        if cv.total and cv.failure_class in ("interpretation", "other_failure"):
            groups[tuple(cv.failing)].append(cv)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), len(kv[0]), kv[0]))
    sigs = []
    for i, (tests, members) in enumerate(ordered, 1):
        sid = f"S{i}"
        for cv in members:
            cv.signature = sid
        sigs.append(Signature(sid, tests, members))
    return sigs


def print_collection_errors(versions: list[CodeVersion], file: Any = None) -> None:
    file = file or sys.stdout
    rows = [cv for cv in versions if cv.failure_class == "collection_error"]
    print(f"\nCollection errors ({len(rows)} code versions)", file=file)
    print("=" * 40, file=file)
    kinds = Counter(cv.collection_kind for cv in rows)
    print("  by kind: " + (", ".join(f"{k}: {n}" for k, n in kinds.most_common()) or "-"), file=file)
    cut = sum(cv.parser_cut for cv in rows)
    print(f"  file cut at a mid-line ``` by parse_code_blocks (harness artifact): "
          f"{cut}/{len(rows)}", file=file)
    details = Counter(cv.collection_detail for cv in rows)
    print("  distinct errors:", file=file)
    for d, n in details.most_common():
        print(f"    {n:>3}  {d}", file=file)
    print("  per code version:", file=file)
    for cv in rows:
        print(f"    {cv.architecture:<13} {cv.model:<18} {cv.regime:<10} "
              f"{cv.run_id[:8]} {cv.stage:<9}  {'[cut] ' if cv.parser_cut else ''}"
              f"{cv.collection_detail}", file=file)


def print_signatures(
    sigs: list[Signature], titles: dict[int, str], cited: dict[str, set[int]],
    file: Any = None,
) -> None:
    file = file or sys.stdout

    def amb(nums: Any) -> str:
        return ", ".join(f"#{n} {titles.get(n, '?')}" for n in sorted(nums)) or "none"

    print("\nFailing-test signatures (code versions with results)", file=file)
    print("=" * 53, file=file)
    for s in sigs:
        n_final = sum(cv.stage == "final" for cv in s.members)
        n_impl = len(s.members) - n_final
        amb_counts = Counter(a for cv in s.members for a in cv.ambiguities)
        no_dev = sum(not cv.ambiguities for cv in s.members)
        cited_nums = {n for t in s.tests for n in cited.get(t, set())}
        print(f"\n{s.sid}: {len(s.tests)} failing, {len(s.members)} code versions "
              f"(final {n_final}, implement {n_impl})", file=file)
        print("  failing: " + ", ".join(s.tests), file=file)
        dev = "; ".join(f"#{a} {titles.get(a, '?')} ({n}/{len(s.members)})"
                        for a, n in sorted(amb_counts.items()))
        if no_dev:
            dev = (dev + "; " if dev else "") + f"no probe deviates ({no_dev}/{len(s.members)})"
        print(f"  README ambiguity (probes): {dev}", file=file)
        print(f"  doc-cited ambiguities of the failing tests: {amb(cited_nums)}", file=file)
        uncited = [t for t in s.tests if t not in cited]
        if uncited:
            print("  failing tests the doc cites for no ambiguity: " + ", ".join(uncited), file=file)
        cells = Counter(f"{cv.model}/{cv.architecture}/{cv.regime}/{cv.stage}" for cv in s.members)
        print("  from: " + ", ".join(f"{k} x{n}" for k, n in sorted(cells.items())), file=file)


def print_summary(versions: list[CodeVersion], file: Any = None) -> None:
    """Counts per class per (model, architecture, regime), plus transitions."""
    file = file or sys.stdout
    finals: dict[tuple[str, str, str], list[CodeVersion]] = defaultdict(list)
    impls: dict[str, CodeVersion] = {}
    for cv in versions:
        if cv.stage == "final":
            finals[(cv.model, cv.architecture, cv.regime)].append(cv)
        else:
            impls[cv.run_id] = cv

    efforts = sorted({cv.effort or "-" for cv in versions})
    print(f"\nSummary of final code per (model, architecture, regime); "
          f"reasoning effort(s): {', '.join(efforts)}", file=file)
    print("=" * 80, file=file)
    abbrev = {"pass": "pass", "collection_error": "coll", "interpretation": "interp",
              "other_failure": "other", "no_results": "nores", "no_code": "nocode"}
    hdr = (f"{'model':<18} {'arch':<13} {'regime':<10} {'n':>3}  "
           + " ".join(f"{abbrev[c]:>6}" for c in CLASSES) + "  collection kinds")
    print(hdr, file=file)
    print("-" * len(hdr), file=file)
    for key in sorted(finals):
        rows = finals[key]
        c = Counter(cv.failure_class for cv in rows)
        kinds = Counter(cv.collection_kind for cv in rows if cv.failure_class == "collection_error")
        print(f"{key[0]:<18} {key[1]:<13} {key[2]:<10} {len(rows):>3}  "
              + " ".join(f"{c.get(k, 0):>6}" for k in CLASSES)
              + "  " + (", ".join(f"{k}: {n}" for k, n in kinds.most_common()) or "-"),
              file=file)

    seq_keys = [k for k in sorted(finals) if k[1] == SEQUENTIAL]
    if not seq_keys:
        return
    print("\nSequential runs: implement-step class -> final class", file=file)
    print("=" * 52, file=file)
    total: Counter[tuple[str, str]] = Counter()
    for key in seq_keys:
        trans: Counter[tuple[str, str]] = Counter()
        for cv in finals[key]:
            impl = impls.get(cv.run_id)
            if impl is None or not impl.failure_class:
                continue
            trans[(impl.failure_class, cv.failure_class)] += 1
        total.update(trans)
        print(f"{key[0]:<18} {key[2]:<10} {_fmt_trans(trans)}", file=file)
    print(f"{'ALL':<18} {'':<10} {_fmt_trans(total)}", file=file)
    repaired = Counter(a for (a, b), n in total.items() for _ in range(n)
                       if a != "pass" and b == "pass")
    broke = Counter(b for (a, b), n in total.items() for _ in range(n)
                    if a == "pass" and b != "pass")
    print("\n  fix step repaired: "
          + (", ".join(f"{_CLASS_LABEL[k]} → pass: {n}" for k, n in repaired.most_common()) or "-"),
          file=file)
    print("  fix step broke:    "
          + (", ".join(f"pass → {_CLASS_LABEL[k]}: {n}" for k, n in broke.most_common()) or "-"),
          file=file)


def _fmt_trans(trans: Counter[tuple[str, str]]) -> str:
    order = {c: i for i, c in enumerate(CLASSES)}
    items = sorted(trans.items(), key=lambda kv: (order[kv[0][0]], order[kv[0][1]]))
    return ", ".join(f"{_CLASS_LABEL[a]} → {_CLASS_LABEL[b]}: {n}" for (a, b), n in items) or "-"


def probe_false_positives(versions: list[CodeVersion]) -> list[CodeVersion]:
    """Passing code on which a probe deviates (probe stricter than the tests)."""
    return [cv for cv in versions if cv.failure_class == "pass" and cv.deviations]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    scorer: Scorer | None = None,
    inspector: Inspector | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--csv", type=Path, default=Path(f"failure_classes_{TASK}.csv"))
    ap.add_argument("--rescored", action="store_true",
                    help="use re-scored results from analysis/rescore_mdtable.csv")
    ap.add_argument("--analysis-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    _check_output_path(args.csv, _REPO_ROOT)

    logs = load_runs(args.runs_dir, TASK)
    if not logs:
        print(f"no run logs for task {TASK!r} in {args.runs_dir}", file=sys.stderr)
        return 1
    rescore = None
    if args.rescored:
        from rescored import ANALYSIS_DIR, require_rescore
        rescore = require_rescore([TASK], args.analysis_dir or ANALYSIS_DIR)
        header = rescore.header([l for _, l in logs])
        logs = [(p, rescore.apply(l)) for p, l in logs]
    scorer = scorer or make_executor_scorer()
    inspector = inspector or make_inspector()
    titles, cited = load_ambiguity_doc()
    missing = sorted({p.ambiguity for p in PROBES} - set(titles))
    if missing:
        print(f"warning: ambiguities {missing} not found in {TASK_SUMMARY}", file=sys.stderr)

    versions: list[CodeVersion] = []
    for i, (path, log) in enumerate(logs, 1):
        print(f"[{i}/{len(logs)}] {path.name}", file=sys.stderr)
        versions.extend(analyze_run(
            path, log, scorer, inspector,
            rebuild_final=rescore is not None and rescore.changed(log["run_id"]) is not None,
        ))
    sigs = build_signatures(versions)

    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(versions[0].csv_dict()))
        w.writeheader()
        for cv in versions:
            w.writerow(cv.csv_dict())

    n_seq = sum(l["architecture_name"] == SEQUENTIAL for _, l in logs)
    print(header if rescore else "Results as logged (no --rescored); implement step parsed "
          "with the current parser")
    print(f"Task {TASK}: {len(logs)} runs ({n_seq} sequential), "
          f"{len(versions)} code versions classified")
    print("Probes (ambiguity numbers = docs/task_summary.md section 4 table):")
    for p in PROBES:
        print(f"  #{p.ambiguity} {titles.get(p.ambiguity, '?'):<28} {p.name}")
    fps = probe_false_positives(versions)
    print(f"Probe false positives on passing code: {len(fps)}"
          + ("  <-- probes stricter than hidden tests: "
             + ", ".join(f"{cv.run_id[:8]}/{cv.stage}:{'+'.join(cv.deviations)}" for cv in fps)
             if fps else ""))
    errs = [cv for cv in versions if cv.probe_errors and cv.failure_class != "collection_error"]
    if errs:
        print(f"Code versions where a probe raised: {len(errs)} (see CSV probe_errors)")

    print_collection_errors(versions)
    print_signatures(sigs, titles, cited)
    print_summary(versions)
    print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

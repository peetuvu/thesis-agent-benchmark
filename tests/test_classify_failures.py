"""Tests for scripts/classify_failures.py using fake run logs and a real inspector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "classify_failures.py"


def _load():
    spec = importlib.util.spec_from_file_location("classify_failures", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cf = _load()

FILE = "mdtable/formatter.py"
HIDDEN = "eval/hidden_tests/mdtable/test_hidden.py::"

# A reference formatter that follows every hidden-test reading (kept tiny:
# no code-block handling, which the probes do not exercise).
GOOD = r'''
import re

def parse_table(lines):
    return None

def format_table(table):
    return []

def _cells(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]

def _fmt(block, minw):
    head, sep, *rows = [_cells(l) for l in block]
    n = len(head)
    rows = [(r + [""] * n)[:n] for r in rows]
    al = ["c" if a.startswith(":") and a.endswith(":") else "r" if a.endswith(":")
          else "l" for a in sep]
    w = [max([minw] + [len(r[i]) for r in [head] + rows]) for i in range(n)]
    def cell(t, i):
        if al[i] == "r":
            return t.rjust(w[i])
        if al[i] == "c":
            pad = w[i] - len(t); return " " * (pad // 2) + t + " " * (pad - pad // 2)
        return t.ljust(w[i])
    line = lambda r: "| " + " | ".join(cell(t, i) for i, t in enumerate(r)) + " |"
    def s(i):
        d = w[i] + 2
        return {"l": "-" * d if not sep[i].startswith(":") else ":" + "-" * (d - 1),
                "r": "-" * (d - 1) + ":", "c": ":" + "-" * (d - 2) + ":"}[al[i]]
    return [line(head), "|" + "|".join(s(i) for i in range(n)) + "|"] + [line(r) for r in rows]

MINW = 0

def format_markdown(text):
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+:?", lines[i + 1]):
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                j += 1
            out += _fmt(lines[i:j], MINW); i = j
        else:
            out.append(lines[i]); i += 1
    return "\n".join(out)
'''
MIN3 = GOOD.replace("MINW = 0", "MINW = 3")
SYNTAX = 'def format_markdown(text):\n    if text.startswith("'
# What a model wrote. The old parse_code_blocks cut it at the inline ``` into
# SYNTAX; the fixed parser keeps it whole.
FENCE_LITERAL = 'def format_markdown(text):\n    if text.startswith("```"):\n        return 1\n'
MISSING = "def format_markdown(text):\n    return text\n"


def test_reference_formatter_matches_every_probe() -> None:
    ns: dict[str, Any] = {}
    exec(GOOD, ns)
    for p in cf.PROBES:
        assert not cf.probe_deviates(p, ns["format_markdown"](p.text)), p.name


def test_min_width_deviates_only_on_ambiguity_3() -> None:
    ns: dict[str, Any] = {}
    exec(MIN3, ns)
    devs = {p.ambiguity for p in cf.PROBES if cf.probe_deviates(p, ns["format_markdown"](p.text))}
    assert devs == {3}


def test_blank_pattern_probe() -> None:
    p = next(p for p in cf.PROBES if p.ambiguity == 6)
    collapsed = p.expected.replace("\n\n\n", "\n\n")
    assert cf.probe_deviates(p, collapsed)
    assert not cf.probe_deviates(p, p.expected.replace("| A |", "| A   |"))


@pytest.mark.parametrize("content, cut", [
    ("x = 1\n", False), ("x = 1\n    ", False), ('if s.startswith("', True), ("", False),
])
def test_block_cut_mid_line(content: str, cut: bool) -> None:
    assert cf.block_cut_mid_line(content) is cut


def test_describe_import_error_kinds() -> None:
    kind, detail = cf.describe_import_error({
        "stmt": "from mdtable.formatter import parse_table, format_table",
        "type": "ImportError", "syntax": False, "name": "mdtable.formatter",
        "msg": "cannot import name 'format_table' from 'mdtable.formatter' (/x)",
    })
    assert kind == "missing name" and "'format_table'" in detail
    kind, detail = cf.describe_import_error({
        "stmt": "from mdtable import format_markdown", "type": "SyntaxError",
        "syntax": True, "msg": "invalid syntax", "file": "mdtable/formatter.py",
        "line": 3, "text": "def (",
    })
    assert kind == "SyntaxError" and "mdtable/formatter.py:3" in detail
    assert cf.describe_import_error(None) == ("", "")


def test_doc_table_parsed() -> None:
    titles, cited = cf.load_ambiguity_doc()
    assert {p.ambiguity for p in cf.PROBES} <= set(titles)
    assert 1 in cited["test_escaped_pipes_in_cells"]


# ---------------------------------------------------------------------------
# Real inspector (child process on a temp copy of the task)
# ---------------------------------------------------------------------------


def test_inspector_reports_syntax_and_missing_name_and_probes() -> None:
    inspect = cf.make_inspector()
    res = inspect({FILE: SYNTAX})
    assert cf.describe_import_error(res["import_error"])[0] == "SyntaxError"
    res = inspect({FILE: MISSING})
    kind, detail = cf.describe_import_error(res["import_error"])
    assert kind == "missing name" and "parse_table" in detail
    res = inspect({FILE: MIN3})
    assert res["import_error"] is None
    assert set(res["probes"]) == {p.name for p in cf.PROBES}


# ---------------------------------------------------------------------------
# End to end with fake logs, fake scorer, real inspector
# ---------------------------------------------------------------------------


def _block(content: str) -> str:
    return f"Code:\n\n```{FILE}\n{content}\n```\n"


def fake_scorer(task: str, files: dict[str, str]) -> tuple[int, int, list[str]]:
    src = files.get(FILE, "")
    if "MINW = 3" in src:
        return 6, 13, [HIDDEN + "test_single_column_table", HIDDEN + "test_x"]
    if "MINW = 0" in src or src.rstrip("\n") == FENCE_LITERAL.rstrip("\n"):
        return 13, 13, []
    if src and "return text" not in src:
        return 0, 0, ["hidden_eval_collection_error"]
    return 0, 13, [HIDDEN + "test_x"]


def write_run(runs: Path, run_id: str, *, arch: str, final: str,
              implement: str | None = None, model: str = "m1") -> None:
    p, t, failing = fake_scorer("mdtable", {FILE: final})
    log = {
        "run_id": run_id, "task_name": "mdtable", "model": model,
        "architecture_name": arch, "status": "completed", "prompt_regime": "minimal",
        "requested_reasoning_effort": "low",
        "test_results": {"hidden_passed": p, "hidden_total": t, "failing_test_names": failing},
        "artifacts_file": f"{run_id}.artifacts.json",
    }
    agent_log = []
    if implement is not None:
        agent_log = [{"role": "implement", "response": _block(implement)},
                     {"role": "review", "response": "Found a bug."},
                     {"role": "fix", "response": _block(final)}]
    art = {"run_id": run_id, "agent_log": agent_log, "modified_files": {FILE: final}}
    (runs / f"{run_id}.json").write_text(json.dumps(log))
    (runs / f"{run_id}.artifacts.json").write_text(json.dumps(art))


def test_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "r1", arch="sequential", implement=MIN3, final=GOOD)
    write_run(runs, "r2", arch="sequential", implement=FENCE_LITERAL, final=GOOD)
    write_run(runs, "r3", arch="single_agent", final=SYNTAX)
    write_run(runs, "r4", arch="single_agent", final=MIN3)
    out_csv = tmp_path / "out.csv"
    assert cf.main(["--runs-dir", str(runs), "--csv", str(out_csv)], scorer=fake_scorer) == 0
    out = capsys.readouterr().out
    assert "interpretation → pass: 1" in out
    assert "pass → pass: 1" in out  # r2: implement block no longer cut
    assert "#3 Minimum column width (2/2)" in out
    assert "Probe false positives on passing code: 0" in out
    assert "harness artifact): 1/1" in out  # r3's logged (old-parser) file
    assert out_csv.read_text().count("\n") == 1 + 6  # header + 4 final + 2 implement


def test_main_refuses_csv_in_runs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cf.main(["--csv", str(_REPO / "runs" / "x.csv")])

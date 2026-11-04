"""Tests for scripts/run_batch.py flag passthrough (no bench subprocess is run)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_batch.py"


def _load_run_batch():
    spec = importlib.util.spec_from_file_location("run_batch", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str]) -> list[list[str]]:
    """Run run_batch.main with a fake subprocess.run; return the bench commands."""
    rb = _load_run_batch()
    monkeypatch.setattr(rb, "RUNS_DIR", tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="Status: success\nHidden: 1/1\n", stderr="")

    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    argv = ["run_batch", "--tasks", "snake", "--models", "opus-5.5",
            "--regimes", "minimal", "--reps", "1", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert rb.main() == 0
    return calls


def test_flags_passed_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _run(monkeypatch, tmp_path, ["--max-output-tokens", "16000", "--max-wall-clock", "900"])
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[cmd.index("--max-output-tokens") + 1] == "16000"
    assert cmd[cmd.index("--max-wall-clock") + 1] == "900.0"


def test_flags_omitted_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without the flags, bench's own defaults apply (nothing is forwarded)."""
    cmd = _run(monkeypatch, tmp_path, [])[0]
    assert "--max-output-tokens" not in cmd
    assert "--max-wall-clock" not in cmd

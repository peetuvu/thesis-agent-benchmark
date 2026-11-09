"""Tests for the handoff setup: pytest as a main dependency, check_env, and bench run --model."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from providers.models import MODELS
from runner import cli

_REPO = Path(__file__).resolve().parent.parent
_CHECK_ENV = _REPO / "scripts" / "check_env.py"


class TestPytestIsMainDependency:
    """A plain `uv sync --locked` must install what every run needs for scoring."""

    def test_pyproject_lists_pytest_in_main_dependencies_only(self) -> None:
        project = tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]
        assert any(d.startswith("pytest>=") for d in project["dependencies"])
        assert not any(
            d.startswith("pytest>=") or d == "pytest"
            for d in project["optional-dependencies"]["dev"]
        )

    def test_lockfile_lists_pytest_in_main_dependencies_only(self) -> None:
        packages = tomllib.loads((_REPO / "uv.lock").read_text())["package"]
        project = next(p for p in packages if p["name"] == "thesis-agents")
        assert "pytest" in {d["name"] for d in project["dependencies"]}
        assert "pytest" not in {d["name"] for d in project["optional-dependencies"]["dev"]}


@pytest.fixture
def check_env(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """check_env loaded with no provider keys set and the bench CLI present."""
    spec = importlib.util.spec_from_file_location("check_env", _CHECK_ENV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs load_dotenv, so keys are cleared afterwards
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/fake/bench")
    return module


class TestCheckEnv:
    def test_no_keys_fails(self, check_env: ModuleType, capsys: pytest.CaptureFixture) -> None:
        assert check_env.main() == 1
        assert "NOT ready" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "key,provider",
        [("ANTHROPIC_API_KEY", "anthropic"), ("OPENAI_API_KEY", "openai")],
    )
    def test_one_key_passes_and_names_its_provider(
        self,
        check_env: ModuleType,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        key: str,
        provider: str,
    ) -> None:
        monkeypatch.setenv(key, "test-key")
        assert check_env.main() == 0
        assert f"Environment ready for providers: {provider}\n" in capsys.readouterr().out

    def test_both_keys_report_both_providers(
        self,
        check_env: ModuleType,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        assert check_env.main() == 0
        assert "Environment ready for providers: anthropic, openai\n" in capsys.readouterr().out

    def test_key_without_bench_cli_fails(
        self, check_env: ModuleType, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr(check_env.shutil, "which", lambda _name: None)
        assert check_env.main() == 1

    def test_no_google_reference(self) -> None:
        assert "google" not in _CHECK_ENV.read_text().lower()


class TestBenchRunModel:
    BASE = ["bench", "run", "--task", "snake", "--arch", "single_agent"]

    def _main(self, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        """Run the CLI with the executor mocked; return the exit code."""
        monkeypatch.setattr(sys, "argv", [*self.BASE, *extra])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        return exc.value.code

    def test_model_is_required(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    ) -> None:
        with patch.object(cli, "Executor") as executor:
            assert self._main(monkeypatch) == 2
        executor.assert_not_called()
        assert "--model" in capsys.readouterr().err

    @pytest.mark.parametrize("model", ["opus-5.6", "claude-opus-5-5"])
    def test_unknown_alias_rejected_before_any_run(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, model: str,
    ) -> None:
        with (
            patch.object(cli, "Executor") as executor,
            patch.object(cli, "_load_architecture") as load,
        ):
            assert self._main(monkeypatch, "--model", model) == 2
        executor.assert_not_called()
        load.assert_not_called()
        err = capsys.readouterr().err
        assert f"Unknown model alias '{model}'" in err
        for alias in MODELS:
            assert alias in err

    def test_known_alias_reaches_the_executor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys, "argv", [*self.BASE, "--model", "haiku-5.5", "--reasoning-effort", "low"],
        )

        class Stop(Exception):
            """Ends main() once the executor has been reached."""

        with patch.object(cli, "Executor", side_effect=Stop) as executor:
            with pytest.raises(Stop):
                cli.main()
        executor.assert_called_once()

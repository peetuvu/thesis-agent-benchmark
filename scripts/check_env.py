#!/usr/bin/env python3
"""Validate the environment before benchmark runs.

Checks Python version, the `bench` CLI, provider API keys, and core package
imports. Prints a clear pass/fail per check and a summary of usable providers.

Usage: python3 scripts/check_env.py

Exit code 0 if at least ANTHROPIC_API_KEY is set and the bench CLI is
available; exit code 1 otherwise. API key values are never printed in full.
"""
import importlib
import os
import shutil
import sys

MIN_PYTHON = (3, 11)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def mask_key(value: str) -> str:
    """Return a masked preview of a secret: first 12 chars + '...' + length."""
    prefix = value[:12]
    return f"{prefix}..., length {len(value)}"


def check_python() -> bool:
    """Check the running interpreter is Python 3.11+."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        ok(f"Python {version_str}")
        return True
    fail(f"Python {version_str} — need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")
    return False


def check_bench_cli() -> bool:
    """Check the `bench` CLI command is on PATH."""
    path = shutil.which("bench")
    if path:
        ok(f"bench CLI found ({path})")
        return True
    fail("bench CLI not found on PATH")
    return False


def check_api_key(name: str, provider: str, required: bool) -> bool:
    """Check an API key env var. Returns True if the provider is usable."""
    value = os.environ.get(name)
    if value:
        ok(f"{name} set ({mask_key(value)})")
        return True
    if required:
        fail(f"{name} not set — {provider} provider unavailable")
    else:
        warn(f"{name} not set — {provider} provider unavailable")
    return False


def check_import(package: str, required: bool) -> bool:
    """Actually import a core package, surfacing import-time errors loudly.

    A real import (rather than find_spec) catches corrupt or partially
    installed packages that resolve a spec but fail on import — exactly the
    kind of problem this pre-run check exists to expose.
    """
    try:
        importlib.import_module(package)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        report = fail if required else warn
        report(f"{package} package not importable — {exc}")
        return False
    ok(f"{package} package importable")
    return True


def main() -> int:
    print("Environment check for thesis benchmark harness")
    print()

    python_ok = check_python()
    bench_ok = check_bench_cli()

    anthropic_key = check_api_key("ANTHROPIC_API_KEY", "anthropic", required=True)
    openai_key = check_api_key("OPENAI_API_KEY", "openai", required=False)
    google_key = check_api_key("GOOGLE_API_KEY", "google", required=False)

    check_import("anthropic", required=True)
    check_import("openai", required=False)
    check_import("pydantic", required=True)

    providers = []
    if anthropic_key:
        providers.append("anthropic")
    if openai_key:
        providers.append("openai")
    if google_key:
        providers.append("google")

    print()
    if providers:
        print(f"Environment ready for providers: {', '.join(providers)}")
    else:
        print("Environment NOT ready — no provider API keys set")

    if python_ok and bench_ok and anthropic_key:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

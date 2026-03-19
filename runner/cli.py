"""CLI entry point for the benchmark harness."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from inspect import isclass

from architectures.base import Architecture
from providers.models import list_models, resolve_model
from runner.config import RunConfig
from runner.executor import Executor


def _load_architecture(name: str) -> Architecture:
    """Dynamically import and instantiate an Architecture subclass by module name.

    Looks in architectures/<name>.py for a class that subclasses Architecture.
    """
    try:
        module = importlib.import_module(f"architectures.{name}")
    except ModuleNotFoundError:
        print(f"Error: architecture module 'architectures.{name}' not found.", file=sys.stderr)
        print("Available architecture modules should be .py files in architectures/.", file=sys.stderr)
        sys.exit(1)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isclass(attr)
            and issubclass(attr, Architecture)
            and attr is not Architecture
        ):
            return attr()

    print(
        f"Error: no Architecture subclass found in architectures.{name}",
        file=sys.stderr,
    )
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Benchmark harness for multi-agent LLM architectures",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Execute a benchmark run")
    run_parser.add_argument("--task", required=True, help="Task name (e.g. snake)")
    run_parser.add_argument("--arch", required=True, help="Architecture module name (e.g. single_agent)")
    run_parser.add_argument("--model", default="sonnet-4.6", help="Model alias or raw API string (default: sonnet-4.6)")
    run_parser.add_argument("--provider", default=None, help="API provider (auto-detected from model registry if omitted)")
    run_parser.add_argument("--prompt-regime", default="minimal", help="Prompt regime (minimal|production)")
    run_parser.add_argument("--max-tokens", type=int, default=100_000, help="Total token budget")
    run_parser.add_argument("--max-tool-calls", type=int, default=10, help="Max tool/API calls")
    run_parser.add_argument("--max-wall-clock", type=float, default=300.0, help="Wall-clock timeout (seconds)")
    run_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    run_parser.add_argument("--thinking", action="store_true", help="Enable extended thinking / reasoning mode")

    sub.add_parser("models", help="List available models")

    return parser


def main() -> None:
    """Entry point for the `bench` CLI command."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "models":
        print(list_models())
        sys.exit(0)

    if args.command == "run":
        # Resolve model alias -> (api_string, auto_provider)
        api_string, auto_provider = resolve_model(args.model)

        # Provider: explicit flag > auto-detected from registry > fallback to anthropic
        provider = args.provider or auto_provider or "anthropic"

        architecture = _load_architecture(args.arch)

        config = RunConfig(
            task_name=args.task,
            architecture_name=args.arch,
            model=api_string,
            provider=provider,
            prompt_regime=args.prompt_regime,
            max_tokens=args.max_tokens,
            max_tool_calls=args.max_tool_calls,
            max_wall_clock_seconds=args.max_wall_clock,
            seed=args.seed,
            thinking_enabled=args.thinking,
        )

        executor = Executor()
        log = asyncio.run(executor.run(config, architecture))

        # Print summary
        from runner.logger import make_run_filename

        tr = log.test_results
        print(f"\n{'='*50}")
        print(f"Run:    {log.run_id}")
        print(f"Status: {log.status}")
        print(f"Hidden: {tr.hidden_passed}/{tr.hidden_total} passed")
        print(f"Public: {tr.public_passed}/{tr.public_total} passed")
        print(f"Policy: {'compliant' if log.policy_compliant else 'VIOLATION'}")
        print(f"Log:    runs/{make_run_filename(log)}.json")
        print(f"{'='*50}")

        if log.status not in ("success",):
            sys.exit(1)


if __name__ == "__main__":
    main()

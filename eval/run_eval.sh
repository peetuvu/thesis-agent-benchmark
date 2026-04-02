#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: run_eval.sh <task_name> [source_dir]}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DIR="${2:-$ROOT/agent-bench/tasks/$TASK}"
TEST_DIR="$ROOT/eval/hidden_tests/$TASK"

if [ ! -d "$TEST_DIR" ]; then
  echo "ERROR: no hidden tests found at $TEST_DIR"
  exit 1
fi

# Resolve Python: prefer venv, fall back to system python3
if [ -f "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi

# Ensure Python can import: task package (from agent task dir) and eval (from repo root)
export PYTHONPATH="$AGENT_DIR:$ROOT"
# Tell conftest.py to resolve imports from the custom source dir
export BENCH_SOURCE_DIR="$AGENT_DIR"

# Run ONLY hidden tests.  --rootdir keeps the rootdir inside $TEST_DIR so
# no parent pytest.ini (e.g. pyproject.toml testpaths) interferes.
# -p no:cacheprovider avoids permission errors when rootdir is read-only.
echo "Running hidden evaluation for task: $TASK"
"$PYTHON" -m pytest -v --tb=line --rootdir="$TEST_DIR" -p no:cacheprovider \
  --override-ini="addopts=" "$TEST_DIR"

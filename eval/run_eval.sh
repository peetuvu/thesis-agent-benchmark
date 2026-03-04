#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: run_eval.sh <task_name>}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DIR="$ROOT/agent-bench/tasks/$TASK"
TEST_DIR="$ROOT/eval/hidden_tests/$TASK"

if [ ! -d "$TEST_DIR" ]; then
  echo "ERROR: no hidden tests found at $TEST_DIR"
  exit 1
fi

# Activate venv (adjust if yours differs)
if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$ROOT/.venv/bin/activate"
fi

# Ensure Python can import: snake (from agent task dir) and eval (from repo root)
export PYTHONPATH="$AGENT_DIR:$ROOT"

# Run ONLY hidden tests and ignore any pytest.ini that might cause recursive collection
echo "Running hidden evaluation for task: $TASK"
pytest -q -c /dev/null "$TEST_DIR"

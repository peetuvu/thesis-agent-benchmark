# Hidden evaluation harness configuration.
# Agents must NOT modify anything under eval/.

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # thesis-agents/

# Allow the runner to override the source dir (e.g. pointing to a temp copy).
# Falls back to the real agent-bench location for manual runs.
AGENT_BENCH = Path(
    os.environ.get("BENCH_SOURCE_DIR", str(ROOT / "agent-bench" / "tasks" / "mdtable"))
)

# Ensure imports resolve to the agent-bench package
if str(AGENT_BENCH) not in sys.path:
    sys.path.insert(0, str(AGENT_BENCH))

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # thesis-agents/

AGENT_BENCH = Path(
    os.environ.get("BENCH_SOURCE_DIR", str(ROOT / "agent-bench" / "tasks" / "calculator"))
)

if str(AGENT_BENCH) not in sys.path:
    sys.path.insert(0, str(AGENT_BENCH))

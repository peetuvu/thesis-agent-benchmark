# Hidden evaluation harness configuration.
# Agents must NOT modify anything under eval/.

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # thesis-agents/
AGENT_BENCH = ROOT / "agent-bench" / "tasks" / "snake"

# Ensure imports resolve to the agent-bench package
if str(AGENT_BENCH) not in sys.path:
    sys.path.insert(0, str(AGENT_BENCH))

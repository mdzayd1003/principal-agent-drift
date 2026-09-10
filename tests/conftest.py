import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drift.llm import SimulatedAgent  # noqa: E402
from drift.negotiation import run_grid  # noqa: E402
from drift.scenarios import episode_grid  # noqa: E402


@pytest.fixture(scope="session")
def episodes():
    return run_grid(SimulatedAgent(), episode_grid())


@pytest.fixture(scope="session")
def episodes_by_id(episodes):
    return {e.id: e for e in episodes}

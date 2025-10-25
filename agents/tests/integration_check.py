"""Curated integration checks for agent compatibility.

This module runs a lightweight subset of pytest scenarios that exercise
critical agent pathways (standalone evaluation, persistence checkpoints,
and multi-agent coordination). It is designed to be invoked via:

    python -m agents.tests.integration_check

so contributors can quickly confirm that shared agent changes preserve
cross-module behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

TEST_TARGETS: List[str] = [
    "tests/test_eea_agent.py::test_stage_regularization_enforces_ratio_bounds",
    "tests/test_persistence_cycle.py::test_persist_agent_checkpoint_cycle",
    "tests/test_multiagent_env.py::TestMultiAgentEnv::test_step_functionality",
]


def main() -> int:
    """Execute the curated pytest scenarios and propagate their exit code."""

    repo_root = Path(__file__).resolve().parents[2]
    test_args = [str(repo_root / target) for target in TEST_TARGETS]

    print("Running agent integration checks via pytest:")
    for target in TEST_TARGETS:
        print(f"  - {target}")

    return pytest.main(test_args)


if __name__ == "__main__":
    sys.exit(main())

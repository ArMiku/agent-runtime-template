"""Shared pytest fixtures for the agent-runtime test suite."""

from __future__ import annotations

import pytest

from agent_runtime.foundation.log import configure_logging


@pytest.fixture(scope="session", autouse=True)
def _configure_logging() -> None:
    # Quiet, deterministic logs during tests.
    configure_logging("WARNING")

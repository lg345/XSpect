"""
Shared pytest fixtures and placeholder step registrations.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from XSpect.analysis.registry import (
    register_step, register_reduction, clear_registry,
    _STEP_REGISTRY, _REDUCTION_REGISTRY,
)


def _register_placeholder_steps():
    """Register placeholder steps for testing. Idempotent."""
    if 'placeholder_add' not in _STEP_REGISTRY:
        @register_step("placeholder_add")
        def placeholder_add(run, on=None, value=0, **kwargs):
            key = on or "data"
            current = run.results.get(key, 0)
            run.results[f"{key}.added"] = current + value

    if 'placeholder_multiply' not in _STEP_REGISTRY:
        @register_step("placeholder_multiply")
        def placeholder_multiply(run, on=None, value=1, **kwargs):
            key = on or "data"
            current = run.results.get(key, 0)
            run.results[f"{key}.multiplied"] = current * value

    if 'placeholder_sum' not in _REDUCTION_REGISTRY:
        @register_reduction("placeholder_sum")
        def placeholder_sum(runs, on=None, **kwargs):
            total = sum(r.results.get(on, 0) for r in runs)
            return {"summed": total}


@pytest.fixture(autouse=True)
def ensure_placeholder_steps():
    """Ensure placeholder steps are registered for every test."""
    _register_placeholder_steps()
    yield


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def mock_run():
    """Create a minimal mock run object for unit testing."""

    class MockExperiment:
        def __init__(self):
            self.lcls_run = 1
            self.hutch = "test"
            self.experiment_id = "test00000"
            self.experiment_directory = "/tmp/xspect_test"

    class MockRun:
        def __init__(self):
            self.spec_experiment = MockExperiment()
            self.run_number = 1
            self.run_file = None
            self.status = []
            self.status_datetime = []
            self.verbose = False
            self.end_index = -1
            self.start_index = 0
            self.results = {}

        def update_status(self, message):
            from datetime import datetime
            self.status.append(message)
            self.status_datetime.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return MockRun()

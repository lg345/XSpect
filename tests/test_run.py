"""Tests for XSpect.model.run"""

import pytest
from XSpect.model.run import spectroscopy_run


class MockExperiment:
    def __init__(self):
        self.lcls_run = 1
        self.hutch = "test"
        self.experiment_id = "test00000"
        self.experiment_directory = "/tmp/xspect_test"


def _make_run():
    exp = MockExperiment()
    run = spectroscopy_run.__new__(spectroscopy_run)
    run.spec_experiment = exp
    run.run_number = 42
    run.run_file = "/tmp/xspect_test/test00000_Run0042.h5"
    run.status = ["init"]
    run.status_datetime = ["2026-01-01 00:00:00"]
    run.verbose = False
    run.end_index = -1
    run.start_index = 0
    run.results = {}
    return run


def test_run_has_results_dict():
    run = _make_run()
    assert hasattr(run, 'results')
    assert isinstance(run.results, dict)
    assert run.results == {}


def test_run_results_writable():
    run = _make_run()
    run.results["epix_ROI_1.simultaneous_laser"] = [1, 2, 3]
    assert run.results["epix_ROI_1.simultaneous_laser"] == [1, 2, 3]


def test_run_results_dot_separated_keys():
    run = _make_run()
    run.results["epix.added"] = 10
    run.results["epix.multiplied"] = 20
    assert "epix.added" in run.results
    assert "epix.multiplied" in run.results


def test_run_status_logging():
    run = _make_run()
    initial_len = len(run.status)
    run.update_status("test message")
    assert len(run.status) == initial_len + 1
    assert run.status[-1] == "test message"
    assert len(run.status_datetime) == len(run.status)


def test_run_preserves_existing_attrs():
    run = _make_run()
    assert run.run_number == 42
    assert run.spec_experiment.hutch == "test"
    assert run.verbose is False
    assert run.start_index == 0
    assert run.end_index == -1

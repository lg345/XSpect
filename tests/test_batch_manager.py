"""Tests for XSpect.controller.batch_manager"""

import pytest
from XSpect.controller.batch_manager import (
    split_into_batches, reconverge_results, run_batched,
)
from XSpect.controller.config_parser import StepConfig


def test_split_shots_even():
    batches = split_into_batches(10000, 2000)
    assert len(batches) == 5
    assert batches[0] == (0, 2000)
    assert batches[4] == (8000, 10000)


def test_split_shots_remainder():
    batches = split_into_batches(10001, 2000)
    assert len(batches) == 6
    assert batches[5] == (10000, 10001)


def test_split_shots_smaller_than_batch():
    batches = split_into_batches(500, 2000)
    assert len(batches) == 1
    assert batches[0] == (0, 500)


def test_split_shots_zero():
    batches = split_into_batches(0, 2000)
    assert batches == []


def test_split_shots_exact_multiple():
    batches = split_into_batches(6000, 2000)
    assert len(batches) == 3
    for i, (start, end) in enumerate(batches):
        assert start == i * 2000
        assert end == (i + 1) * 2000


def test_split_shots_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        split_into_batches(100, 0)


def test_reconverge_single_batch():
    results = reconverge_results([{"a": 1, "b": 2}])
    assert results == {"a": 1, "b": 2}


def test_reconverge_multiple_batches():
    batch_results = [
        {"a": 10, "b": 20},
        {"a": 30, "b": 40},
    ]
    merged = reconverge_results(batch_results)
    assert merged["a"] == [10, 30]
    assert merged["b"] == [20, 40]


def test_reconverge_empty():
    assert reconverge_results([]) == {}


def test_reconverge_partial_keys():
    batch_results = [
        {"a": 1, "b": 2},
        {"a": 3},
    ]
    merged = reconverge_results(batch_results)
    assert merged["a"] == [1, 3]
    assert merged["b"] == 2


def test_run_batched_single_core(mock_run):
    mock_run.total_shots = 100
    steps = [StepConfig(step="placeholder_add", args={"on": "test", "value": 5})]
    run_batched(mock_run, steps, cores=1, batch_size=50)
    assert "test.added" in mock_run.results


def test_run_batched_no_total_shots(mock_run):
    steps = [StepConfig(step="placeholder_add", args={"on": "test", "value": 3})]
    run_batched(mock_run, steps, cores=1, batch_size=50)
    assert "test.added" in mock_run.results

"""
Batch manager for parallel pipeline execution.

Splits runs into shot-range batches, optionally parallelizes via
multiprocessing.Pool, and reconverges batch results.
"""

import copy
from multiprocessing import Pool
from functools import partial

from XSpect.controller.config_parser import StepConfig
from XSpect.controller.pipeline_runner import run_pipeline


def split_into_batches(total_shots: int, batch_size: int) -> list[tuple[int, int]]:
    """
    Split a range of shots into contiguous batches.

    Parameters
    ----------
    total_shots : int
        Total number of shots in the run.
    batch_size : int
        Maximum shots per batch.

    Returns
    -------
    list of (start, end) tuples
        Each tuple defines a half-open range [start, end).
    """
    if total_shots <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batches = []
    start = 0
    while start < total_shots:
        end = min(start + batch_size, total_shots)
        batches.append((start, end))
        start = end
    return batches


def _process_batch(batch_range, run, steps):
    """
    Process a single batch: create a sub-run view and execute pipeline steps.

    This function is designed to be called via Pool.map. It creates a shallow
    copy of the run with adjusted start/end indices, runs the pipeline, and
    returns the results dict.
    """
    start, end = batch_range
    batch_run = copy.copy(run)
    batch_run.start_index = start
    batch_run.end_index = end
    batch_run.results = {}
    batch_run.status = []
    batch_run.status_datetime = []

    run_pipeline(batch_run, steps)
    return batch_run.results


def reconverge_results(batch_results: list[dict]) -> dict:
    """
    Merge results from multiple batches into a single results dict.

    For now, uses a simple strategy: collect all batch results into lists
    keyed by result name. Downstream steps or the pipeline itself decide
    how to combine them (sum, concatenate, etc).

    Parameters
    ----------
    batch_results : list[dict]
        List of results dicts from each batch.

    Returns
    -------
    dict
        Merged results. If a key appears in multiple batches, the values
        are collected into a list.
    """
    if not batch_results:
        return {}

    if len(batch_results) == 1:
        return batch_results[0]

    merged = {}
    all_keys = set()
    for br in batch_results:
        all_keys.update(br.keys())

    for key in all_keys:
        values = [br[key] for br in batch_results if key in br]
        if len(values) == 1:
            merged[key] = values[0]
        else:
            merged[key] = values

    return merged


def run_batched(run, pipeline_steps: list[StepConfig], cores: int = 1, batch_size: int = 2000) -> None:
    """
    Execute pipeline steps on a run with optional batch parallelism.

    If cores == 1, runs sequentially without spawning a Pool.
    If cores > 1, splits into batches and uses multiprocessing.

    Parameters
    ----------
    run : spectroscopy_run
        The run object. Must have a total_shots attribute or equivalent.
    pipeline_steps : list[StepConfig]
        Steps to execute on each batch.
    cores : int
        Number of worker processes.
    batch_size : int
        Shots per batch.
    """
    total_shots = getattr(run, 'total_shots', None)
    if total_shots is None:
        run_pipeline(run, pipeline_steps)
        return

    batches = split_into_batches(total_shots, batch_size)

    if not batches:
        return

    if cores <= 1 or len(batches) == 1:
        batch_results = []
        for batch_range in batches:
            result = _process_batch(batch_range, run, pipeline_steps)
            batch_results.append(result)
    else:
        process_fn = partial(_process_batch, run=run, steps=pipeline_steps)
        with Pool(processes=cores) as pool:
            batch_results = pool.map(process_fn, batches)

    run.results = reconverge_results(batch_results)

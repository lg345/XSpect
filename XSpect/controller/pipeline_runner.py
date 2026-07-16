"""
Pipeline step dispatch loop.

Iterates over a list of StepConfig entries, looks up each step in the registry,
and calls it with the run object and the step's YAML arguments.
"""

import logging

from XSpect.analysis.registry import get_step, get_reduction
from XSpect.controller.config_parser import StepConfig

logger = logging.getLogger("XSpect")


def run_pipeline(run, steps: list[StepConfig]) -> None:
    """
    Execute a sequence of pipeline steps on a run object.

    Each step mutates run.results in place.

    Parameters
    ----------
    run : spectroscopy_run
        The run object whose .results dict will be populated.
    steps : list[StepConfig]
        Ordered list of steps to execute.
    """
    for i, step_config in enumerate(steps):
        step_fn = get_step(step_config.step)
        run.update_status(f"Running step: {step_config.step}")
        logger.debug(
            "step %d/%d: %s  args=%s",
            i + 1,
            len(steps),
            step_config.step,
            step_config.args,
        )
        try:
            step_fn(run, **step_config.args)
        except Exception:
            logger.exception(
                "step %d/%d '%s' FAILED", i + 1, len(steps), step_config.step
            )
            raise
        run.update_status(f"Completed step: {step_config.step}")


def run_reductions(runs: list, reduction_steps: list[StepConfig]) -> dict:
    """
    Execute reduction steps across all analyzed runs.

    Parameters
    ----------
    runs : list[spectroscopy_run]
        All analyzed run objects.
    reduction_steps : list[StepConfig]
        Ordered list of reduction operations.

    Returns
    -------
    dict
        Aggregated reduction results.
    """
    results = {}
    for step_config in reduction_steps:
        reduction_fn = get_reduction(step_config.step)
        result = reduction_fn(runs, **step_config.args)
        if isinstance(result, dict):
            results.update(result)
    return results

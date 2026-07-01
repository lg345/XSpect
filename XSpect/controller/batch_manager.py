"""
Batch manager for parallel pipeline execution.

Splits runs into shot-range batches, optionally parallelizes via
multiprocessing.Pool, and reconverges batch results.
"""

import copy
from multiprocessing import Pool
from functools import partial

import numpy as np

from XSpect.controller.config_parser import StepConfig
from XSpect.controller.pipeline_runner import run_pipeline


# Attributes that are part of the run's infrastructure (not step outputs)
_INFRASTRUCTURE_ATTRS = frozenset(
    {
        "spec_experiment",
        "run_number",
        "run_file",
        "status",
        "status_datetime",
        "verbose",
        "end_index",
        "start_index",
        "abs_start_index",
        "abs_end_index",
        "results",
        "total_shots",
        "run_shots",
        "xray",
        "laser",
        "simultaneous",
        "h5",
    }
)


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


def _collect_batch_attrs(batch_run, pre_attrs):
    """Collect attributes created by pipeline steps (not present before pipeline ran)."""
    new_attrs = {}
    for attr in vars(batch_run):
        if attr in _INFRASTRUCTURE_ATTRS or attr in pre_attrs:
            continue
        val = getattr(batch_run, attr)
        if val is not None:
            new_attrs[attr] = val
    return new_attrs


def _slice_arrays_for_batch(batch_run, start, end):
    """Slice multi-shot arrays on a batch run to the batch's shot range."""
    total = getattr(batch_run, "total_shots", None)
    if total is None:
        return
    for attr in list(vars(batch_run)):
        if attr in _INFRASTRUCTURE_ATTRS:
            continue
        val = getattr(batch_run, attr)
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == total:
            setattr(batch_run, attr, val[start:end].copy())


def _derive_shot_masks(batch_run):
    """Derive xray/laser/simultaneous masks if lightStatus keys were loaded."""
    xray = getattr(batch_run, "xray", None)
    laser = getattr(batch_run, "laser", None)
    if xray is not None and laser is not None:
        batch_run.xray = xray.astype(bool)
        batch_run.laser = laser.astype(bool)
        batch_run.simultaneous = np.logical_and(batch_run.xray, batch_run.laser)


def _slice_shot_masks(batch_run, start, end, parent_run):
    """Slice shot masks and scalar keys from the parent run to the batch range."""
    parent_total = getattr(parent_run, "total_shots", None)
    if parent_total is None:
        return
    for attr in list(vars(batch_run)):
        if attr in _INFRASTRUCTURE_ATTRS:
            continue
        val = getattr(batch_run, attr, None)
        if (
            isinstance(val, np.ndarray)
            and val.ndim >= 1
            and val.shape[0] == parent_total
        ):
            setattr(batch_run, attr, val[start:end].copy())
    for attr in ("xray", "laser", "simultaneous"):
        parent_val = getattr(parent_run, attr, None)
        if parent_val is not None and isinstance(parent_val, np.ndarray):
            setattr(batch_run, attr, parent_val[start:end].copy())


def _load_batch_from_hdf5(
    batch_run, run_file, abs_start, abs_end, detector_configs, scalar_keys
):
    """Load batch data slice directly from HDF5."""
    import h5py

    with h5py.File(run_file, "r") as fh:
        for key, name in scalar_keys:
            try:
                setattr(batch_run, name, np.array(fh[key][abs_start:abs_end]))
            except KeyError:
                pass

        for hdf5_path, name, transpose in detector_configs:
            try:
                data = np.array(fh[hdf5_path][abs_start:abs_end, :, :])
                if transpose:
                    data = np.transpose(data, axes=(0, 2, 1))
                setattr(batch_run, name, data)
            except KeyError:
                pass


def _make_batch_run(run, start, end):
    """Create a lightweight batch run object."""
    batch_run = copy.copy(run)
    parent_start = getattr(run, "start_index", 0)
    batch_run.start_index = 0
    batch_run.end_index = end - start
    batch_run.total_shots = end - start
    # Absolute HDF5 indices: used by steps that read directly from the source
    # file (e.g. droplet_reconstruction) and need to know which rows to access.
    batch_run.abs_start_index = parent_start + start
    batch_run.abs_end_index = parent_start + end
    batch_run.results = {}
    batch_run.status = []
    batch_run.status_datetime = []
    return batch_run


def _inject_precomputed(batch_run, precomputed_attrs, start, end, total_shots):
    """Inject precomputed scalar/axis attributes into a batch run.

    Static attributes (e.g. ccm_bins, ccm_energies, time_bins) are copied
    directly.  Per-shot arrays (shape[0] == total_shots) are sliced to the
    batch range so that binning steps (ccm_binning, time_binning) receive the
    correct per-batch indices rather than recomputing them from incomplete data.
    """
    if not precomputed_attrs:
        return
    for attr, value in precomputed_attrs.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == total_shots
        ):
            setattr(batch_run, attr, value[start:end].copy())
        else:
            setattr(batch_run, attr, value)


def _process_batch_sequential(
    batch_range,
    run,
    steps,
    detector_configs=None,
    scalar_keys=None,
    precomputed_attrs=None,
):
    """
    Process a single batch sequentially (in-process).

    If detector data is already loaded on the run (total_shots matches array
    sizes), slices from memory. Otherwise loads from HDF5.
    """
    start, end = batch_range
    batch_run = _make_batch_run(run, start, end)

    run_file = getattr(run, "run_file", None)
    has_preloaded = any(
        isinstance(getattr(run, attr, None), np.ndarray)
        and getattr(run, attr).ndim >= 2
        for attr in vars(run)
        if attr not in _INFRASTRUCTURE_ATTRS
    )

    if has_preloaded:
        _slice_arrays_for_batch(batch_run, start, end)
    elif run_file and detector_configs:
        abs_start = getattr(run, "start_index", 0) + start
        abs_end = getattr(run, "start_index", 0) + end
        _load_batch_from_hdf5(
            batch_run, run_file, abs_start, abs_end, detector_configs, scalar_keys or []
        )

    _slice_shot_masks(batch_run, start, end, run)

    parent_total = getattr(run, "total_shots", None)
    if precomputed_attrs and parent_total is not None:
        _inject_precomputed(batch_run, precomputed_attrs, start, end, parent_total)

    pre_attrs = set(vars(batch_run).keys())
    run_pipeline(batch_run, steps)
    return _collect_batch_attrs(batch_run, pre_attrs)


def _process_batch_parallel(
    batch_range,
    run_file,
    start_index,
    detector_configs,
    scalar_keys,
    steps,
    precomputed_attrs=None,
    parent_total_shots=None,
):
    """
    Process a single batch in a worker process by reloading from HDF5.
    Avoids pickling large arrays across processes.

    precomputed_attrs, if provided, are injected before the pipeline runs so
    that axis steps (make_ccm_axis, time_binning) see a globally consistent
    axis and skip re-derivation.
    """
    from XSpect.model.run import spectroscopy_run

    start, end = batch_range
    abs_start = start_index + start
    abs_end = start_index + end

    batch_run = spectroscopy_run.__new__(spectroscopy_run)
    batch_run.run_file = run_file
    batch_run.run_number = 0
    batch_run.start_index = 0
    batch_run.end_index = end - start
    batch_run.abs_start_index = abs_start
    batch_run.abs_end_index = abs_end
    batch_run.results = {}
    batch_run.status = []
    batch_run.status_datetime = []
    batch_run.verbose = False
    batch_run.total_shots = end - start

    _load_batch_from_hdf5(
        batch_run, run_file, abs_start, abs_end, detector_configs, scalar_keys
    )

    _derive_shot_masks(batch_run)

    if precomputed_attrs and parent_total_shots is not None:
        _inject_precomputed(
            batch_run, precomputed_attrs, start, end, parent_total_shots
        )

    pre_attrs = set(vars(batch_run).keys())
    run_pipeline(batch_run, steps)
    return _collect_batch_attrs(batch_run, pre_attrs)


def reconverge_results(batch_results: list[dict]) -> dict:
    """
    Merge results from multiple batches by summing numeric arrays.

    For numpy arrays: sums across batches (appropriate for photon-counting
    spectroscopy where batch spectra should be summed).
    For scalars: sums them.
    For non-numeric values: takes the last batch's value (geometry axes, etc.).

    Parameters
    ----------
    batch_results : list[dict]
        List of attribute dicts from each batch.

    Returns
    -------
    dict
        Merged results with summed arrays.
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
            continue

        first = values[0]
        if isinstance(first, np.ndarray):
            if (
                key in ("ccm_energies", "ccm_bins", "time_bins", "time_delays")
                or key.endswith("_energy")
                or key.endswith("_axis")
                or key.endswith("_bins")
                or key.endswith("_delays")
            ):
                merged[key] = first
            else:
                shapes = [v.shape for v in values]
                if all(s == shapes[0] for s in shapes):
                    merged[key] = np.nansum(values, axis=0)
                else:
                    try:
                        merged[key] = np.concatenate(values, axis=0)
                    except ValueError:
                        merged[key] = values[-1]
        elif isinstance(first, (int, float)):
            # Per-detector geometry scalars (angles, etc.) are identical across
            # batches — take the mean rather than summing.
            if key.endswith("_angle"):
                merged[key] = float(np.mean(values))
            else:
                merged[key] = sum(values)
        else:
            merged[key] = values[-1]

    return merged


def run_batched(
    run,
    pipeline_steps: list[StepConfig],
    cores: int = 1,
    batch_size: int = 2000,
    detector_configs=None,
    scalar_keys=None,
    precomputed_attrs=None,
) -> None:
    """
    Execute pipeline steps on a run with optional batch parallelism.

    If cores == 1, runs sequentially without spawning a Pool.
    If cores > 1, splits into batches and uses multiprocessing (reloading
    data from HDF5 in each worker to avoid pickling large arrays).

    After all batches complete, reconverged attributes are set on the
    original run object so downstream code can access them.

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
    detector_configs : list of tuples, optional
        [(hdf5_path, name, transpose), ...] for parallel reloading.
    scalar_keys : list of tuples, optional
        [(hdf5_path, friendly_name), ...] for parallel reloading.
    precomputed_attrs : dict, optional
        Scalar/axis attributes pre-computed on the full run (e.g. ccm_bins,
        ccm_energies, ccm_bin_indices).  Static attributes are copied directly
        into each batch; per-shot arrays (shape[0] == total_shots) are sliced.
    """
    total_shots = getattr(run, "total_shots", None)
    if total_shots is None:
        run_pipeline(run, pipeline_steps)
        return

    batches = split_into_batches(total_shots, batch_size)

    if not batches:
        return

    if cores <= 1 or len(batches) == 1:
        batch_results = []
        for batch_range in batches:
            result = _process_batch_sequential(
                batch_range,
                run,
                pipeline_steps,
                detector_configs=detector_configs,
                scalar_keys=scalar_keys,
                precomputed_attrs=precomputed_attrs,
            )
            batch_results.append(result)
    else:
        run_file = getattr(run, "run_file", None)
        start_index = getattr(run, "start_index", 0)
        if run_file is None:
            batch_results = []
            for batch_range in batches:
                result = _process_batch_sequential(
                    batch_range,
                    run,
                    pipeline_steps,
                    precomputed_attrs=precomputed_attrs,
                )
                batch_results.append(result)
        else:
            if detector_configs is None:
                detector_configs = []
            if scalar_keys is None:
                scalar_keys = []
            process_fn = partial(
                _process_batch_parallel,
                run_file=run_file,
                start_index=start_index,
                detector_configs=detector_configs,
                scalar_keys=scalar_keys,
                steps=pipeline_steps,
                precomputed_attrs=precomputed_attrs,
                parent_total_shots=total_shots,
            )
            with Pool(processes=cores) as pool:
                batch_results = pool.map(process_fn, batches)

    merged = reconverge_results(batch_results)
    for key, value in merged.items():
        setattr(run, key, value)
    run.update_status(f"Batched execution complete: {len(batches)} batches reconverged")

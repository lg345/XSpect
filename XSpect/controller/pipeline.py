"""
Pipeline: the user-facing entry point for YAML-driven analysis.

Usage:
    pipeline = Pipeline.from_yaml("my_analysis.yaml")
    pipeline.run(cores=16, batch_size=2000)
    results = pipeline.results
"""

import logging
import os

from XSpect.controller.config_parser import parse_yaml, PipelineConfig
from XSpect.controller.pipeline_runner import run_pipeline, run_reductions
from XSpect.controller.batch_manager import run_batched
from XSpect.model.experiment import experiment, spectroscopy_experiment
from XSpect.model.run import spectroscopy_run

logger = logging.getLogger("XSpect")


def enable_logging(level=logging.INFO, log_file=None):
    """Attach handlers to the XSpect logger so pipeline progress is visible.

    Call once before ``Pipeline.run()``:

        from XSpect.controller.pipeline import enable_logging
        enable_logging()                          # stderr only
        enable_logging(log_file="xspect.log")     # stderr + file
        enable_logging(log_file="xspect.log", level=logging.DEBUG)

    Safe to call multiple times; it will not add duplicate handlers.

    Parameters
    ----------
    level : int
        Logging level (default logging.INFO). Use logging.DEBUG for per-step lines.
    log_file : str or None
        If given, also write logs to this file (appended). Worker subprocesses
        do NOT inherit this handler, but the main process logs every batch's
        completion, so the file captures full pipeline progress including the
        point of any hang/OOM.
    """
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[XSpect %(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # stderr handler (for notebook/terminal)
    if not any(getattr(h, "_xspect_stream", False) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh._xspect_stream = True
        logger.addHandler(sh)

    # file handler
    if log_file is not None:
        log_file = os.path.abspath(os.path.expanduser(log_file))
        already = any(
            getattr(h, "_xspect_file", None) == log_file for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(log_file, mode="a")
            fh.setFormatter(fmt)
            fh._xspect_file = log_file
            logger.addHandler(fh)
            logger.info("XSpect logging to file: %s", log_file)

    return logger


class Pipeline:
    """
    YAML-driven analysis pipeline.

    Parses a YAML config, creates experiment/run objects, dispatches
    registered steps, and collects results.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = {}
        self.analyzed_runs = []
        self._status_log = []

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        """
        Create a Pipeline from a YAML configuration file.

        Parameters
        ----------
        path : str
            Path to the YAML config file.

        Returns
        -------
        Pipeline
            Configured pipeline ready to run.
        """
        config = parse_yaml(path)
        return cls(config)

    def run(self, cores: int = 1, batch_size: int = 2000) -> None:
        """
        Execute the pipeline.

        1. Creates an experiment from config
        2. For each run number, creates a spectroscopy_run and executes pipeline steps
        3. After all runs, executes reduction steps
        4. Populates self.results

        Parameters
        ----------
        cores : int
            Number of parallel workers for batch processing.
        batch_size : int
            Number of shots per batch.
        """
        self._status_log.append("Pipeline execution started")
        logger.info(
            "Pipeline execution started (cores=%d, batch_size=%d)", cores, batch_size
        )

        exp = self._create_experiment()

        detector_configs = [
            (dc.hdf5_path, dc.name, dc.transpose, dc.row_range)
            for dc in self.config.data.detector_keys
        ]
        scalar_keys = [(dk.hdf5_path, dk.friendly_name) for dk in self.config.data.keys]
        logger.info("Runs to process: %s", list(self.config.data.runs))

        for run_number in self.config.data.runs:
            self._status_log.append(f"Processing run {run_number}")
            logger.info("── Run %s: creating run object", run_number)
            run = self._create_run(exp, run_number)
            self._load_data(run, skip_detector=True)
            total = getattr(run, "total_shots", None)
            logger.info("Run %s: %s total shots", run_number, total)

            if hasattr(run, "total_shots") and run.total_shots > batch_size:
                n_batches = -(-run.total_shots // batch_size)  # ceil
                logger.info(
                    "Run %s: BATCHED path — %d shots / %d per batch = %d batches on %d cores",
                    run_number,
                    run.total_shots,
                    batch_size,
                    n_batches,
                    cores,
                )
                # Pre-run the pipeline on the parent run (no detector loaded).
                # Detector-dependent steps return gracefully; scalar-only steps
                # (e.g. make_ccm_axis, ccm_binning, time_binning) fire here and
                # produce a globally consistent axis.  The resulting attributes
                # are then injected into every batch so each batch shares the
                # same ccm_bins / ccm_energies / time_bins instead of deriving
                # its own from an incomplete slice of the data.
                run_pipeline(run, self.config.pipeline)
                precomputed_attrs = self._collect_scalar_precomputed(run)
                logger.info(
                    "Run %s: pre-pass complete, dispatching batches…", run_number
                )

                run_batched(
                    run,
                    self.config.pipeline,
                    cores=cores,
                    batch_size=batch_size,
                    detector_configs=detector_configs,
                    scalar_keys=scalar_keys,
                    precomputed_attrs=precomputed_attrs,
                )
            else:
                logger.info(
                    "Run %s: SINGLE path — loading detector into memory", run_number
                )
                self._load_detector(run)
                run_pipeline(run, self.config.pipeline)

            self.analyzed_runs.append(run)
            self._status_log.append(f"Completed run {run_number}")
            logger.info("Run %s: complete", run_number)

        if self.config.reduction:
            self._status_log.append("Running reduction steps")
            logger.info("Running %d reduction step(s)", len(self.config.reduction))
            reduction_results = run_reductions(
                self.analyzed_runs, self.config.reduction
            )
            self.results.update(reduction_results)

        for run in self.analyzed_runs:
            for key, value in run.results.items():
                self.results[f"run_{run.run_number}.{key}"] = value
            self._collect_run_attributes(run)

        self._status_log.append("Pipeline execution complete")

    def _collect_scalar_precomputed(self, run) -> dict:
        """Collect scalar-derived attributes set during the pre-pipeline run.

        Returns attributes that are either non-array scalars or small arrays
        whose first dimension does NOT equal total_shots (i.e. axis/bin arrays
        like ccm_bins, ccm_energies, time_bins), plus per-shot arrays that
        should be sliced per-batch (ccm_bin_indices, timing_bin_indices).

        These are later injected into each batch so that axis steps (make_ccm_axis,
        time_binning) see a pre-populated result and skip re-derivation.

        Raw input data keys (scalar_keys and detector_keys) are excluded: they
        are reloaded fresh from HDF5 in each batch worker and must not be
        overwritten by a pre-pipeline-mutated (e.g. union_shots-filtered) copy.
        """
        import numpy as np

        # Names that will be reloaded per-batch from HDF5 — never inject these.
        input_names = {dk.friendly_name for dk in self.config.data.keys}
        input_names |= {dc.name for dc in self.config.data.detector_keys}

        skip = {
            "spec_experiment",
            "run_number",
            "run_file",
            "status",
            "status_datetime",
            "verbose",
            "end_index",
            "start_index",
            "results",
            "total_shots",
            "run_shots",
            "xray",
            "laser",
            "simultaneous",
            "h5",
        } | input_names

        total = getattr(run, "total_shots", None)
        attrs = {}
        for attr, value in vars(run).items():
            if attr in skip or attr.startswith("_"):
                continue
            if isinstance(value, np.ndarray):
                attrs[attr] = value
            elif not isinstance(value, np.ndarray) and value is not None:
                # Skip non-array objects (h5 handles, etc.) but keep scalars
                if isinstance(value, (int, float, bool, str)):
                    attrs[attr] = value
        return attrs

    def _collect_run_attributes(self, run):
        """Collect pipeline-generated attributes from a run into self.results."""
        skip = {
            "spec_experiment",
            "run_number",
            "run_file",
            "status",
            "status_datetime",
            "verbose",
            "end_index",
            "start_index",
            "results",
            "total_shots",
            "run_shots",
            "xray",
            "laser",
            "simultaneous",
            "h5",
        }
        import numpy as np

        for attr, value in vars(run).items():
            if attr in skip or attr.startswith("_"):
                continue
            if isinstance(value, np.ndarray):
                self.results[attr] = value

    def _create_experiment(self):
        """Create experiment object from config. Wraps the directory lookup failure gracefully."""
        cfg = self.config.experiment
        try:
            exp = spectroscopy_experiment(
                cfg.lcls_run,
                cfg.hutch,
                cfg.experiment_id,
                smalldata_dir=cfg.smalldata_dir,
            )
        except Exception:
            exp = _MockExperiment(cfg.lcls_run, cfg.hutch, cfg.experiment_id)
        return exp

    def _create_run(self, exp, run_number: int):
        """Create a spectroscopy_run for the given run number."""
        max_shots = self.config.data.max_shots
        end_index = max_shots if max_shots is not None else -1
        try:
            run = spectroscopy_run(exp, run_number, end_index=end_index)
        except Exception:
            run = spectroscopy_run.__new__(spectroscopy_run)
            run.spec_experiment = exp
            run.run_number = run_number
            run.run_file = None
            run.status = []
            run.status_datetime = []
            run.verbose = False
            run.end_index = end_index
            run.start_index = 0
            run.results = {}
        return run

    def _load_data(self, run, skip_detector=False):
        """Load data keys into the run if the HDF5 file is accessible.

        Parameters
        ----------
        skip_detector : bool
            If True, skip loading detector arrays (used when batching will
            reload per-batch from HDF5 to avoid loading multi-GB arrays).
        """
        if run.run_file is None or not _file_exists(run.run_file):
            return

        keys = [dk.hdf5_path for dk in self.config.data.keys]
        names = [dk.friendly_name for dk in self.config.data.keys]
        if keys:
            run.load_run_keys(keys, names)

        run.get_run_shot_properties()

        if not skip_detector:
            for det_config in self.config.data.detector_keys:
                kwargs = {}
                if det_config.rois is not None:
                    kwargs["rois"] = det_config.rois
                    kwargs["combine"] = det_config.combine_rois
                if det_config.row_range is not None:
                    kwargs["row_range"] = det_config.row_range
                run.load_run_key_delayed(
                    [det_config.hdf5_path],
                    [det_config.name],
                    **kwargs,
                )

            # Close the h5py file handle so the run object is picklable for multiprocessing
            if hasattr(run, "h5"):
                run.h5.close()
                del run.h5

    def _load_detector(self, run):
        """Load detector data into run (for non-batched path)."""
        if run.run_file is None or not _file_exists(run.run_file):
            return
        for det_config in self.config.data.detector_keys:
            kwargs = {}
            if det_config.rois is not None:
                kwargs["rois"] = det_config.rois
                kwargs["combine"] = det_config.combine_rois
            kwargs["transpose"] = det_config.transpose
            if det_config.row_range is not None:
                kwargs["row_range"] = det_config.row_range
            run.load_run_key_delayed(
                [det_config.hdf5_path],
                [det_config.name],
                **kwargs,
            )
        if hasattr(run, "h5"):
            run.h5.close()
            del run.h5


class _MockExperiment:
    """Fallback experiment object when real data paths are unavailable."""

    def __init__(self, lcls_run, hutch, experiment_id):
        self.lcls_run = lcls_run
        self.hutch = hutch
        self.experiment_id = experiment_id
        self.experiment_directory = "/tmp/xspect_mock"


def _file_exists(path: str) -> bool:
    """Check if a file exists without importing os at module level."""
    from pathlib import Path

    return Path(path).exists()

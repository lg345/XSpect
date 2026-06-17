"""
Pipeline: the user-facing entry point for YAML-driven analysis.

Usage:
    pipeline = Pipeline.from_yaml("my_analysis.yaml")
    pipeline.run(cores=16, batch_size=2000)
    results = pipeline.results
"""

from XSpect.controller.config_parser import parse_yaml, PipelineConfig
from XSpect.controller.pipeline_runner import run_pipeline, run_reductions
from XSpect.controller.batch_manager import run_batched
from XSpect.model.experiment import experiment, spectroscopy_experiment
from XSpect.model.run import spectroscopy_run


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

        exp = self._create_experiment()

        for run_number in self.config.data.runs:
            self._status_log.append(f"Processing run {run_number}")
            run = self._create_run(exp, run_number)
            self._load_data(run)

            if hasattr(run, 'total_shots') and run.total_shots > batch_size:
                run_batched(run, self.config.pipeline, cores=cores, batch_size=batch_size)
            else:
                run_pipeline(run, self.config.pipeline)

            self.analyzed_runs.append(run)
            self._status_log.append(f"Completed run {run_number}")

        if self.config.reduction:
            self._status_log.append("Running reduction steps")
            reduction_results = run_reductions(self.analyzed_runs, self.config.reduction)
            self.results.update(reduction_results)

        for run in self.analyzed_runs:
            for key, value in run.results.items():
                self.results[f"run_{run.run_number}.{key}"] = value

        self._status_log.append("Pipeline execution complete")

    def _create_experiment(self):
        """Create experiment object from config. Wraps the directory lookup failure gracefully."""
        cfg = self.config.experiment
        try:
            exp = spectroscopy_experiment(cfg.lcls_run, cfg.hutch, cfg.experiment_id)
        except Exception:
            exp = _MockExperiment(cfg.lcls_run, cfg.hutch, cfg.experiment_id)
        return exp

    def _create_run(self, exp, run_number: int):
        """Create a spectroscopy_run for the given run number."""
        try:
            run = spectroscopy_run(exp, run_number)
        except Exception:
            run = spectroscopy_run.__new__(spectroscopy_run)
            run.spec_experiment = exp
            run.run_number = run_number
            run.run_file = None
            run.status = []
            run.status_datetime = []
            run.verbose = False
            run.end_index = -1
            run.start_index = 0
            run.results = {}
        return run

    def _load_data(self, run):
        """Load data keys into the run if the HDF5 file is accessible."""
        if run.run_file is None or not _file_exists(run.run_file):
            return

        keys = [dk.hdf5_path for dk in self.config.data.keys]
        names = [dk.friendly_name for dk in self.config.data.keys]
        if keys:
            run.load_run_keys(keys, names)

        run.get_run_shot_properties()

        for det_config in self.config.data.detector_keys:
            kwargs = {}
            if det_config.rois is not None:
                kwargs['rois'] = det_config.rois
                kwargs['combine'] = det_config.combine_rois
            run.load_run_key_delayed(
                [det_config.hdf5_path],
                [det_config.name],
                **kwargs,
            )

        # Close the h5py file handle so the run object is picklable for multiprocessing
        if hasattr(run, 'h5'):
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

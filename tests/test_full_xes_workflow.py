"""
End-to-end integration test: XES time-resolved workflow via Pipeline.

Proves that a YAML config replicating XESBatchAnalysis can execute
through the Pipeline class and produce meaningful results.
Uses synthetic data (no HDF5, no S3DF).
"""

import numpy as np
import pytest
import yaml
import tempfile
import os

from XSpect.controller.pipeline import Pipeline
from XSpect.controller.config_parser import parse_yaml
from XSpect.analysis.registry import get_step, list_steps


class SyntheticRun:
    """Mimics a spectroscopy_run loaded with data, bypassing HDF5."""

    def __init__(self, n_shots=1000, n_pixels=80):
        np.random.seed(42)
        self.run_number = 1
        self.status = []
        self.status_datetime = []
        self.results = {}
        self.total_shots = n_shots
        self.start_index = 0
        self.end_index = n_shots
        self.verbose = False

        # Shot properties
        self.xray = (np.random.random(n_shots) > 0.2).astype(float)
        self.laser = (np.random.random(n_shots) > 0.4).astype(float)
        self.simultaneous = (self.xray.astype(bool) & self.laser.astype(bool)).astype(float)

        # IPM
        self.ipm = np.random.exponential(3e4, n_shots)
        self.ipm[::50] = 500  # some low-intensity shots

        # Detector (already reduced to 2D: shots x pixels)
        base_spectrum = np.exp(-0.5 * ((np.arange(n_pixels) - 40) / 5) ** 2)
        self.epix_ROI_1 = np.outer(np.ones(n_shots), base_spectrum) * np.random.poisson(50, (n_shots, n_pixels))
        # Add noise
        self.epix_ROI_1 += np.random.normal(0, 2, (n_shots, n_pixels))

        # Timing
        self.lxt_ttc = np.random.uniform(-2, 10, n_shots)
        self.encoder = np.zeros(n_shots)

    def update_status(self, msg):
        self.status.append(msg)


@pytest.fixture
def xes_yaml_path():
    """Create a YAML config for XES workflow."""
    config = {
        "experiment": {
            "hutch": "mfx",
            "experiment_id": "mfx00000",
            "lcls_run": 1,
        },
        "data": {
            "runs": [1],
            "keys": {"ipm4/sum": "ipm"},
            "detector_keys": {
                "epix_1/ROI_0_area": {
                    "name": "epix",
                    "rois": [[0, 80]],
                    "combine_rois": True,
                }
            },
        },
        "pipeline": [
            {"step": "filter_shots", "on": "xray", "filter_key": "ipm", "threshold": 1000},
            {"step": "filter_shots", "on": "simultaneous", "filter_key": "ipm", "threshold": 1000},
            {"step": "filter_detector_adu", "on": "epix_ROI_1", "adu_threshold": 1.0},
            {"step": "union_shots", "on": "epix_ROI_1", "filter_keys": ["simultaneous", "laser"]},
            {"step": "separate_shots", "on": "epix_ROI_1", "filter_keys": ["xray", "laser"]},
            {"step": "time_binning", "bins": [-2, 10, 25], "lxt_key": "lxt_ttc"},
            {"step": "union_shots", "on": "timing_bin_indices", "filter_keys": ["simultaneous", "laser"]},
            {"step": "separate_shots", "on": "timing_bin_indices", "filter_keys": ["xray", "laser"]},
            {
                "step": "reduce_detector_temporal",
                "on": "epix_ROI_1_simultaneous_laser",
                "timing_bin_key": "timing_bin_indices_simultaneous_laser",
            },
            {
                "step": "reduce_detector_temporal",
                "on": "epix_ROI_1_xray_not_laser",
                "timing_bin_key": "timing_bin_indices_xray_not_laser",
            },
            {"step": "normalize_xes", "on": "epix_ROI_1_simultaneous_laser_time_binned"},
            {"step": "normalize_xes", "on": "epix_ROI_1_xray_not_laser_time_binned"},
            {
                "step": "make_energy_axis",
                "detector_key": "epix_ROI_1",
                "crystal_detector_distance": 200.0,
                "crystal_radius": 250.0,
                "d_spacing": 1.637,
                "name": "xes",
            },
        ],
        "reduction": [
            {
                "step": "combine_runs",
                "detector_key": "epix_ROI_1",
            }
        ],
        "output": {"format": "hdf5", "path": "./results/"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        path = f.name

    yield path
    os.unlink(path)


def test_xes_workflow_parses(xes_yaml_path):
    """YAML config parses without error."""
    config = parse_yaml(xes_yaml_path, validate_steps=True)
    assert len(config.pipeline) == 13
    assert len(config.reduction) == 1


def test_xes_workflow_step_dispatch(xes_yaml_path):
    """Each step in the YAML can be looked up in the registry."""
    config = parse_yaml(xes_yaml_path, validate_steps=True)
    for step_config in config.pipeline:
        fn = get_step(step_config.step)
        assert callable(fn)


def test_xes_workflow_executes(xes_yaml_path):
    """Full pipeline executes on synthetic data and produces expected outputs."""
    config = parse_yaml(xes_yaml_path, validate_steps=True)

    # Create synthetic run (bypassing HDF5 loading)
    run = SyntheticRun(n_shots=1000, n_pixels=80)

    # Execute pipeline steps manually (simulating what pipeline_runner does)
    from XSpect.controller.pipeline_runner import run_pipeline, run_reductions
    run_pipeline(run, config.pipeline)

    # Check expected outputs exist
    assert hasattr(run, "epix_ROI_1_simultaneous_laser_time_binned")
    assert hasattr(run, "epix_ROI_1_xray_not_laser_time_binned")
    assert hasattr(run, "epix_ROI_1_simultaneous_laser_time_binned_normalized")
    assert hasattr(run, "epix_ROI_1_xray_not_laser_time_binned_normalized")
    assert hasattr(run, "xes_energy")

    # Check shapes
    on_binned = run.epix_ROI_1_simultaneous_laser_time_binned
    off_binned = run.epix_ROI_1_xray_not_laser_time_binned
    assert on_binned.shape == (25, 80)  # 25 time bins x 80 pixels
    assert off_binned.shape == (25, 80)

    # Normalized should sum to 1 per row
    on_norm = run.epix_ROI_1_simultaneous_laser_time_binned_normalized
    row_sums = np.sum(on_norm, axis=1)
    # Rows with data should sum to ~1
    nonzero_rows = row_sums[row_sums > 0]
    if len(nonzero_rows) > 0:
        np.testing.assert_allclose(nonzero_rows, 1.0, atol=1e-10)

    # Energy axis should be monotonic
    assert len(run.xes_energy) == 80
    diffs = np.diff(run.xes_energy)
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_xes_workflow_reduction(xes_yaml_path):
    """Reduction step aggregates across runs."""
    config = parse_yaml(xes_yaml_path, validate_steps=True)
    from XSpect.controller.pipeline_runner import run_pipeline, run_reductions

    # Process two runs
    runs = []
    for i in range(2):
        run = SyntheticRun(n_shots=500, n_pixels=80)
        run.run_number = i + 1
        run_pipeline(run, config.pipeline)
        runs.append(run)

    # Run reduction (returns flat dict with keys from combine_runs return value)
    reduction_results = run_reductions(runs, config.reduction)
    assert "laser_on_summed" in reduction_results
    assert "laser_off_summed" in reduction_results
    assert "difference" in reduction_results
    assert reduction_results["laser_on_summed"].shape == (25, 80)


def test_all_registered_steps_discoverable():
    """Verify all expected steps are registered after import."""
    steps = list_steps()
    expected = [
        "filter_shots", "filter_detector_adu", "union_shots", "separate_shots",
        "reduce_detector_spatial", "reduce_detector_temporal", "time_binning",
        "normalize_xes", "make_energy_axis", "make_ccm_axis", "ccm_binning",
        "reduce_detector_ccm", "reduce_detector_ccm_temporal",
        "load_run_keys", "load_detector", "get_run_shot_properties",
        "reduce_detector_shots", "bin_uniques", "purge_keys",
        "apply_roi", "patch_pixels",
    ]
    for name in expected:
        assert name in steps, f"Step '{name}' not registered"

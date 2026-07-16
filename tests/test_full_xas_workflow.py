"""
End-to-end integration test: XAS workflow via pipeline.

Proves that a YAML config replicating XASBatchAnalysis can execute
and produce time-energy 2D binned results.
"""

import numpy as np
import pytest
import os

from XSpect.controller.config_parser import parse_yaml
from XSpect.controller.pipeline_runner import run_pipeline
from XSpect.analysis.registry import get_step


class SyntheticXASRun:
    """Mimics a spectroscopy_run with XAS data (CCM energy scan)."""

    def __init__(self, n_shots=800, n_pixels=60):
        np.random.seed(123)
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
        self.ipm = np.random.exponential(2e4, n_shots)

        # CCM energy (scanning 7100-7200 eV across shots)
        self.ccm = np.linspace(7100, 7200, n_shots) + np.random.normal(0, 0.5, n_shots)

        # Detector (shots x pixels)
        self.epix_ROI_1 = np.random.poisson(8, (n_shots, n_pixels)).astype(float)

        # Timing
        self.lxt_ttc = np.random.uniform(-1, 5, n_shots)
        self.encoder = np.zeros(n_shots)

    def update_status(self, msg):
        self.status.append(msg)


@pytest.fixture
def xas_yaml_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "test_xas_workflow.yaml")


def test_xas_workflow_parses(xas_yaml_path):
    """XAS YAML config parses and validates."""
    config = parse_yaml(xas_yaml_path, validate_steps=True)
    assert len(config.pipeline) == 17
    assert config.experiment.hutch == "xcs"


def test_xas_workflow_executes(xas_yaml_path):
    """Full XAS pipeline produces 2D time-energy binned results."""
    config = parse_yaml(xas_yaml_path, validate_steps=True)
    run = SyntheticXASRun(n_shots=800, n_pixels=60)

    run_pipeline(run, config.pipeline)

    # Check 2D reduction outputs
    assert hasattr(run, "epix_ROI_1_simultaneous_laser_time_energy_binned")
    assert hasattr(run, "epix_ROI_1_xray_not_laser_time_energy_binned")
    assert hasattr(run, "ipm_simultaneous_laser_time_energy_binned")
    assert hasattr(run, "ipm_xray_not_laser_time_energy_binned")

    # Check shapes: (time_bins x energy_bins x pixels) for detector
    epix_result = run.epix_ROI_1_simultaneous_laser_time_energy_binned
    assert epix_result.ndim == 3
    assert epix_result.shape[0] == 15  # time bins
    assert epix_result.shape[1] == 51  # energy bins (len(ccm_bins))
    assert epix_result.shape[2] == 60  # pixels

    # IPM is scalar: (time_bins x energy_bins)
    ipm_result = run.ipm_simultaneous_laser_time_energy_binned
    assert ipm_result.ndim == 2
    assert ipm_result.shape == (15, 51)


def test_xas_ccm_axis_correct(xas_yaml_path):
    """CCM axis has expected energy range."""
    config = parse_yaml(xas_yaml_path, validate_steps=True)
    run = SyntheticXASRun()

    run_pipeline(run, config.pipeline)

    assert hasattr(run, "ccm_energies")
    assert len(run.ccm_energies) == 50
    assert run.ccm_energies[0] == pytest.approx(7100, abs=1)
    assert run.ccm_energies[-1] == pytest.approx(7200, abs=1)

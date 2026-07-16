"""Tests for XAS-specific pipeline steps."""

import numpy as np
import pytest
from XSpect.analysis.registry import get_step


class MockRun:
    def __init__(self):
        self.status = []
        self.results = {}

    def update_status(self, msg):
        self.status.append(msg)


@pytest.fixture
def xas_run():
    """Run with CCM energy scan data."""
    np.random.seed(42)
    run = MockRun()
    n_shots = 500
    n_pixels = 60

    # Simulate CCM energy values (scanning across an edge)
    run.ccm = np.random.uniform(7100, 7200, n_shots)

    # Simulate detector data
    run.epix_ROI_1 = np.random.poisson(10, (n_shots, n_pixels)).astype(float)
    run.ipm = np.random.exponential(2e4, n_shots)

    # Timing data
    run.lxt_ttc = np.random.uniform(-1, 5, n_shots)
    run.encoder = np.zeros(n_shots)

    # Masks
    run.xray = np.random.random(n_shots) > 0.3
    run.laser = np.random.random(n_shots) > 0.5
    run.simultaneous = run.xray & run.laser

    return run


class TestMakeCCMAxis:
    def test_ccm_axis_from_linspace(self, xas_run):
        step = get_step("make_ccm_axis")
        step(xas_run, energies=[7100, 7200, 50])
        assert hasattr(xas_run, "ccm_bins")
        assert hasattr(xas_run, "ccm_energies")
        assert len(xas_run.ccm_energies) == 50

    def test_ccm_axis_from_list(self, xas_run):
        energies = np.linspace(7100, 7200, 30).tolist()
        step = get_step("make_ccm_axis")
        step(xas_run, energies=energies)
        assert len(xas_run.ccm_energies) == 30


class TestCCMBinning:
    def test_ccm_binning(self, xas_run):
        # First create CCM axis
        get_step("make_ccm_axis")(xas_run, energies=[7100, 7200, 50])

        # Then bin
        step = get_step("ccm_binning")
        step(xas_run, ccm_key="ccm", ccm_bins_key="ccm_bins")
        assert hasattr(xas_run, "ccm_bin_indices")
        assert len(xas_run.ccm_bin_indices) == 500


class TestReduceDetectorCCM:
    def test_reduce_ccm_1d(self, xas_run):
        get_step("make_ccm_axis")(xas_run, energies=[7100, 7200, 50])
        get_step("ccm_binning")(xas_run, ccm_key="ccm", ccm_bins_key="ccm_bins")

        step = get_step("reduce_detector_ccm")
        step(xas_run, on="ipm", ccm_bin_key="ccm_bin_indices")
        result = getattr(xas_run, "ipm_energy_binned")
        assert result is not None
        assert result.shape == (51,)  # len(ccm_bins)

    def test_reduce_ccm_2d(self, xas_run):
        get_step("make_ccm_axis")(xas_run, energies=[7100, 7200, 50])
        get_step("ccm_binning")(xas_run, ccm_key="ccm", ccm_bins_key="ccm_bins")

        step = get_step("reduce_detector_ccm")
        step(xas_run, on="epix_ROI_1", ccm_bin_key="ccm_bin_indices")
        result = getattr(xas_run, "epix_ROI_1_energy_binned")
        assert result.shape == (51, 60)  # (n_bins, n_pixels)

    def test_reduce_ccm_average(self, xas_run):
        get_step("make_ccm_axis")(xas_run, energies=[7100, 7200, 50])
        get_step("ccm_binning")(xas_run, ccm_key="ccm", ccm_bins_key="ccm_bins")

        step = get_step("reduce_detector_ccm")
        step(xas_run, on="ipm", ccm_bin_key="ccm_bin_indices", average=True)
        result = getattr(xas_run, "ipm_energy_binned")
        # Averaged values should be in the range of raw IPM values
        nonzero = result[result > 0]
        if len(nonzero) > 0:
            assert np.mean(nonzero) < 1e5


class TestReduceDetectorCCMTemporal:
    def test_2d_reduction(self, xas_run):
        # Set up time bins
        get_step("time_binning")(xas_run, bins=[-1, 5, 15], lxt_key="lxt_ttc")
        # Set up CCM bins
        get_step("make_ccm_axis")(xas_run, energies=[7100, 7200, 20])
        get_step("ccm_binning")(xas_run, ccm_key="ccm", ccm_bins_key="ccm_bins")

        step = get_step("reduce_detector_ccm_temporal")
        step(xas_run, on="epix_ROI_1",
             timing_bin_key="timing_bin_indices",
             ccm_bin_key="ccm_bin_indices")
        result = getattr(xas_run, "epix_ROI_1_time_energy_binned")
        assert result is not None
        # Should be (time_bins x energy_bins x pixels)
        assert result.shape == (15, 21, 60)

    def test_2d_reduction_scalar(self, xas_run):
        get_step("time_binning")(xas_run, bins=[-1, 5, 10], lxt_key="lxt_ttc")
        get_step("make_ccm_axis")(xas_run, energies=[7100, 7200, 20])
        get_step("ccm_binning")(xas_run, ccm_key="ccm", ccm_bins_key="ccm_bins")

        step = get_step("reduce_detector_ccm_temporal")
        step(xas_run, on="ipm",
             timing_bin_key="timing_bin_indices",
             ccm_bin_key="ccm_bin_indices")
        result = getattr(xas_run, "ipm_time_energy_binned")
        # Should be (time_bins x energy_bins)
        assert result.shape == (10, 21)

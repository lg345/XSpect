"""Tests for registered spectroscopy analysis steps using synthetic data."""

import numpy as np
import pytest
from XSpect.analysis.registry import get_step, get_reduction


class MockRun:
    """Minimal run object for testing steps without HDF5."""

    def __init__(self, n_shots=1000, n_pixels=100):
        self.status = []
        self.status_datetime = []
        self.results = {}
        self.total_shots = n_shots

        # Simulate xray/laser masks (70% xray, 50% laser, 35% simultaneous)
        self.xray = np.random.random(n_shots) > 0.3
        self.laser = np.random.random(n_shots) > 0.5
        self.simultaneous = self.xray & self.laser

        # Simulate IPM (intensity monitor)
        self.ipm = np.random.exponential(2e4, n_shots)
        self.ipm[::20] = np.nan  # some NaN shots

        # Simulate 2D detector data (shots x pixels)
        self.epix_ROI_1 = np.random.poisson(5, (n_shots, n_pixels)).astype(float)
        # Add some noise below ADU threshold
        self.epix_ROI_1 += np.random.normal(0, 1, (n_shots, n_pixels))

        # Simulate timing data
        self.lxt_ttc = np.random.uniform(-2, 10, n_shots)  # picoseconds
        self.encoder = np.zeros(n_shots)  # no fast delay

    def update_status(self, msg):
        self.status.append(msg)


@pytest.fixture
def mock_run():
    np.random.seed(42)
    return MockRun(n_shots=500, n_pixels=80)


@pytest.fixture
def mock_run_3d():
    """Run with 3D detector (shots x rows x cols)."""
    np.random.seed(42)
    run = MockRun(n_shots=200, n_pixels=80)
    run.epix = np.random.poisson(3, (200, 50, 80)).astype(float)
    return run


class TestFilterSteps:
    def test_filter_shots_ipm_threshold(self, mock_run):
        step = get_step("filter_shots")
        before = np.sum(mock_run.xray)
        step(mock_run, on="xray", filter_key="ipm", threshold=1e4)
        after = np.sum(mock_run.xray)
        assert after <= before
        assert after > 0

    def test_filter_shots_range_threshold(self, mock_run):
        step = get_step("filter_shots")
        step(mock_run, on="simultaneous", filter_key="ipm", threshold=[5000, 50000])
        assert np.sum(mock_run.simultaneous) > 0

    def test_filter_detector_adu(self, mock_run):
        step = get_step("filter_detector_adu")
        step(mock_run, on="epix_ROI_1", adu_threshold=3.0)
        # All values should be 0 or > 3.0
        data = mock_run.epix_ROI_1
        assert np.all((data == 0) | (data > 3.0))

    def test_filter_detector_adu_range(self, mock_run):
        step = get_step("filter_detector_adu")
        step(mock_run, on="epix_ROI_1", adu_threshold=[2.0, 20.0])
        data = mock_run.epix_ROI_1
        assert np.all((data == 0) | ((data > 2.0) & (data < 20.0)))


class TestShotCombination:
    def test_union_shots(self, mock_run):
        step = get_step("union_shots")
        step(mock_run, on="epix_ROI_1", filter_keys=["simultaneous", "laser"])
        key = "epix_ROI_1_simultaneous_laser"
        result = getattr(mock_run, key)
        expected_count = int(np.sum(mock_run.simultaneous & mock_run.laser))
        assert result.shape[0] == expected_count

    def test_separate_shots(self, mock_run):
        step = get_step("separate_shots")
        step(mock_run, on="epix_ROI_1", filter_keys=["xray", "laser"])
        key = "epix_ROI_1_xray_not_laser"
        result = getattr(mock_run, key)
        expected_count = int(np.sum(mock_run.xray & ~mock_run.laser))
        assert result.shape[0] == expected_count


class TestSpatialReduction:
    def test_reduce_detector_spatial_combine(self, mock_run_3d):
        step = get_step("reduce_detector_spatial")
        step(mock_run_3d, on="epix", rois=[[10, 40], [50, 70]], combine_rois=True, purge=False)
        result = getattr(mock_run_3d, "epix_ROI_1")
        # Spatial dimension summed away: (shots, rows, cols[masked]) -> (shots, rows)
        assert result.ndim == 2
        assert result.shape[0] == 200

    def test_reduce_detector_spatial_separate(self, mock_run_3d):
        step = get_step("reduce_detector_spatial")
        step(mock_run_3d, on="epix", rois=[[10, 30], [50, 70]], combine_rois=False, purge=False)
        roi1 = getattr(mock_run_3d, "epix_ROI_1")
        roi2 = getattr(mock_run_3d, "epix_ROI_2")
        assert roi1 is not None
        assert roi2 is not None

    def test_apply_roi_preserves_spatial(self, mock_run_3d):
        step = get_step("apply_roi")
        step(mock_run_3d, on="epix", rois=[[20, 60]], combine_rois=True)
        result = getattr(mock_run_3d, "epix_ROI_1")
        # ROI preserves spatial: (shots, rows, cols_subset)
        assert result.ndim == 3
        assert result.shape[2] == 40  # 60-20=40 pixels


class TestTimeBinning:
    def test_time_binning_creates_indices(self, mock_run):
        step = get_step("time_binning")
        step(mock_run, bins=[-2, 10, 25], lxt_key="lxt_ttc")
        assert hasattr(mock_run, "time_bins")
        assert hasattr(mock_run, "timing_bin_indices")
        assert len(mock_run.time_bins) == 25
        assert len(mock_run.timing_bin_indices) == mock_run.total_shots

    def test_reduce_detector_temporal(self, mock_run):
        # First set up time bins
        bins_step = get_step("time_binning")
        bins_step(mock_run, bins=[-2, 10, 20], lxt_key="lxt_ttc")

        # Now reduce temporally
        step = get_step("reduce_detector_temporal")
        step(mock_run, on="epix_ROI_1", timing_bin_key="timing_bin_indices")
        result = getattr(mock_run, "epix_ROI_1_time_binned")
        assert result.shape == (20, 80)  # (n_bins, n_pixels)

    def test_reduce_detector_temporal_average(self, mock_run):
        bins_step = get_step("time_binning")
        bins_step(mock_run, bins=[-2, 10, 20], lxt_key="lxt_ttc")

        step = get_step("reduce_detector_temporal")
        step(mock_run, on="epix_ROI_1", timing_bin_key="timing_bin_indices", average=True)
        result = getattr(mock_run, "epix_ROI_1_time_binned")
        # Averaged values should be reasonable (not huge sums)
        assert np.nanmax(result) < 100


class TestShotReduction:
    def test_reduce_detector_shots_sum(self, mock_run):
        step = get_step("reduce_detector_shots")
        step(mock_run, on="epix_ROI_1", reduction="sum", purge=False)
        result = getattr(mock_run, "epix_ROI_1_reduced")
        assert result.shape == (80,)

    def test_reduce_detector_shots_mean(self, mock_run):
        step = get_step("reduce_detector_shots")
        step(mock_run, on="epix_ROI_1", reduction="mean", purge=False)
        result = getattr(mock_run, "epix_ROI_1_reduced")
        assert result.shape == (80,)


class TestBinUniques:
    def test_bin_uniques(self, mock_run):
        mock_run.scan_var = np.repeat(np.arange(10), 50)
        step = get_step("bin_uniques")
        step(mock_run, on="scan_var")
        assert hasattr(mock_run, "scanvar_indices")
        assert hasattr(mock_run, "scanvar_bins")
        assert len(mock_run.scanvar_bins) == 11  # 10 unique + 1 edge


class TestPurgeKeys:
    def test_purge_keys(self, mock_run):
        assert mock_run.epix_ROI_1 is not None
        step = get_step("purge_keys")
        step(mock_run, keys=["epix_ROI_1"])
        assert mock_run.epix_ROI_1 is None

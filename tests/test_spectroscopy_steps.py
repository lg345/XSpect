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


class TestCommonModeCorrection:
    def test_per_row_offset_removed(self):
        # 3 shots, 4 rows, 6 cols; each row carries a known constant offset
        run = MockRun()
        offsets = np.array([1.0, 2.0, 3.0, 4.0])
        data = np.zeros((3, 4, 6)) + offsets[np.newaxis, :, np.newaxis]
        run.det = data.copy()
        get_step("common_mode_correction")(run, on="det", axis="row")
        np.testing.assert_allclose(run.det, 0.0, atol=1e-12)

    def test_flat_frame_unchanged(self):
        run = MockRun()
        run.det = np.full((2, 5, 5), 7.0)
        get_step("common_mode_correction")(run, on="det", axis="row", method="mean")
        np.testing.assert_allclose(run.det, 0.0, atol=1e-12)

    def test_reference_band_isolates_offset(self):
        # signal in cols 0-2, dark band in cols 3-5 carrying offset 5
        run = MockRun()
        data = np.zeros((1, 3, 6))
        data[0, :, 0:3] = 100.0
        data[0, :, 3:6] = 5.0
        run.det = data.copy()
        get_step("common_mode_correction")(run, on="det", axis="row", reference=[3, 6])
        np.testing.assert_allclose(run.det[0, :, 0:3], 95.0, atol=1e-12)
        np.testing.assert_allclose(run.det[0, :, 3:6], 0.0, atol=1e-12)

    def test_column_axis(self):
        run = MockRun()
        col_offsets = np.array([1.0, 2.0, 3.0])
        run.det = np.zeros((2, 4, 3)) + col_offsets[np.newaxis, np.newaxis, :]
        get_step("common_mode_correction")(run, on="det", axis="column")
        np.testing.assert_allclose(run.det, 0.0, atol=1e-12)

    def test_bank_axis(self):
        # 8 cols, bank_size 4 -> two banks with distinct offsets
        run = MockRun()
        data = np.zeros((1, 3, 8))
        data[0, :, 0:4] = 10.0
        data[0, :, 4:8] = 20.0
        run.det = data.copy()
        get_step("common_mode_correction")(run, on="det", axis="bank", bank_size=4)
        np.testing.assert_allclose(run.det, 0.0, atol=1e-12)

    def test_shape_preserved_2d(self):
        run = MockRun()
        run.det = np.random.rand(6, 6)
        get_step("common_mode_correction")(run, on="det", axis="row")
        assert run.det.shape == (6, 6)

    def test_missing_key_noop(self):
        run = MockRun()
        get_step("common_mode_correction")(run, on="nope", axis="row")
        assert any("not found" in s for s in run.status)


class TestSubtractPolynomialBackground:
    def test_gaussian_on_linear_background(self):
        # narrow peak, generous mask so tails don't contaminate the fit region
        n = 100
        x = np.arange(n, dtype=float)
        bkg = 2.0 + 0.05 * x
        peak = 500.0 * np.exp(-((x - 50) ** 2) / (2 * 3.0**2))
        run = MockRun()
        run.spec = bkg + peak
        get_step("subtract_polynomial_background")(
            run, on="spec", order=1, peak_mask=[30, 70]
        )
        result = run.spec_bkgsub
        np.testing.assert_allclose(result[0:25], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[75:], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.sum(), peak.sum(), rtol=1e-4)

    def test_pure_background_yields_zero(self):
        n = 80
        x = np.arange(n, dtype=float)
        run = MockRun()
        run.spec = 3.0 - 0.02 * x + 0.001 * x**2
        get_step("subtract_polynomial_background")(run, on="spec", order=2)
        np.testing.assert_allclose(run.spec_bkgsub, 0.0, atol=1e-8)

    def test_background_ranges(self):
        n = 100
        x = np.arange(n, dtype=float)
        peak = 300.0 * np.exp(-((x - 50) ** 2) / (2 * 3.0**2))
        run = MockRun()
        run.spec = 10.0 + 0.1 * x + peak
        get_step("subtract_polynomial_background")(
            run, on="spec", order=1, background=[[0, 30], [70, 100]]
        )
        np.testing.assert_allclose(run.spec_bkgsub[0:28], 0.0, atol=1e-6)

    def test_multiple_peak_masks(self):
        # two dispersed emission lines on one axis; mask both
        n = 150
        x = np.arange(n, dtype=float)
        bkg = 5.0 + 0.02 * x
        line1 = 400.0 * np.exp(-((x - 40) ** 2) / (2 * 3.0**2))
        line2 = 300.0 * np.exp(-((x - 100) ** 2) / (2 * 3.0**2))
        run = MockRun()
        run.spec = bkg + line1 + line2
        get_step("subtract_polynomial_background")(
            run, on="spec", order=1, peak_mask=[[25, 55], [85, 115]]
        )
        result = run.spec_bkgsub
        # background between and around the two lines returns to zero
        # (atol allows for negligible Gaussian-tail leakage at the mask edges)
        np.testing.assert_allclose(result[0:20], 0.0, atol=1e-3)
        np.testing.assert_allclose(result[65:80], 0.0, atol=1e-3)
        np.testing.assert_allclose(result[125:], 0.0, atol=1e-3)
        # combined line area preserved
        np.testing.assert_allclose(
            result.sum(), (line1 + line2).sum(), rtol=1e-3
        )

    def test_2d_rows_fit_independently(self):
        n_bins, n_pix = 5, 100
        x = np.arange(n_pix, dtype=float)
        data = np.zeros((n_bins, n_pix))
        for i in range(n_bins):
            data[i] = (1.0 + i) + 0.03 * x
            data[i] += 200.0 * np.exp(-((x - 50) ** 2) / (2 * 3.0**2))
        run = MockRun()
        run.spec = data
        get_step("subtract_polynomial_background")(
            run, on="spec", order=1, peak_mask=[30, 70]
        )
        result = run.spec_bkgsub
        assert result.shape == (n_bins, n_pix)
        np.testing.assert_allclose(result[:, 0:25], 0.0, atol=1e-6)

    def test_nans_do_not_propagate(self):
        n = 80
        x = np.arange(n, dtype=float)
        spec = 5.0 + 0.1 * x
        spec[10] = np.nan
        spec[20] = np.nan
        run = MockRun()
        run.spec = spec
        get_step("subtract_polynomial_background")(run, on="spec", order=1)
        result = run.spec_bkgsub
        assert np.isnan(result[10]) and np.isnan(result[20])
        finite = result[np.isfinite(result)]
        np.testing.assert_allclose(finite, 0.0, atol=1e-6)

    def test_non_destructive(self):
        run = MockRun()
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        run.spec = original.copy()
        get_step("subtract_polynomial_background")(run, on="spec", order=1)
        np.testing.assert_array_equal(run.spec, original)
        assert hasattr(run, "spec_bkgsub")

    def test_missing_key_noop(self):
        run = MockRun()
        get_step("subtract_polynomial_background")(run, on="nope")
        assert any("not found" in s for s in run.status)

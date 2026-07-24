"""Tests for XES-specific pipeline steps."""

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
def xes_run():
    """Run with time-binned XES data ready for normalization."""
    run = MockRun()
    # Simulate time-binned XES spectrum: 20 time bins x 80 pixels
    np.random.seed(42)
    spectra = np.random.poisson(100, (20, 80)).astype(float)
    # Add a peak at pixels 30-40
    spectra[:, 30:40] += 500
    run.epix_ROI_1_simultaneous_laser_time_binned = spectra
    run.epix_ROI_1_xray_not_laser_time_binned = spectra * 0.3
    return run


class TestNormalizeXES:
    def test_normalize_full_range(self, xes_run):
        step = get_step("normalize_xes")
        step(xes_run, on="epix_ROI_1_simultaneous_laser_time_binned")
        result = getattr(
            xes_run, "epix_ROI_1_simultaneous_laser_time_binned_normalized"
        )
        assert result is not None
        # Each row should sum to 1.0
        row_sums = np.sum(result, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_normalize_pixel_range(self, xes_run):
        step = get_step("normalize_xes")
        step(
            xes_run,
            on="epix_ROI_1_simultaneous_laser_time_binned",
            pixel_range=[20, 60],
        )
        result = getattr(
            xes_run, "epix_ROI_1_simultaneous_laser_time_binned_normalized"
        )
        assert result is not None
        # Row sums over full range won't be exactly 1 since we normalized by partial range
        assert result.shape == (20, 80)

    def test_normalize_handles_zeros(self):
        run = MockRun()
        run.zero_data = np.zeros((5, 80))
        step = get_step("normalize_xes")
        step(run, on="zero_data")
        result = getattr(run, "zero_data_normalized")
        # Should not produce NaN or Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestMakeEnergyAxis:
    def test_energy_axis_basic(self):
        run = MockRun()
        run.epix_ROI_1 = np.zeros((10, 100))  # 100 pixels
        step = get_step("make_energy_axis")
        step(
            run,
            detector_key="epix_ROI_1",
            crystal_detector_distance=200.0,
            crystal_radius=250.0,
            d_spacing=1.637,
            mm_per_pixel=0.05,
            name="xes",
        )
        assert hasattr(run, "xes_energy")
        assert len(run.xes_energy) == 100
        # Energy should be monotonically increasing or decreasing
        diffs = np.diff(run.xes_energy)
        assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_energy_axis_explicit_pixels(self):
        run = MockRun()
        step = get_step("make_energy_axis")
        step(
            run,
            n_pixels=200,
            crystal_detector_distance=150.0,
            crystal_radius=300.0,
            d_spacing=1.92,
            name="cu_ka",
        )
        assert hasattr(run, "cu_ka_energy")
        assert len(run.cu_ka_energy) == 200

    def test_energy_axis_missing_params(self):
        run = MockRun()
        step = get_step("make_energy_axis")
        step(run, detector_key="nonexistent")
        # Should not crash, just log status
        assert any("missing" in s for s in run.status)


class TestPatchPixels:
    def test_patch_interpolate(self):
        run = MockRun()
        data = np.ones((10, 50))
        data[:, 25] = 1000.0  # bad pixel
        run.spectrum = data
        step = get_step("patch_pixels")
        step(run, on="spectrum", pixels=[25], mode="interpolate")
        # Patched pixel should be close to neighbors (1.0) via polynomial fit
        np.testing.assert_allclose(run.spectrum[:, 25], 1.0, atol=1e-3)

    def test_patch_zero(self):
        run = MockRun()
        data = np.ones((10, 50))
        data[:, 25] = 1000.0
        run.spectrum = data
        step = get_step("patch_pixels")
        step(run, on="spectrum", pixels=[25], mode="zero")
        np.testing.assert_allclose(run.spectrum[:, 25], 0.0)

    def test_auto_detect_spike(self):
        """auto_detect catches a bright spike column via ratio method."""
        run = MockRun()
        data = np.ones((10, 100))
        data[:, 50] = 100.0  # extreme spike (100x baseline)
        run.spectrum = data
        step = get_step("patch_pixels")
        step(run, on="spectrum", auto_detect=True, threshold=5.0)
        # Column 50 should be detected and patched
        assert 50 in run.spectrum_auto_patched_pixels
        np.testing.assert_allclose(run.spectrum[:, 50], 1.0, atol=1e-3)

    def test_auto_detect_subtle_gap(self):
        """auto_detect catches a subtle ~50% reduced column via z-score method."""
        run = MockRun()
        # Smooth signal with a narrow 2-col dip at cols 40-41
        data = np.ones((20, 200)) * 100.0
        data[:, 40] = 50.0  # 50% reduced (ratio=0.5 — below 5x threshold)
        data[:, 41] = 50.0
        run.spectrum = data
        step = get_step("patch_pixels")
        step(run, on="spectrum", auto_detect=True, threshold=5.0, nsigma=5.0)
        # These should be caught by z-score method (narrow dark cluster)
        assert 40 in run.spectrum_auto_patched_pixels
        assert 41 in run.spectrum_auto_patched_pixels

    def test_auto_detect_ignores_wide_gradient(self):
        """auto_detect does NOT flag a wide signal gradient as bad pixels."""
        run = MockRun()
        # Smooth gradient across 200 columns
        data = np.tile(np.linspace(50, 150, 200), (10, 1))
        run.spectrum = data
        step = get_step("patch_pixels")
        step(run, on="spectrum", auto_detect=True, threshold=5.0, nsigma=5.0)
        # No columns should be flagged for a smooth gradient
        assert len(run.spectrum_auto_patched_pixels) == 0

    def test_auto_detect_merges_manual(self):
        """Manual pixels are merged with auto-detected ones."""
        run = MockRun()
        data = np.ones((10, 100))
        data[:, 50] = 100.0  # will be auto-detected
        run.spectrum = data
        step = get_step("patch_pixels")
        step(run, on="spectrum", auto_detect=True, pixels=[10], threshold=5.0)
        # Both manual (10) and auto (50) should be patched
        patched = run.spectrum_auto_patched_pixels + run.spectrum_manual_patched_pixels
        assert 10 in run.spectrum_manual_patched_pixels
        assert 50 in run.spectrum_auto_patched_pixels


class TestFilterDetectorVariance:
    def test_removes_constant_pixels_2d(self):
        """Constant (zero-variance) pixels are zeroed; varying pixels kept."""
        run = MockRun()
        np.random.seed(0)
        # 50 shots x 20 pixels; columns 5 and 12 are constant (dead/hot)
        data = np.random.poisson(100, (50, 20)).astype(float)
        data[:, 5] = 42.0
        data[:, 12] = 0.0
        run.det = data
        step = get_step("filter_detector_variance")
        step(run, on="det", variance_threshold=0.0)
        result = run.det
        # Constant columns zeroed
        assert np.all(result[:, 5] == 0.0)
        assert np.all(result[:, 12] == 0.0)
        # A high-variance column survives unchanged
        assert np.any(result[:, 0] != 0.0)

    def test_mask_stored_and_shape_preserved(self):
        """Output keeps original shape; retained mask stored with pixel shape."""
        run = MockRun()
        np.random.seed(1)
        data = np.random.poisson(50, (30, 8, 8)).astype(float)
        data[:, 3, 4] = 7.0  # one constant pixel
        run.epix = data
        step = get_step("filter_detector_variance")
        step(run, on="epix", variance_threshold=0.0)
        assert run.epix.shape == (30, 8, 8)
        assert run.epix_variance_mask.shape == (8, 8)
        assert run.epix_variance_mask[3, 4] == False
        assert np.all(run.epix[:, 3, 4] == 0.0)

    def test_threshold_drops_low_variance(self):
        """A higher threshold removes low- but nonzero-variance pixels too."""
        run = MockRun()
        rng = np.random.default_rng(2)
        data = rng.normal(100.0, 10.0, (200, 10))
        data[:, 4] = rng.normal(100.0, 0.01, 200)  # tiny variance
        run.det = data
        step = get_step("filter_detector_variance")
        step(run, on="det", variance_threshold=1.0)
        assert np.all(run.det[:, 4] == 0.0)
        assert np.any(run.det[:, 0] != 0.0)

    def test_missing_key_is_noop(self):
        run = MockRun()
        step = get_step("filter_detector_variance")
        step(run, on="nonexistent")  # should not raise
        step(run)  # no 'on' key
        assert not hasattr(run, "nonexistent")

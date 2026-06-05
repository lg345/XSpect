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
        result = getattr(xes_run, "epix_ROI_1_simultaneous_laser_time_binned_normalized")
        assert result is not None
        # Each row should sum to 1.0
        row_sums = np.sum(result, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_normalize_pixel_range(self, xes_run):
        step = get_step("normalize_xes")
        step(xes_run, on="epix_ROI_1_simultaneous_laser_time_binned", pixel_range=[20, 60])
        result = getattr(xes_run, "epix_ROI_1_simultaneous_laser_time_binned_normalized")
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
        step(run,
             detector_key="epix_ROI_1",
             crystal_detector_distance=200.0,
             crystal_radius=250.0,
             d_spacing=1.637,
             mm_per_pixel=0.05,
             name="xes")
        assert hasattr(run, "xes_energy")
        assert len(run.xes_energy) == 100
        # Energy should be monotonically increasing or decreasing
        diffs = np.diff(run.xes_energy)
        assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_energy_axis_explicit_pixels(self):
        run = MockRun()
        step = get_step("make_energy_axis")
        step(run,
             n_pixels=200,
             crystal_detector_distance=150.0,
             crystal_radius=300.0,
             d_spacing=1.92,
             name="cu_ka")
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
        # Patched pixel should be average of neighbors (1.0)
        np.testing.assert_allclose(run.spectrum[:, 25], 1.0)

    def test_patch_zero(self):
        run = MockRun()
        data = np.ones((10, 50))
        data[:, 25] = 1000.0
        run.spectrum = data
        step = get_step("patch_pixels")
        step(run, on="spectrum", pixels=[25], mode="zero")
        np.testing.assert_allclose(run.spectrum[:, 25], 0.0)

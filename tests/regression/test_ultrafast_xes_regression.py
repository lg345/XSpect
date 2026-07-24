"""
Regression tests for mfxl1027922 ultrafast time-resolved XES pipeline.

Compares pipeline outputs against saved reference arrays (.npz).
If references don't exist, tests skip with instructions to generate.

Run with:
    pytest tests/regression/test_ultrafast_xes_regression.py -v
    pytest -m regression
"""

import numpy as np
import pytest

pytestmark = pytest.mark.regression


class TestUltrafastShapes:
    """Verify output array shapes match the reference."""

    def test_n_runs(self, ultrafast_reference):
        assert int(ultrafast_reference["_n_runs"]) == 3

    def test_laser_on_shape(self, ultrafast_reference):
        n_runs = int(ultrafast_reference["_n_runs"])
        for i in range(n_runs):
            key = f"run{i}_laser_on_time_binned"
            assert key in ultrafast_reference
            assert ultrafast_reference[key].shape == (40, 705)

    def test_laser_off_shape(self, ultrafast_reference):
        n_runs = int(ultrafast_reference["_n_runs"])
        for i in range(n_runs):
            key = f"run{i}_laser_off_time_binned"
            assert key in ultrafast_reference
            assert ultrafast_reference[key].shape == (40, 705)

    def test_energy_axis_shape(self, ultrafast_reference):
        assert ultrafast_reference["kbeta_energy"].shape == (705,)

    def test_time_bins_shape(self, ultrafast_reference):
        assert ultrafast_reference["time_bins"].shape == (40,)

    def test_difference_shape(self, ultrafast_reference):
        assert ultrafast_reference["difference_avg_off"].shape == (40, 705)


class TestUltrafastValues:
    """Verify numeric values against reference within tolerance."""

    def test_energy_range(self, ultrafast_reference):
        energy = ultrafast_reference["kbeta_energy"]
        assert energy.min() == pytest.approx(7022.6, abs=1)
        assert energy.max() == pytest.approx(7119.3, abs=1)

    def test_energy_monotonic(self, ultrafast_reference):
        energy = ultrafast_reference["kbeta_energy"]
        diffs = np.diff(energy)
        assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_energy_from_geometry(self, ultrafast_reference):
        """Energy axis must match the vonHamos formula with YAML parameters."""
        energy = ultrafast_reference["kbeta_energy"]
        A, R, d, mm = 50.6, 250.0, 0.895, 0.05
        gl = np.arange(705, dtype=np.float64) * mm
        ll = gl / 2.0 - (np.amax(gl) - np.amin(gl)) / 4.0
        expected = 12398.42 / (2.0 * d * np.sin(np.arctan(R / (ll + A))))
        np.testing.assert_allclose(energy, expected, rtol=1e-12)

    def test_time_bins_range(self, ultrafast_reference):
        bins = ultrafast_reference["time_bins"]
        assert bins[0] == pytest.approx(-0.9, abs=0.01)
        assert bins[-1] == pytest.approx(0.9, abs=0.01)

    def test_difference_range(self, ultrafast_reference):
        diff = ultrafast_reference["difference_avg_off"]
        assert diff.min() == pytest.approx(-0.0066, abs=0.002)
        assert diff.max() == pytest.approx(0.0036, abs=0.002)

    def test_svd_sv1_fraction(self, ultrafast_reference):
        sv1 = float(ultrafast_reference["svd_sv1_fraction"][0])
        assert sv1 == pytest.approx(0.888, abs=0.03)

    def test_laser_on_nonzero(self, ultrafast_reference):
        n_runs = int(ultrafast_reference["_n_runs"])
        for i in range(n_runs):
            data = ultrafast_reference[f"run{i}_laser_on_time_binned"]
            assert np.sum(data) > 0

    def test_laser_off_nonzero(self, ultrafast_reference):
        n_runs = int(ultrafast_reference["_n_runs"])
        for i in range(n_runs):
            data = ultrafast_reference[f"run{i}_laser_off_time_binned"]
            assert np.sum(data) > 0

    def test_bincounts_reasonable(self, ultrafast_reference):
        n_runs = int(ultrafast_reference["_n_runs"])
        for i in range(n_runs):
            key = f"run{i}_laser_on_bincount"
            if key in ultrafast_reference:
                counts = ultrafast_reference[key]
                assert np.sum(counts > 0) >= len(counts) * 0.5


class TestUltrafastReproducibility:
    """Re-run pipeline and compare against reference. Requires HDF5 data."""

    def test_laser_on_reproducible(self, ultrafast_pipeline, ultrafast_reference):
        n_runs = int(ultrafast_reference["_n_runs"])
        for i, run in enumerate(ultrafast_pipeline.analyzed_runs[:n_runs]):
            ref = ultrafast_reference[f"run{i}_laser_on_time_binned"]
            actual = run.epix_ROI_1_simultaneous_laser_time_binned
            np.testing.assert_allclose(actual, ref, rtol=1e-4, atol=1e-6,
                                       err_msg=f"Run {i} laser-on mismatch")

    def test_laser_off_reproducible(self, ultrafast_pipeline, ultrafast_reference):
        n_runs = int(ultrafast_reference["_n_runs"])
        for i, run in enumerate(ultrafast_pipeline.analyzed_runs[:n_runs]):
            ref = ultrafast_reference[f"run{i}_laser_off_time_binned"]
            actual = run.epix_ROI_1_xray_not_laser_time_binned
            np.testing.assert_allclose(actual, ref, rtol=1e-4, atol=1e-6,
                                       err_msg=f"Run {i} laser-off mismatch")

    def test_energy_reproducible(self, ultrafast_pipeline, ultrafast_reference):
        ref = ultrafast_reference["kbeta_energy"]
        actual = ultrafast_pipeline.analyzed_runs[0].kbeta_energy
        np.testing.assert_allclose(actual, ref, rtol=1e-12)

    def test_difference_reproducible(self, ultrafast_pipeline, ultrafast_reference):
        """Combined difference (avg_all_laser_off) should match."""
        p = ultrafast_pipeline
        laser_on = np.zeros((40, 705))
        laser_off = np.zeros((40, 705))
        for run in p.analyzed_runs:
            laser_on += run.epix_ROI_1_simultaneous_laser_time_binned
            laser_off += run.epix_ROI_1_xray_not_laser_time_binned

        norm_on = np.nansum(laser_on, axis=1)
        on_norm = np.divide(laser_on.T, norm_on).T
        off_sum = np.nansum(laser_off, axis=0)
        off_tiled = np.tile(off_sum, (laser_off.shape[0], 1))
        norm_factor = np.nansum(on_norm) / np.nansum(off_tiled)
        off_norm = norm_factor * off_tiled
        actual_diff = on_norm - off_norm

        ref = ultrafast_reference["difference_avg_off"]
        np.testing.assert_allclose(actual_diff, ref, rtol=5e-4, atol=1e-5)

    def test_svd_fraction_reproducible(self, ultrafast_pipeline, ultrafast_reference):
        """SVD SV1 fraction should be within tolerance."""
        p = ultrafast_pipeline
        laser_on = np.zeros((40, 705))
        laser_off = np.zeros((40, 705))
        for run in p.analyzed_runs:
            laser_on += run.epix_ROI_1_simultaneous_laser_time_binned
            laser_off += run.epix_ROI_1_xray_not_laser_time_binned

        norm_on = np.nansum(laser_on, axis=1)
        on_norm = np.divide(laser_on.T, norm_on).T
        off_sum = np.nansum(laser_off, axis=0)
        off_tiled = np.tile(off_sum, (laser_off.shape[0], 1))
        norm_factor = np.nansum(on_norm) / np.nansum(off_tiled)
        off_norm = norm_factor * off_tiled
        diff = on_norm - off_norm

        U, s, Vt = np.linalg.svd(diff[2:], full_matrices=False)
        sv1_frac = s[0] ** 2 / np.sum(s ** 2)
        ref_sv1 = float(ultrafast_reference["svd_sv1_fraction"][0])
        assert sv1_frac == pytest.approx(ref_sv1, abs=0.02)

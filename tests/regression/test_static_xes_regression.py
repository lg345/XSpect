"""
Regression tests for mfx101080524 static XES pipeline.

Compares pipeline outputs against saved reference arrays (.npz).
If references don't exist, tests skip with instructions to generate.

Run with:
    pytest tests/regression/test_static_xes_regression.py -v
    pytest -m regression
"""

import numpy as np
import pytest

pytestmark = pytest.mark.regression


class TestStaticShapes:
    """Verify output array shapes match the reference."""

    def test_n_runs(self, static_reference):
        assert int(static_reference["_n_runs"]) >= 1

    def test_per_run_spectrum_shape(self, static_reference):
        n_runs = int(static_reference["_n_runs"])
        for i in range(n_runs):
            key = f"run{i}_spectrum"
            assert key in static_reference
            assert static_reference[key].shape == (707,)

    def test_energy_axis_shape(self, static_reference):
        assert static_reference["xes_energy"].shape == (707,)

    def test_combined_spectrum_shape(self, static_reference):
        assert static_reference["combined_spectrum"].shape == (707,)


class TestStaticValues:
    """Verify numeric values against reference within tolerance."""

    def test_energy_range(self, static_reference):
        energy = static_reference["xes_energy"]
        assert energy.min() == pytest.approx(6377.2, abs=2)
        assert energy.max() == pytest.approx(6452.3, abs=2)

    def test_energy_monotonic(self, static_reference):
        energy = static_reference["xes_energy"]
        diffs = np.diff(energy)
        assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_energy_from_geometry(self, static_reference):
        """Energy axis must match vonHamos formula with YAML parameters."""
        energy = static_reference["xes_energy"]
        A, R, d, mm = 42.75, 250.0, 0.981, 0.05
        gl = np.arange(707, dtype=np.float64) * mm
        ll = gl / 2.0 - (np.amax(gl) - np.amin(gl)) / 4.0
        expected = 12398.42 / (2.0 * d * np.sin(np.arctan(R / (ll + A))))
        np.testing.assert_allclose(energy, expected, rtol=1e-12)

    def test_combined_spectrum_peak(self, static_reference):
        combined = static_reference["combined_spectrum"]
        assert combined.max() == pytest.approx(10503.8, rel=0.05)

    def test_combined_spectrum_positive(self, static_reference):
        combined = static_reference["combined_spectrum"]
        assert np.all(combined >= 0)

    def test_per_run_spectra_nonzero(self, static_reference):
        n_runs = int(static_reference["_n_runs"])
        for i in range(n_runs):
            spec = static_reference[f"run{i}_spectrum"]
            assert np.sum(spec) > 0

    def test_combined_is_sum_of_runs(self, static_reference):
        n_runs = int(static_reference["_n_runs"])
        total = np.zeros(707, dtype=np.float32)
        for i in range(n_runs):
            total += static_reference[f"run{i}_spectrum"]
        np.testing.assert_allclose(
            static_reference["combined_spectrum"], total, rtol=1e-5
        )


class TestStaticReproducibility:
    """Re-run pipeline and compare against reference. Requires HDF5 data."""

    def test_spectrum_reproducible(self, static_pipeline, static_reference):
        n_runs = int(static_reference["_n_runs"])
        for i, run in enumerate(static_pipeline.analyzed_runs[:n_runs]):
            ref = static_reference[f"run{i}_spectrum"]
            actual = run.epix_reduced_ROI_1
            np.testing.assert_allclose(actual, ref, rtol=1e-4, atol=1.0,
                                       err_msg=f"Run {i} spectrum mismatch")

    def test_energy_reproducible(self, static_pipeline, static_reference):
        ref = static_reference["xes_energy"]
        actual = static_pipeline.analyzed_runs[0].xes_energy
        np.testing.assert_allclose(actual, ref, rtol=1e-12)

    def test_combined_spectrum_reproducible(self, static_pipeline, static_reference):
        combined = np.zeros(707)
        for run in static_pipeline.analyzed_runs:
            combined += run.epix_reduced_ROI_1
        ref = static_reference["combined_spectrum"]
        np.testing.assert_allclose(combined, ref, rtol=1e-4, atol=1.0)

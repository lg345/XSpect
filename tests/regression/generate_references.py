#!/usr/bin/env python
"""
Generate regression reference .npz files from real data.

Run on S3DF where HDF5 data is accessible:
    python tests/regression/generate_references.py

Creates:
    tests/regression/references/mfxl1027922_ultrafast.npz
    tests/regression/references/mfx101080524_static.npz
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XSpect.controller.pipeline import Pipeline

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
REFERENCES_DIR = Path(__file__).parent / "references"


def generate_ultrafast():
    """Generate reference for mfxl1027922 ultrafast XES."""
    print("=" * 60)
    print("Generating ultrafast XES reference (mfxl1027922)")
    print("=" * 60)

    yaml_path = str(EXAMPLES_DIR / "mfxl1027922_ultrafast_xes.yaml")
    p = Pipeline.from_yaml(yaml_path)
    p.run(cores=4, batch_size=500)

    arrays = {}

    for i, run in enumerate(p.analyzed_runs):
        prefix = f"run{i}"

        on = getattr(run, "epix_ROI_1_simultaneous_laser_time_binned", None)
        off = getattr(run, "epix_ROI_1_xray_not_laser_time_binned", None)

        if on is not None:
            arrays[f"{prefix}_laser_on_time_binned"] = on
            print(f"  Run {run.run_number} laser-on: {on.shape}")
        if off is not None:
            arrays[f"{prefix}_laser_off_time_binned"] = off
            print(f"  Run {run.run_number} laser-off: {off.shape}")

        on_bc = getattr(run, "epix_ROI_1_simultaneous_laser_bincount", None)
        off_bc = getattr(run, "epix_ROI_1_xray_not_laser_bincount", None)
        if on_bc is not None:
            arrays[f"{prefix}_laser_on_bincount"] = on_bc
        if off_bc is not None:
            arrays[f"{prefix}_laser_off_bincount"] = off_bc

        energy = getattr(run, "kbeta_energy", None)
        if energy is not None and "kbeta_energy" not in arrays:
            arrays["kbeta_energy"] = energy
            print(f"  Energy: {energy.shape}, [{energy.min():.1f}, {energy.max():.1f}] eV")

        time_bins = getattr(run, "time_bins", None)
        if time_bins is not None and "time_bins" not in arrays:
            arrays["time_bins"] = time_bins

    # Compute combined difference using avg_all_laser_off normalization
    n_runs = len(p.analyzed_runs)
    if n_runs > 0:
        first_on = arrays["run0_laser_on_time_binned"]
        laser_on = np.zeros_like(first_on)
        laser_off = np.zeros_like(first_on)
        for i in range(n_runs):
            laser_on += arrays[f"run{i}_laser_on_time_binned"]
            laser_off += arrays[f"run{i}_laser_off_time_binned"]

        norm_on = np.nansum(laser_on, axis=1)
        on_norm = np.divide(laser_on.T, norm_on).T

        off_sum = np.nansum(laser_off, axis=0)
        off_tiled = np.tile(off_sum, (laser_off.shape[0], 1))
        norm_factor = np.nansum(on_norm) / np.nansum(off_tiled)
        off_norm = norm_factor * off_tiled

        diff = on_norm - off_norm
        arrays["difference_avg_off"] = diff
        print(f"  Difference range: [{diff.min():.6f}, {diff.max():.6f}]")

        # SVD on difference (skip first 2 bins)
        U, s, Vt = np.linalg.svd(diff[2:], full_matrices=False)
        arrays["svd_singular_values"] = s
        sv1_frac = s[0] ** 2 / np.sum(s ** 2)
        arrays["svd_sv1_fraction"] = np.array([sv1_frac])
        print(f"  SVD SV1 fraction: {sv1_frac * 100:.1f}%")

    arrays["_n_runs"] = np.array([n_runs])
    arrays["_run_numbers"] = np.array([r.run_number for r in p.analyzed_runs])

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REFERENCES_DIR / "mfxl1027922_ultrafast.npz"
    np.savez_compressed(out_path, **arrays)
    print(f"\nSaved: {out_path} ({len(arrays)} arrays)")


def generate_static():
    """Generate reference for mfx101080524 static XES."""
    print("\n" + "=" * 60)
    print("Generating static XES reference (mfx101080524)")
    print("=" * 60)

    yaml_path = str(EXAMPLES_DIR / "mfx101080524_static_xes.yaml")
    p = Pipeline.from_yaml(yaml_path)
    p.run(cores=4, batch_size=500)

    arrays = {}

    for i, run in enumerate(p.analyzed_runs):
        prefix = f"run{i}"

        spectrum = getattr(run, "epix_reduced_ROI_1", None)
        if spectrum is not None:
            arrays[f"{prefix}_spectrum"] = spectrum
            print(f"  Run {run.run_number} spectrum: {spectrum.shape}")

        energy = getattr(run, "xes_energy", None)
        if energy is not None and "xes_energy" not in arrays:
            arrays["xes_energy"] = energy
            print(f"  Energy: {energy.shape}, [{energy.min():.1f}, {energy.max():.1f}] eV")

    all_spectra = [arrays[k] for k in sorted(arrays.keys()) if k.endswith("_spectrum")]
    if all_spectra:
        combined = np.sum(all_spectra, axis=0)
        arrays["combined_spectrum"] = combined
        print(f"  Combined spectrum: {combined.shape}, peak={combined.max():.1f}")

    arrays["_n_runs"] = np.array([len(p.analyzed_runs)])
    arrays["_run_numbers"] = np.array([r.run_number for r in p.analyzed_runs])

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REFERENCES_DIR / "mfx101080524_static.npz"
    np.savez_compressed(out_path, **arrays)
    print(f"\nSaved: {out_path} ({len(arrays)} arrays)")


if __name__ == "__main__":
    generate_ultrafast()
    generate_static()
    print("\n" + "=" * 60)
    print("All references generated successfully.")
    print("=" * 60)

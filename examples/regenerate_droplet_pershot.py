"""Regenerate results/mfx101609126_droplet_pershot_xes/run<NNNN>.h5 for every run.

Runs the droplet per-shot XES pipeline (mfx101609126_droplet_pershot_xes.yaml)
over all runs listed in the YAML (78-82) and writes one HDF5 per run containing
the xray-on-filtered per-shot spectra used for stochastic RIXS:

    xrt_hproj        (N_xray, 2048)  incident XRT spectrum
    epix_seer_ROI_1  (N_xray, 620)   SEER spectrometer (epix100_1)
    epix_spec_ROI_1  (N_xray, 300)   emission (epix100_0)
    ipm              (N_xray,)       beam intensity monitor

union_shots inside the pipeline already restricts every array to xray-on shots,
so each per-run file contains only xray shots and the shot axis is aligned
across all four arrays.
"""

import os
import sys

sys.path.insert(0, "..")

import h5py
import numpy as np

from XSpect.controller.pipeline import Pipeline

YAML = "mfx101609126_droplet_pershot_xes.yaml"

# Arrays we persist per run (per-shot, xray-filtered).
SAVE_KEYS = ("xrt_hproj", "epix_seer_ROI_1", "epix_spec_ROI_1", "ipm")

pipeline = Pipeline.from_yaml(YAML)
pipeline.run(cores=1, batch_size=100000)

out_dir = pipeline.config.output.path
os.makedirs(out_dir, exist_ok=True)

for run in pipeline.analyzed_runs:
    out_file = os.path.join(out_dir, f"run{run.run_number:04d}.h5")
    with h5py.File(out_file, "w") as f:
        for key in SAVE_KEYS:
            value = getattr(run, key, None)
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
                print(f"  run {run.run_number}: saved {key} {value.shape}")
    print(f"Output written to: {out_file}\n")

print("All runs complete.")

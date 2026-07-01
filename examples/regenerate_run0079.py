"""Regenerate results/mfx101609126_pershot_xes/run0079.h5 from the YAML pipeline.

Mirrors the run cell in mfx101609126_pixel_patch_diagnostic.ipynb:
  1. Run Pipeline.from_yaml('mfx101609126_pershot_xes.yaml')
  2. Save every ndarray in pipeline.results to run<NNNN>.h5
"""

import os
import sys

sys.path.insert(0, "..")

import h5py
import numpy as np

from XSpect.controller.pipeline import Pipeline

YAML = "mfx101609126_pershot_xes.yaml"

pipeline = Pipeline.from_yaml(YAML)
pipeline.run(cores=1, batch_size=100000)

out_dir = pipeline.config.output.path
os.makedirs(out_dir, exist_ok=True)
run_num = pipeline.config.data.runs[0]
out_file = os.path.join(out_dir, f"run{run_num:04d}.h5")

with h5py.File(out_file, "w") as f:
    for key, value in pipeline.results.items():
        if isinstance(value, np.ndarray):
            f.create_dataset(key, data=value)
            print(f"  saved: {key} {value.shape}")

print(f"\nOutput written to: {out_file}")

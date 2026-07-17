# Step reference

Every registered pipeline step, grouped by what it does, with the arrays it
reads, the arrays it writes, and the parameters you set in YAML.

For the auto-generated, source-linked docstrings see
[Analysis steps (API)](../source_analysis.md). For how steps fit into a full
pipeline file, see the [YAML pipeline guide](../YAML_PIPELINE_GUIDE.md).

## The step contract

A step is a stateless function `step(run, **kwargs) -> None`. The `run` object
is a shared blackboard: a step reads attributes off it, does its work, and
writes results back as new attributes. The next step picks those up by name.
Nothing is returned; everything flows through `run`.

So in this reference:

- **Reads** = the run attributes (and their shapes) a step expects to already exist.
- **Writes** = the run attributes it sets.
- **Parameters** = the keys you pass in the YAML `step:` block. `on` is almost
  always the attribute a step operates on.

A step that can't find its input (`on` missing, or the attribute is `None`)
returns quietly and logs a status message rather than raising. This lets the
same pipeline run on a parent that has no detector loaded (batched path) and on
each batch that does.

## Array shapes

The docs use a consistent vocabulary for array shapes:

| Shape | Meaning |
|-------|---------|
| `(shots, rows, cols)` | 3D detector, one frame per shot |
| `(shots, pixels)` | 2D detector after one spatial axis is reduced |
| `(shots,)` | per-shot scalar or 1D key (IPM, delay, CCM energy) |
| `(shots,)` bool | shot mask (`xray`, `laser`, `simultaneous`) |
| `(n_time_bins, pixels)` or `(n_time_bins,)` | time-binned spectrum |
| `(n_energy, pixels)` or `(n_energy,)` | energy (CCM) binned spectrum |
| `(n_time, n_energy[, pixels])` | 2D time+energy binned |

Detector steps generally accept 2D or 3D and act on the shot axis (axis 0) or
the last (dispersion) axis; each entry says which.

## Naming conventions

Steps build output names from the input key plus a suffix describing the
operation, so a chain reads left to right:

```
epix                                                  (loaded detector, 3D)
epix_ROI_1                                             (reduce_detector_spatial)
epix_ROI_1_simultaneous_laser                         (union_shots)
epix_ROI_1_simultaneous_laser_time_binned             (reduce_detector_temporal)
epix_ROI_1_simultaneous_laser_time_binned_normalized  (normalize_xes)
```

Each step's `on` is the previous step's output name.

## All steps

| Step | Group | Reads | Writes |
|------|-------|-------|--------|
| [`load_run_keys`](loading_filtering.md#load_run_keys) | Loading | HDF5 | named per-shot keys |
| [`load_detector`](loading_filtering.md#load_detector) | Loading | HDF5 | 3D detector |
| [`get_run_shot_properties`](loading_filtering.md#get_run_shot_properties) | Loading | lightStatus | `xray`, `laser`, `simultaneous` |
| [`droplet_reconstruction`](loading_filtering.md#droplet_reconstruction) | Loading | sparse HDF5 | `new_key` 3D stack |
| [`filter_shots`](loading_filtering.md#filter_shots) | Filtering | mask + key | overwrites mask |
| [`filter_detector_adu`](loading_filtering.md#filter_detector_adu) | Filtering | detector | overwrites `on` |
| [`filter_detector_variance`](loading_filtering.md#filter_detector_variance) | Filtering | 3D detector | overwrites `on` + `_variance_mask` |
| [`hitfinding`](loading_filtering.md#hitfinding) | Filtering | 3D detector | overwrites `on` (fewer shots) |
| [`union_shots`](loading_filtering.md#union_shots) | Filtering | `on` + masks | `new_key` |
| [`separate_shots`](loading_filtering.md#separate_shots) | Filtering | `on` + 2 masks | `new_key` |
| [`common_mode_correction`](detector.md#common_mode_correction) | Detector | 3D/2D detector | overwrites `on` |
| [`patch_pixels`](detector.md#patch_pixels) | Detector | detector | overwrites `on` |
| [`find_rotation_angle`](detector.md#find_rotation_angle) | Detector | 3D/2D detector | `<on>_angle` |
| [`rotate_detector`](detector.md#rotate_detector) | Detector | 3D/2D detector | overwrites `on` |
| [`apply_roi`](detector.md#apply_roi) | Spatial | 2D/3D detector | `<on>_ROI_n` (keeps spatial) |
| [`reduce_detector_spatial`](detector.md#reduce_detector_spatial) | Spatial | 3D/2D detector | `<on>_ROI_n` (reduced) |
| [`reduce_detector_shots`](detector.md#reduce_detector_shots) | Spatial | detector | `<on>_reduced` |
| [`time_binning`](binning.md#time_binning) | Binning | timing keys | `time_bins`, `timing_bin_indices` |
| [`make_ccm_axis`](binning.md#make_ccm_axis) | Binning | `ccm` | `ccm_bins`, `ccm_energies` |
| [`ccm_binning`](binning.md#ccm_binning) | Binning | `ccm`, `ccm_bins` | `ccm_bin_indices` |
| [`bin_uniques`](binning.md#bin_uniques) | Binning | scan var | `scanvar_indices`, `scanvar_bins` |
| [`make_energy_axis`](binning.md#make_energy_axis) | Binning | detector shape | `<name>_energy` |
| [`reduce_detector_temporal`](binning.md#reduce_detector_temporal) | Binning | 1D/2D + time indices | `<on>_time_binned` |
| [`reduce_detector_ccm`](binning.md#reduce_detector_ccm) | Binning | 1D/2D/3D + ccm indices | `<on>_energy_binned` |
| [`reduce_detector_ccm_temporal`](binning.md#reduce_detector_ccm_temporal) | Binning | 1D/2D + both indices | `<on>_time_energy_binned` |
| [`normalize_xes`](spectra.md#normalize_xes) | Spectra | 1D/2D spectrum | `<on>_normalized` |
| [`subtract_polynomial_background`](spectra.md#subtract_polynomial_background) | Spectra | 1D/2D spectrum | `<on>_bkgsub` |
| [`purge_keys`](spectra.md#purge_keys) | Utility | nothing | sets keys to `None` |
| [`combine_runs`](spectra.md#combine_runs) | Reduction | per-run binned data | results dict |

# Template A — Static XES

One spectrum per sample/condition; no laser, no scan.

```yaml
experiment: {hutch: mfx, experiment_id: <exp>, lcls_run: <N>}
data:
  runs: [<run list>]
  keys:
    lightStatus/xray: xray
    lightStatus/laser: laser
    <ipm_key>: ipm
  detector_keys:
    epix100_0/ROI_area: {name: epix, transpose: true, row_range: [<lo>, <hi>]}
pipeline:
  - {step: union_shots, on: epix, filter_keys: [xray, xray], new_key: epix}   # xray-only
  - {step: filter_detector_adu, on: epix, adu_threshold: 3.0}
  - {step: hitfinding, on: epix, min_sum: 1.0}          # reject dark shots (see note)
  - {step: reduce_detector_shots, on: epix, reduction: sum, purge: true}
  - {step: patch_pixels, on: epix_reduced, auto_detect: true, mode: polynomial, axis: 1}
  - {step: reduce_detector_spatial, on: epix_reduced, rois: [[<ka_lo>, <ka_hi>], [<kb_lo>, <kb_hi>]], combine_rois: false, reduction: sum, axis: 0}
  - {step: make_energy_axis, detector_key: epix_reduced_ROI_1, n_pixels: <N>, crystal_detector_distance: <mm>, crystal_radius: <mm>, d_spacing: <ang>, mm_per_pixel: 0.05, name: kalpha}
  - {step: make_energy_axis, detector_key: epix_reduced_ROI_2, n_pixels: <N>, crystal_detector_distance: <mm>, crystal_radius: <mm>, d_spacing: <ang>, mm_per_pixel: 0.05, name: kbeta}
output: {format: hdf5, path: <experiments/<exp>/results/>}
```

`hitfinding min_sum: 1.0` rejects shots with zero signal after the ADU cut
(useful at reduced rep-rates where many shots are dark). Set `min_sum` higher for
stricter single-photon selection.

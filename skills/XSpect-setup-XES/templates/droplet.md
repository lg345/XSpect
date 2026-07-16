# Template D — Droplet / photon-counted XES

Low-flux data where per-photon reconstruction beats ADU integration. No
`detector_keys`: the image comes from the reconstruction step, which applies the
ROI on the sparse arrays (full panel never allocated). Coordinates are full-panel
raw (not transposed). See
`experiments/mfx102101026/mfx102101026_droplet_recon_xes.yaml` and the working
`examples/mfx101609126_droplet_pershot_xes.yaml`.

```yaml
pipeline:
  - {step: droplet_reconstruction, det: epix100_0, new_key: epix, roi: [<r0>, <r1>, <c0>, <c1>]}
  - {step: union_shots, on: epix, filter_keys: [xray, xray], new_key: epix}
  # NO filter_detector_adu — data are photon counts (1,2,3…), an ADU cut zeros signal
  - {step: reduce_detector_shots, on: epix, reduction: sum, purge: true}
  - {step: rotate_detector, on: epix_reduced, angle: <deg>, reshape: false}
  - {step: reduce_detector_spatial, on: epix_reduced, rois: [[<lo>, <hi>]], axis: 1, reduction: sum}
```

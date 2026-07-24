# Template C — CCM-scanned XES / RIXS

Incident energy scanned by a channel-cut mono; adds an energy axis.

```yaml
pipeline:
  - {step: make_ccm_axis, energies: [<E0>, <E1>, <n>]}
  - {step: ccm_binning, ccm_key: <ccm_setpoint>, ccm_bins_key: ccm_bins}
  - {step: reduce_detector_ccm, on: epix_ROI_1, ccm_bin_key: ccm_bin_indices}   # 1D XAS-like
  # or reduce_detector_ccm_temporal for a 2D (energy × time) RIXS plane
```

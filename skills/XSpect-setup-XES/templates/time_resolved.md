# Template B — Time-resolved / ultrafast pump-probe XES

Laser-on vs laser-off difference spectra binned by pump-probe delay. Adds timing
and splits the shots.

```yaml
pipeline:
  - {step: get_run_shot_properties}          # xray/laser/simultaneous masks
  - {step: filter_shots, on: simultaneous, filter_key: ipm, threshold: <I0min>}
  - {step: time_binning, bins: [<t0>, <t1>, <n>], lxt_key: <lxt>, fast_delay_key: <enc>, tt_correction_key: <ttc>}
  - {step: union_shots, on: epix, filter_keys: [simultaneous, laser], new_key: epix}    # pump-on
  - {step: reduce_detector_temporal, on: epix_simultaneous_laser, timing_bin_key: timing_bin_indices_simultaneous_laser}
  - {step: separate_shots, on: epix, filter_keys: [xray, laser], new_key: epix}         # pump-off reference
  - {step: reduce_detector_temporal, on: epix_xray_not_laser, timing_bin_key: ...}
reduction:
  - {step: combine_runs, detector_key: epix_ROI_1, laser_on_suffix: simultaneous_laser, laser_off_suffix: xray_not_laser}
```

# Example configs

Complete pipeline YAML for each analysis mode, pulled live from the
[`examples/`](https://github.com/lg345/XSpect/tree/master/examples) directory.
Each one runs end to end with `Pipeline.from_yaml(...)`. Read the
[YAML pipeline guide](YAML_PIPELINE_GUIDE.md) for the section-by-section
reference and the [step reference](steps/index.md) for what each step does.

## Static XES

Single-state emission spectra, no time or energy axis. Loads the ePix, filters
to x-ray shots, patches dead columns, rotates the dispersion axis, and sums to
one spectrum per run. `examples/mfx101080524_static_xes.yaml`.

```yaml
--8<-- "examples/mfx101080524_static_xes.yaml"
```

## Time-resolved (pump-probe) XES

Laser-on minus laser-off emission, binned by pump-probe delay. Adds
`time_binning` and the laser masks, then `reduce_detector_temporal` and the
`combine_runs` reduction to build the transient.
`examples/mfxl1027922_ultrafast_xes.yaml`.

```yaml
--8<-- "examples/mfxl1027922_ultrafast_xes.yaml"
```

## 2D XAS (energy × delay)

Simultaneous incident-energy scan and pump-probe delay, producing a transient
absorption map Δμ(E, t). Uses `make_ccm_axis`, `ccm_binning`, `time_binning`,
and `reduce_detector_ccm_temporal`. `examples/xcs101591326_2d_xas.yaml`.

```yaml
--8<-- "examples/xcs101591326_2d_xas.yaml"
```

## Temporal XAS

Fluorescence-detected XAS at a fixed incident energy, scanned over delay
(I_f / I_0). `examples/xcs101591326_temporal_xas.yaml`.

```yaml
--8<-- "examples/xcs101591326_temporal_xas.yaml"
```

## Droplet / photon-counting XES

Per-shot XES from the MFX droplet2photon sparse layout. `droplet_reconstruction`
rebuilds dense frames from photon positions before the usual XES chain.
`examples/mfx101609126_droplet_pershot_xes.yaml`.

```yaml
--8<-- "examples/mfx101609126_droplet_pershot_xes.yaml"
```

## RIXS

Incident-energy scan with 2D emission images, giving a RIXS plane (incident
energy from DCCM, emission energy from the von Hamos spectrometer).
`examples/mfx101609126_static_rixs.yaml`.

```yaml
--8<-- "examples/mfx101609126_static_rixs.yaml"
```

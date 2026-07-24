# mfx100895324 — Fe Kα / Kβ XES redox series

Static (non-time-resolved) X-ray emission spectroscopy of aqueous iron
cyanide complexes at various oxidation states. Reference data for upcoming
experiment **mfx102101026**.

## Chemistry

A ferricyanide → ferrocyanide redox series of prepared mixtures:

- **Fe(III)** = ferricyanide, [Fe(CN)₆]³⁻ (oxidized, `reduced_fraction = 0`)
- **Fe(II)**  = ferrocyanide, [Fe(CN)₆]⁴⁻ (reduced, `reduced_fraction = 1`)
- Mixtures at nominal 50:50, 20:80, and 10:90 (Fe(II):Fe(III))

`reduced_fraction` is the nominal Fe(II) (ferrocyanide) fraction.

## Instrument

Both emission lines are recorded on the **same detector** (`epix100_1`,
shape `(nshots, 300, 710)`) at different cross-dispersion rows:

| Line | Analyzer | d (Å) | Detector rows | ROI key |
|------|----------|-------|---------------|---------|
| Fe Kα | LiNbO₃ (2 3 -4) | 0.981  | 228–248 | `epix_reduced_ROI_1` |
| Fe Kβ | Ge (6 2 0)      | 0.8946 | 158–180 | `epix_reduced_ROI_2` |

- Dispersion axis = columns; cross-dispersion axis = rows.
- No optical laser (`lightStatus/laser == 0`) — purely static.

## Runs

The **active analysis set is runs 36–54** (Shift-1 prepared-mixture
calibration series). Full run/sample mapping with OCV values is in
[`runs.csv`](runs.csv). The **complete beamtime record** for both shifts —
including the Shift-2 in-situ chronoamperometry series (runs 56–114) — is in
[`BEAMTIME_MANIFEST.md`](BEAMTIME_MANIFEST.md). Summary of the analysis set:

| Runs | Sample | Nominal Fe(II) | Role |
|------|--------|---------------|------|
| 36, 37, 38 | Fe(III) ferricyanide | 0.00 | **reference** (36) / oxidized |
| 39, 40, 41, 42 | Fe(II) ferrocyanide | 1.00 | reduced end-member |
| 43, 44, 45, 46 | 50:50 mix | 0.50 | mixture |
| **47** | **Fe foil** | — | **excluded** (metallic calibration standard) |
| 48, 49, 50 | 20:80 mix | 0.20 | mixture |
| 51, 52, 53, 54 | 10:90 mix | 0.10 | mixture |

Shot counts vary widely (full runs ~28.8k; short runs: 38≈1141, 42≈2247,
46≈3769, 47≈14416, 50≈21219, 52≈18988, 54≈133). Analysis is
area-normalized so short runs remain comparable (though noisier).

> **Run 47 is metallic Fe foil**, not a cyanide sample. It is excluded from
> all titration / deviation analysis and only shown as a sanity-check outlier.

## Files

- **`mfx100895324_static_xes.yaml`** — XSpect pipeline: per-shot ADU filter →
  sum → bad-pixel patch → two row-ROIs (Kα, Kβ) → energy axes. Processes
  runs 36–54. Reads existing smalldata read-only; does not modify it.
- **`mfx100895324_static_xes_visualization.ipynb`** — per-run visualization
  plus redox-deviation analysis:
  1. Detector diagnostic
  2. Fe Kα per-run grid
  3. Fe Kβ per-run grid
  4. Per-run overlay
  5. Run-to-run stability (peak centroid vs run)
  6. **IAD** — integrated absolute difference (% spectral change vs reference)
  7. **LCF** — two-component linear-combination fit → fraction reduced,
     validated against nominal composition
  8. **Pointwise % deviation** vs energy
  9. Optional HDF5 save
- **`runs.csv`** — machine-readable manifest for the analysis set (runs 36–54;
  run, sample, nominal Fe(II) fraction, OCV, role). Loaded by the notebook.
- **`BEAMTIME_MANIFEST.md`** — full run log for both shifts (all ~114 runs),
  including the Shift-2 in-situ electrochemistry series (future analysis).

## Running

From this directory:

```bash
cd /sdf/data/lcls/ds/mfx/mfx100895324/results/lbgee/XSpect/experiments/mfx100895324
jupyter lab mfx100895324_static_xes_visualization.ipynb
```

The notebook adds the XSpect repo root to `sys.path` automatically, so it
works from this subdirectory without installing the package.

## Energy calibration caveat

The `crystal_detector_distance` values in the YAML are **starting estimates**.
The stability plot shows a ~47 eV Kβ offset from the tabulated line — refine
`crystal_detector_distance` against a known reference before quoting absolute
energies. Relative (deviation) metrics are unaffected.

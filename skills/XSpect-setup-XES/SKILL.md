---
name: XSpect-setup-XES
description: >-
  Set up a new X-ray Emission Spectroscopy (XES) experiment for analysis with
  XSpect at LCLS. Use when a user wants to start analyzing a new XES beamtime:
  configuring the LUTE smalldata pipeline, writing the LUTE YAML, writing the
  XSpect pipeline YAML, and creating diagnostic + analysis notebooks. Handles
  static XES, time-resolved (ultrafast pump-probe) XES, and CCM-scanned XES.
  Trigger on: "set up a new XES experiment", "new von Hamos experiment",
  "configure smalldata for epix XES", "make an XSpect YAML", "analyze Fe Kbeta",
  "pump-probe XES pipeline".
---

# XSpect: Set Up a New XES Experiment

You are helping a scientist stand up the analysis for a new XES beamtime at
LCLS. The end state is: (1) a working LUTE smalldata pipeline that produces the
per-shot HDF5, (2) a correct XSpect pipeline YAML, and (3) diagnostic and
analysis notebooks. Work through the phases below **in order**, asking the user
the decision questions before writing any files.

Prefer reading the authoritative references over guessing:
- `docs/YAML_PIPELINE_GUIDE.md` — the full list of registered pipeline steps and YAML sections.
- `XSpect/analysis/*.py` — the actual step implementations (source of truth for parameter names).
- `experiments/mfx102101026/` — a complete worked static-XES example (YAML + notebooks).
- The `ask-lute` skill — for LUTE task/config/SLURM details.

---

## Phase 0 — Determine the experiment type (ASK THE USER)

Before anything else, establish what kind of XES this is. Ask:

1. **What is being measured?**
   - **Static XES** — one spectrum per sample/condition; no laser, no scan. (e.g. redox speciation, reference standards.)
   - **Time-resolved / ultrafast pump-probe XES** — laser-on vs laser-off difference spectra binned by pump-probe delay.
   - **CCM-scanned XES / RIXS** — incident energy scanned by a channel-cut mono (adds an energy axis).
   - **Droplet / photon-counted XES** — low-flux data where per-photon reconstruction beats ADU integration.

2. **Experiment identity:** hutch (mfx/xcs/xpp/...), experiment_id, LCLS run number, and the run numbers to analyze.

3. **Detector(s):** which area detector holds the emission (e.g. `epix100_0`, `epix_1`, `Epix10k2M`)? Are there multiple (e.g. a spectrometer + a SEER incident monitor)?

4. **Spectrometer geometry:** von Hamos? Which analyzer crystal(s), radius, d-spacing? Which emission line(s) (Fe Kα ~6404 eV, Fe Kβ ~7058 eV, etc.)? Is the dispersion along detector rows or columns?

5. **Diagnostics available:** IPM/beam-intensity key, timetool (for time-resolved), CCM setpoint key (for scans), laser delay stage + encoder keys.

Record the answers — they drive every later choice. If the user doesn't know
the detector/geometry yet, that's fine: the diagnostic notebook (Phase 4) is
built precisely to discover ROIs and geometry from the first run.

---

## Phase 1 — LUTE smalldata pipeline

The smalldata HDF5 must exist before XSpect can run. This is produced by LUTE's
`SubmitSMD` task, driven by a `mfx_lute.yaml` (or hutch-appropriate) config plus
a smalldata producer template.

### 1a. Locate / create the LUTE config

LUTE configs live in the **experiment data area**, not the XSpect repo:
```
/sdf/data/lcls/ds/<hutch>/<exp>/results/lute_output/mfx_lute.yaml
```
If the user already has one from the DAQ/ops team, edit it. Otherwise consult
the `ask-lute` skill to scaffold one.

### 1b. Configure the detector, ROIs, and diagnostics

In the `SubmitSMD`/producer section, set (names match `smd_producer.py`):

```yaml
    detnames: ["epix100_0"]           # emission detector(s)
    getROIs:
      epix100_0:
        - ROI: [[0, 704], [0, 768]]   # full panel initially; tighten after diagnostics
          writeArea: true
          thresADU: null
    epicsPV: [...]                     # IPM, CCM setpoints, delay stage, timetool
    epicsArchFilePV: [...]             # von Hamos crystal motor RBVs, etc.
    detSumAlgos:
      epix100_0: ["calib", "calib_max"]
```

**Common-mode gotcha (ePix100):** do NOT pass `getDetParams: {cmpars: [...]}` for
ePix100 unless you know the pedestals are deployed and 3D. Passing a `cmpars`
tuple sends the data through psana `common_mode_apply`, which indexes 3D; if the
detector returns a 2D `(704,768)` frame it raises
`IndexError: too many indices for array`, and the detector silently produces no
data. Leave `getDetParams`/`cmpars` out (common_mode defaults to the safe path).

### 1c. Droplet / photon-counting (only if Phase 0 = droplet XES)

Two distinct smalldata products:

- `getDropletParams` → `droplet_sparse_*` keys = per-droplet **centroid + total
  ADU + npix**. This is a **diagnostic/calibration** product (hit-rate QA,
  droplet-size studies, per-droplet ADU spectrum to calibrate the single-photon
  energy). NOT a photon image.
- `getDroplet2Photons` → `droplet_droplet2phot_sparse_*` keys = per-**photon**
  hits (`data` = photon count). This is what the XSpect `droplet_reconstruction`
  step reads for true photon-counted imaging.

```yaml
    getDroplet2Photons:
      epix100_0:
        droplet: {threshold: 5, thresholdLow: 5, thresADU: 60, useRms: true}
        aduspphot: 95      # single-photon TOTAL ADU (measure from droplet spectrum!)
        nData: 100000
        cputime: true
```
`aduspphot` is in the detector's native units (ADU for an ADU-calibrated ePix,
keV for a keV-calibrated one). Measure it from the per-droplet ADU histogram: the
first peak is one photon. **Known LUTE gap (see XSpect issue #97):** the
`smd2_prod_config_template.py` may hardcode `aduspphot`/`nData` and ignore the
YAML — verify the rendered `prod_config` actually uses your values, and check the
LUTE `Droplet2PhotonParams` model default (`aduspphot: 162`) is not silently used.

### 1d. Submit and verify

Submit via LUTE (`ask-lute` skill for SLURM/ARP details). Then verify the HDF5:
```python
import h5py
f = h5py.File("/sdf/data/lcls/ds/<hutch>/<exp>/hdf5/smalldata/<exp>_Run####.h5")
list(f["epix100_0"].keys())   # expect ROI_area (+ droplet_* if configured)
```
Sanity-check that the detector actually has signal (not all-zero / no-beam) and
confirm the frame shape `(nshots, rows, cols)`.

---

## Phase 2 — Determine geometry from the data (before the XSpect YAML)

Do NOT guess ROIs. Sum a few thousand xray-on frames and inspect:

```python
import h5py, numpy as np
f = h5py.File(SMD_FILE)
xray = f["lightStatus/xray"][:].astype(bool)
frames = f["epix100_0/ROI_area"][:3000][xray[:3000]]
img = np.where(frames > 3, frames, 0).sum(0)     # (rows, cols)
row_prof, col_prof = img.sum(1), img.sum(0)
```
- The axis with a **broad** signal spread = **dispersion** (spectral) axis.
- The axis with a **narrow** band = **cross-dispersion** axis; the band position
  and width define the ROI.
- Multiple emission lines (Kα, Kβ) appear as separate cross-dispersion bands.

Decide whether a `transpose` is needed so the geometry matches your convention
(XSpect steps treat detector arrays as `(shots, rows, cols)`; ROIs in
`reduce_detector_spatial` select along the chosen axis).

---

## Phase 3 — Write the XSpect pipeline YAML

Live in the repo under `experiments/<exp>/<exp>_<type>_xes.yaml`. Five sections
(`experiment`, `data`, `pipeline` required; `reduction`, `output` optional). See
`docs/YAML_PIPELINE_GUIDE.md` for the complete step list and
`experiments/mfx102101026/mfx102101026_static_xes.yaml` for a full example.

Read the actual step signatures in `XSpect/analysis/*.py` before using a step —
do not invent parameter names.

### Memory: `row_range` import-time crop

For large detectors, crop the cross-dispersion axis **at import** so the full
frame is never materialized (prevents OOM in batched runs):
```yaml
  detector_keys:
    epix100_0/ROI_area:
      name: epix
      transpose: true
      row_range: [130, 280]   # in the (post-transpose) frame; slices at HDF5 read time
```
ROIs in later `reduce_detector_spatial` steps stay in **absolute** (full-frame)
coordinates — XSpect auto-translates them to the cropped frame using the recorded
offset. (Implemented in `model/run.py`, `controller/batch_manager.py`,
`analysis/spectroscopy.py`.)

### Template A — Static XES

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

### Template B — Time-resolved / ultrafast pump-probe XES

Add timing and split laser-on / laser-off:
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

### Template C — CCM-scanned XES / RIXS

```yaml
pipeline:
  - {step: make_ccm_axis, energies: [<E0>, <E1>, <n>]}
  - {step: ccm_binning, ccm_key: <ccm_setpoint>, ccm_bins_key: ccm_bins}
  - {step: reduce_detector_ccm, on: epix_ROI_1, ccm_bin_key: ccm_bin_indices}   # 1D XAS-like
  # or reduce_detector_ccm_temporal for a 2D (energy × time) RIXS plane
```

### Template D — Droplet / photon-counted XES

No `detector_keys` — the image comes from the reconstruction step, which applies
the ROI **on the sparse arrays** (full panel never allocated). Coordinates are
full-panel raw (not transposed). See
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

### Validate the YAML parses

```python
from XSpect.controller.config_parser import parse_yaml
cfg = parse_yaml("experiments/<exp>/<exp>_<type>_xes.yaml")
print([s.step for s in cfg.pipeline])
```

---

## Phase 4 — Diagnostic + analysis notebooks

Create notebooks in `experiments/<exp>/`. Model them on the mfx102101026 set:

1. **`<exp>_<type>_xes_visualization.ipynb`** — runs the pipeline and shows:
   - A **detector diagnostic** cell: summed 2D image + row/col projections with
     the ROI bands overlaid, so the user can verify/refine ROIs against real data.
     Include a `DIAG_RUN` knob. **This is the first thing to run** — it closes the
     loop with Phase 2 and lets the user fix ROIs before trusting spectra.
   - Per-run spectra, energy axes, shot counts.
   - A Save cell writing per-run results to `results/`.

2. **`<exp>_aggregate_xes.ipynb`** (multi-run) — reads saved per-run HDF5 (no
   pipeline re-run), interpolates onto a common energy grid, combines, and shows
   cumulative buildup / per-run overlay / stability.

3. **`<exp>_droplet_visualization.ipynb`** (droplet experiments) — droplet-level
   diagnostics (Sections: droplets/shot, ADU spectrum, hit map, occupancy) PLUS a
   photon-counted reconstruction section using the native `droplet_reconstruction`
   step. Stream sparse arrays in contiguous slabs reading only the first ~600
   columns (avoid the full `(nshots, 100000)` read that hangs the kernel).

4. **`<exp>_speciation_simple.ipynb`** (optional, for non-experts) — user lists
   runs; runs are pipeline-processed only if not already cached per-run; then
   combine → baseline → normalize → speciation vs a reference, with an eLog
   beamline-summary section (write `report.html` under
   `/sdf/data/lcls/ds/<hutch>/<exp>/stats/summary/<category>/` — surfaced by the
   eLog "Summaries" tab, no auth needed).

### Notebook conventions
- Add the repo root to `sys.path` so `from XSpect import Pipeline` works without install:
  ```python
  REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', '..'))
  sys.path.insert(0, REPO_ROOT)
  ```
- Enable pipeline logging so long runs aren't silent: `from XSpect import enable_logging; enable_logging()` (or `enable_logging(log_file=...)`). Omit if the user wants a quiet notebook.
- To process a single run without editing the YAML, override in memory:
  ```python
  from dataclasses import replace
  p = Pipeline.from_yaml(YAML)
  p.config = replace(p.config, data=replace(p.config.data, runs=[run]))
  p.run(cores=..., batch_size=...)
  ```
- Cache per-run results (`run_<N>_static_xes.h5`) so re-running the same run is instant.

---

## Phase 5 — Track down open items

- Keep a `runs.csv` in `experiments/<exp>/` recording each run's sample, role
  (e.g. `excluded` for no-beam runs), and composition — the analysis notebooks
  read it.
- If a step or LUTE feature is missing (e.g. the droplet2photon template gap),
  open a GitHub issue on the branch documenting exactly what must be implemented,
  as in XSpect issue #97.

---

## Guardrails

- **Never invent step names or parameters.** Read `docs/YAML_PIPELINE_GUIDE.md`
  and the `XSpect/analysis/*.py` source.
- **Verify units.** Droplet/ADU vs keV mistakes silently zero all signal. The
  minimum droplet value equals `thresADU`; a single-photon peak tells you the
  ADU-per-photon.
- **Verify the transpose/ROI coordinate frame.** Droplet sparse coords are raw
  (un-transposed); the XES `ROI_area` path may be transposed. Confirm which frame
  each ROI is expressed in before trusting a spectrum.
- **Confirm the run actually has beam** before debugging analysis — check IPM and
  whether frames are non-zero.
- Ask the user for geometry/threshold values you cannot measure; do not hardcode
  guesses without flagging them as TODO in the YAML comments.

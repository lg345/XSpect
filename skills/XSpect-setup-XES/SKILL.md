---
name: XSpect-setup-XES
description: >-
  Set up a new X-ray Emission Spectroscopy (XES) experiment for analysis with
  XSpect at LCLS, from LUTE smalldata through the pipeline YAML and notebooks.
  Handles static, time-resolved (ultrafast pump-probe), CCM-scanned, and droplet
  photon-counted XES. Trigger on: "set up a new XES experiment", "new von Hamos
  experiment", "configure smalldata for epix XES", "make an XSpect YAML",
  "analyze Fe Kbeta", "pump-probe XES pipeline".
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

3. **Detector(s):** which area detector holds the emission? (See Phase 0.5 for
   how to identify it from the HDF5.) Are there multiple (e.g. a spectrometer +
   a SEER incident monitor)?

4. **Spectrometer geometry:** von Hamos? Which analyzer crystal(s), radius, d-spacing? Which emission line(s) (Fe Kα ~6404 eV, Fe Kβ ~7058 eV, etc.)? Is the dispersion along detector rows or columns?

5. **Diagnostics available:** IPM/beam-intensity key, timetool (for time-resolved), CCM setpoint key (for scans), laser delay stage + encoder keys.

Record the answers — they drive every later choice. If the user doesn't know
the detector/geometry yet, that's fine: the diagnostic notebook (Phase 4) is
built precisely to discover ROIs and geometry from the first run.

---

## Phase 0.5 — Identify the HDF5 keys (do this early, ask when ambiguous)

Never guess dataset paths. Open the smalldata file and enumerate the keys, then
map them to roles. **If more than one candidate fits a role, ask the user which
to use — do not pick silently.**

```python
import h5py
f = h5py.File(SMD_FILE, "r")
top = list(f.keys())                                  # detectors + diagnostic groups
print("top-level:", top)
for g in top:                                          # inspect a detector/group
    if isinstance(f[g], h5py.Group):
        print(g, "->", list(f[g].keys()))
```

### Role → naming conventions (LCLS)

**Emission / spectroscopy detector (the XES signal).**
Almost always an **ePix100**. The group name varies by experiment:
`epix100_0`, `epix100_1`, `epix_0`, `epix_1`, `epix_alc_0`, `epix_ladm_1`, and
similar variants. The per-shot 2D data is under `<det>/ROI_area` (or a numbered
ROI like `<det>/ROI_0_area`). If there are two ePix100s, one is often the
spectrometer and the other a **SEER** incident-energy monitor — **ask which is
which**; don't assume `_0` is the spectrometer.

**Scattering / imaging detector (usually NOT the XES signal).**
Large-area **`epix10k2m`** (a.k.a. Epix10k2M) or **`jungfrau16m`** (Jungfrau
16M). If one of these is present alongside an ePix100, the ePix100 is almost
certainly the emission detector; confirm with the user before excluding the big
detector.

**Normalization / beam-intensity monitor ("IPM").**
Standard hutches use `ipm<N>/sum` (e.g. `ipm4/sum`, `ipm5/sum`) plus
`ipm<N>/xpos`, `ipm<N>/ypos`. **MFX uses non-standard names** — commonly a beam
monitor like `MfxDg2BmMon/totalIntensityJoules` (also `MfxDg1BmMon/...`,
`MfxDg2Imp/...`). If you see an `MfxDg*BmMon` group, that's the I0 monitor. When
several intensity-like keys exist, ask which the user normalizes to.

**Light-status masks (shot classification).** `lightStatus/xray`,
`lightStatus/laser` — used to build xray / laser / simultaneous masks.

**Timing (time-resolved only).** Laser delay stage (`lxt`, `lxt_ttc`, `enc/...`
encoder) and timetool correction (`tt/ttCorr`, `tt/FLTPOS_PS`, or similar). Names
vary a lot — enumerate and ask.

**Scan variable (CCM / RIXS).** Channel-cut mono setpoint, often `ccm/E`,
`epics/ccmE`, `scan/ccm_E`, or an EPICS-archived PV. Confirm which key holds the
incident energy.

**Detector geometry PVs (von Hamos).** Crystal motor readbacks appear as
archived EPICS PVs (e.g. `MFX:SPEC:C1:TILT.RBV`, `...:X.RBV`, `...:ROT.RBV`).
Useful for recording geometry; not required for the pipeline.

### Ambiguity protocol

When a role has 0 or >1 clear candidates:
- **0 candidates** — tell the user the role's data seems absent; ask for the key
  or whether that step should be skipped (e.g. no timetool ⇒ no time-resolved).
- **>1 candidates** — list them with their shapes/dtypes and ask the user to pick.
- Echo back the final key→role mapping for confirmation before writing any YAML.

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

### Pick the template for the Phase 0 experiment type

Read the one matching template, fill the `<...>` placeholders from the recorded
answers and measured geometry, and skip the rest:

- **Static XES** → [`templates/static.md`](templates/static.md)
- **Time-resolved / ultrafast pump-probe** → [`templates/time_resolved.md`](templates/time_resolved.md)
- **CCM-scanned XES / RIXS** → [`templates/ccm.md`](templates/ccm.md)
- **Droplet / photon-counted** → [`templates/droplet.md`](templates/droplet.md)

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

The failure modes that silently produce wrong or empty spectra, each with its
canonical rule earlier in the skill:

- **HDF5 keys** — enumerate and map to roles, ask when ambiguous (Phase 0.5).
- **Step names / parameters** — read the guide and `XSpect/analysis/*.py` source
  (Phase 3), never invent them.
- **Units** — droplet/ADU vs keV mistakes zero all signal. Minimum droplet value
  equals `thresADU`; the single-photon peak gives ADU-per-photon (Phase 1c).
- **Transpose / ROI frame** — droplet sparse coords are raw; the `ROI_area` path
  may be transposed. Confirm which frame each ROI uses (Phases 2, 3).
- **Beam present** — check IPM and non-zero frames before debugging analysis
  (Phase 1d).
- **Unmeasurable values** — ask for geometry/thresholds you cannot measure; flag
  any guess as a TODO in the YAML comments.

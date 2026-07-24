# XSpect Domain Glossary

## Core Entities

**Experiment** — An LCLS beamtime allocation identified by hutch + experiment_id + LCLS run number. Maps to a directory on S3DF containing smalldata HDF5 files.

**Run** — A numbered data collection within an experiment. Each run produces one HDF5 file containing shot-by-shot detector and diagnostic data. Not the same as an "LCLS run" (which is the beamtime allocation).

**Shot** — A single XFEL pulse event. Each shot produces one detector frame and associated diagnostic scalar values. Shots are classified by light status (xray, laser, or simultaneous).

**Shot Mask** — A boolean array over all shots in a run indicating which shots satisfy a given condition (e.g., `xray`, `laser`, `simultaneous`, or a filtered subset).

**Detector Key** — An HDF5 dataset path for a 2D area detector (e.g., ePix). Large, loaded per-batch with memory management.

**Diagnostic Key** — An HDF5 dataset path for a 1D scalar array (e.g., IPM intensity, encoder position, time tool correction). Small, loaded immediately.

## Analysis Concepts

**Pipeline** — An ordered list of analysis steps applied to each shot batch. Defined in YAML. Executed in order, no conditionals.

**Step** — A single registered analysis operation that reads from and writes to `run.results`. Stateless. All parameters from YAML.

**Reduction** — A post-pipeline operation that receives results from all batches/runs and produces aggregate outputs (e.g., summed spectra, normalized differences).

**Results Dict** — Flat dictionary on the run object (`run.results`) holding all intermediate and final data products. Keys are dot-separated strings.

**Batch** — A contiguous range of shots within a run, processed as a unit on one core. The controller splits runs into batches for parallelism and reconverges results afterward. Batch size is an operational parameter, not a scientific one.

## Instrument Concepts

**von Hamos Geometry** — A dispersive X-ray spectrometer geometry using a cylindrically bent crystal. Defines the mapping from detector pixel position to photon energy via crystal d-spacing, crystal radius, and detector distance.

**ROI (Region of Interest)** — A pixel range on the detector selected for analysis. Multiple ROIs can be defined and either combined or processed separately.

**ADU Threshold** — Analog-to-digital unit cutoff applied to detector pixels. Pixels below the threshold are zeroed to remove electronic noise.

**CCM (Channel-Cut Monochromator)** — Upstream monochromator that selects incident X-ray energy for XAS measurements. Defines the energy axis for absorption spectroscopy.

## Shot Classification

**xray** — Shots where the X-ray beam was on (from `lightStatus/xray`).

**laser** — Shots where the optical laser was on (from `lightStatus/laser`).

**simultaneous** — Shots where both xray AND laser were on. Used for pump-probe "laser-on" signal.

**xray_not_laser** — Shots where xray was on but laser was off. Used for pump-probe "laser-off" reference.

## Data Reduction Operations

**Union** — Select shots from a data array where multiple masks are simultaneously true (logical AND). E.g., "give me epix frames that are both simultaneous and laser."

**Separate** — Select shots where one mask is true and another is false (A AND NOT B). E.g., "give me epix frames that are xray but NOT laser."

**Temporal Reduction** — Bin shot-level data into time delay bins using timing diagnostics (laser delay stage + encoder + time tool correction).

**Spatial Reduction** — Collapse a spatial detector dimension by summing within ROIs.

**CCM Reduction** — Bin shot-level data into incident energy bins defined by monochromator positions.

---

## Architecture Overhaul: YAML-Driven Pipeline (Issue #82)

### Goal

Replace the 6 controller subclasses (`XESBatchAnalysis`, `XESBatchAnalysisRotation`, `XASBatchAnalysis`, `XASBatchAnalysis_1D_ccm`, `XASBatchAnalysis_1D_time`, `ScanAnalysis_1D`, `ScanAnalysis_1D_XES`) with a single generic `Pipeline` class that dispatches analysis steps from a YAML recipe. One YAML file = one reproducible analysis workflow.

### Target Module Layout

```
XSpect/
├── __init__.py
├── model/
│   ├── experiment.py        # experiment, spectroscopy_experiment
│   ├── run.py               # spectroscopy_run (results dict interface)
│   └── von_hamos.py         # vonHamos crystal geometry
├── analysis/
│   ├── registry.py          # @register_step, @register_reduction, dispatch
│   ├── spectroscopy.py      # base operations (filter, union, separate, reduce)
│   ├── xes.py               # XES-specific steps (normalize, energy axis, combine)
│   └── xas.py               # XAS-specific steps (ccm axis, ccm binning)
├── controller/
│   ├── config_parser.py     # YAML parsing + validation
│   ├── pipeline_runner.py   # step dispatch loop + reduction orchestration
│   └── batch_manager.py     # shot chunking, multiprocessing, reconvergence
├── visualization/
├── diagnostics/
└── postprocessing/
```

### Phase Structure (with dependencies)

```
Phase 1: MVP (Static XES end-to-end)
│
├── 1A: Core Infrastructure (#84)
│   ├── model/run.py          — spectroscopy_run with self.results = {}
│   ├── model/experiment.py   — move experiment classes as-is
│   ├── analysis/registry.py  — @register_step, @register_reduction, dispatch
│   ├── controller/config_parser.py  — YAML parse + section validation
│   ├── controller/pipeline_runner.py — dispatch loop (pipeline then reductions)
│   ├── controller/batch_manager.py  — shot chunking + Pool + reconvergence
│   └── Pipeline class         — from_yaml(), run(), results
│   Dependencies: None (foundational)
│
├── 1B: Register Static XES Steps (#85)
│   ├── filter_shots           — threshold filter on diagnostic key
│   ├── union_shots            — combine masks (AND)
│   ├── separate_shots         — exclude masks (A AND NOT B)
│   ├── filter_detector_adu    — zero pixels below threshold
│   ├── reduce_detector_shots  — sum detector across shot dimension
│   ├── reduce_detector_spatial — ROI reduction of spatial dimension
│   ├── apply_roi              — apply ROI mask
│   ├── rotate                 — scipy.ndimage.rotate wrapper
│   ├── patch_pixels           — bad pixel interpolation
│   └── hit_finding            — event detection + filter
│   Dependencies: 1A (registry must exist)
│
└── 1C: Static XES Integration (#86)
    ├── Example YAML config (xcsp23820_static.yaml)
    ├── model/von_hamos.py     — move vonHamos class
    ├── Numerical validation   — old path vs new path on same data
    └── combine_runs reduction — basic cross-run summation
    Dependencies: 1A + 1B (all steps registered, pipeline runnable)

Phase 2: Time-Resolved XES (#87)
│
├── New pipeline steps:
│   ├── time_binning           — bin shots by delay stage + encoder + TT
│   ├── reduce_detector_temporal — bin detector into time bins
│   ├── normalize_xes          — area-normalize XES spectra
│   ├── make_energy_axis       — energy from von Hamos geometry
│   └── droplet_reconstruction — sparse-to-dense detector reconstruction
│
├── New reduction steps:
│   ├── combine_runs (full)    — sum across runs with uncertainty propagation
│   ├── normalize_combined     — normalize laser-on/off
│   └── compute_difference     — normalized difference spectra
│
└── Dependencies: Phase 1 complete (1C passes validation)

Phase 3: XAS Pipelines (#88)
│
├── New pipeline steps:
│   ├── make_ccm_axis          — CCM energy bins from setpoints
│   ├── ccm_binning            — digitize shots into CCM bins
│   ├── reduce_detector_ccm    — 1D XAS (energy only)
│   ├── reduce_detector_ccm_temporal — 2D XAS (energy + time)
│   └── bin_uniques            — bin by arbitrary scan variable
│
├── Workflows replaced:
│   ├── XASBatchAnalysis       → 2D YAML
│   ├── XASBatchAnalysis_1D_ccm → energy-only YAML
│   └── XASBatchAnalysis_1D_time → time-only YAML
│
└── Dependencies: Phase 2 complete (time_binning shared with XAS time-resolved)

Phase 4: Cleanup + Full Migration (#89)
│
├── Scan analysis:
│   └── reduce_det_scanvar     — bin by arbitrary scan variable
│
├── PostProcessing integration:
│   └── Port fitting/kinetics as registered steps or standalone utilities
│
├── Visualization refactor:
│   └── Read from run.results dict, not getattr(run, key)
│
├── Deletion:
│   └── Remove XSpect_Controller.py (all subclasses dead)
│
└── Dependencies: Phases 1-3 complete (all analysis paths ported)
```

### Dependency Graph (critical path)

```
1A ──→ 1B ──→ 1C ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
                         │                        │
                         └── shares time_binning ──┘
```

Critical path: registry.py (1A) → step registration (1B) → integration test (1C). Everything after Phase 1 is incremental step additions with the same pattern.

### Design Decisions

| Decision | Resolution |
|----------|-----------|
| Pipeline model | Flat ordered list, top-to-bottom, no conditionals |
| Step dispatch | Registry via `@register_step(name)` decorator |
| Run state | `run.results` flat dict, dot-separated keys |
| `on:` field | String passed to step; step interprets (mask name, detector key, or derived key) |
| Parallelism | Transparent. `batch_size`/`cores` at runtime, not in YAML |
| Reduction lifecycle | `Pipeline.run()` executes two phases: batch-parallel pipeline, then serial reduction across all runs |
| Backwards compat | Old imports remain via shim `__init__.py` until Phase 4 deletes them |
| Test data | Small synthetic HDF5 fixture committed to `tests/fixtures/` for unit tests |
| YAML validation | Steps self-describe expected args; config_parser raises on unknown step names or missing required fields |
| Notebook interface | `Pipeline.from_yaml(path).run(cores=16, batch_size=2000)` |

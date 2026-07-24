# XSpect Quarterly Review: 2026-04-16 to 2026-07-16

Summary of features and bug fixes over the last three months, for discussion 2026-07-16.

The dominant theme was the architecture overhaul (issue #82): moving from controller subclasses to a YAML-driven pipeline with MVC separation. Alongside that, several detector-processing and binning bugs were fixed, and new analysis steps were added for droplet reconstruction, bad-pixel patching, and variance filtering.

## Architecture overhaul: YAML pipeline (issue #82)

Merged to master via PR #100 on 2026-07-16. Replaces the old pattern where each experiment subclassed a controller. Analysis now runs from a declarative YAML file that lists pipeline steps in order.

Delivered in phases:

- **Phase 1A (#84)**: core infrastructure. Step/reduction registry (`@register_step`, `get_step`), model classes (`experiment`, `spectroscopy_run`, `vonHamos`), and the controller skeleton (`Pipeline`, batch manager, config parser).
- **Phase 1B (#85)**: registered the static XES analysis steps against the registry.
- **Phase 1C (#86)**: proved static XES runs end-to-end from YAML.
- **Phase 2 (#87)**: time-resolved (pump-probe) XES pipeline steps.

The user guide covering YAML structure, every registered step, naming conventions, and complete examples landed 2026-06-05 (`docs/YAML_PIPELINE_GUIDE.md`).

### Why it matters
- Adding a new experiment is now a YAML file, not a Python subclass.
- Steps are stateless functions `step(run, **kwargs)`, so they are unit-testable in isolation.
- A regression suite with saved reference arrays (2026-06-18) guards the numerical output of the ultrafast and static XES paths.

## New analysis steps

- **droplet_reconstruction** (2026-07-01): reconstructs sparse photon-counting data from the droplet HDF5 layout used at MFX. Native step, replacing per-experiment scripts.
- **patch_pixels** overhaul (2026-07-01): auto-detection of bad columns by two methods (bright-spike ratio and subtle z-score dip), vectorized polynomial fit for interpolation, and merging of manual with auto-detected pixels.
- **find_rotation_angle** and **rotate_detector** (2026-06-16, 2026-07-01): detector rotation correction, registered as a pipeline step.
- **filter_detector_variance** (#36, 2026-07-16): zeros low-variance detector pixels using `sklearn.feature_selection.VarianceThreshold`, as a data-driven alternative to ADU/keV threshold filtering. Stores the retained-pixel mask and preserves input shape.
- **spectrum derivative analyzer** (2026-05-22): interactive widget for foil energy scans. Click-drag over a peak; paired with the `vonHamos` class it back-calculates the correct crystal-to-detector distance from a reference energy.

## Bug fixes

- **multiprocessing pickling** (2026-06-16): h5py file handles were left open after data load, which broke pickling for the `multiprocessing.Pool` batch path. Fixed by closing the handle after load.
- **pipeline step ordering** (2026-06-16): reordered steps to match the old controller workflow so YAML output matches the legacy path.
- **batch parallelism, energy calibration, spatial reduction** (2026-06-18): corrected several defects surfaced when running the YAML pipelines under batch parallelism.
- **CCM/temporal binning** (2026-07-01): added 3D detector support, NaN handling, and fixed the `ccm_energies` length mismatch.
- **scalar pre-pipeline pass** (2026-07-01): added a scalar precompute pass, a `max_shots` limit, and absolute HDF5 index handling in the batch manager.

## LUTE / smalldata work

- **smalldata sums (#93)** and **epics variables (#94)**, closed 2026-06-23: LUTE needs cross-shot sums and epics variables exposed in smalldata.
- **LUTE at XCS (#78), LUTE Lesson (#79), xcs101591326 setup (#92)**, closed 2026-07-03: LUTE workflow setup and documentation for XCS.

## Experiment pipelines added

- XCS xcs101591326: ultrafast/temporal/2D XAS configs and analysis notebook (2026-07-01).
- MFX mfx101609126: per-shot XES, static RIXS, SEER, droplet configs, and diagnostic/stochastic RIXS notebooks (2026-07-01).
- MFX mfx102101026: Fe XES/XAS speciation pipeline with import-time ROI and analysis notebooks (2026-07-15).

## Tooling

- **XSpect-setup-XES skill** (2026-07-15/16): end-to-end setup of a new XES experiment. Covers LUTE smalldata, LUTE YAML, XSpect pipeline YAML, and diagnostic/analysis notebooks for static, time-resolved, CCM-scanned, and droplet/photon-counted XES. Includes an HDF5 key identification phase with detector-naming conventions.

## Merged PRs

- **#90** Xcs101237825 (2026-05-22)
- **#95** Droplet reconstruction, patch_pixels auto-detect, XAS fixes, XCS/MFX pipelines (2026-07-01)
- **#100** YAML-driven pipeline architecture with MVC separation (2026-07-16)

## Discussion points

- The YAML architecture is merged to master. Remaining phase work beyond Phase 2 (time-resolved) should be scoped.
- Legacy top-level modules (`XSpect_Analysis.py`, `XSpect_Controller.py`, `XSpect_Diagnostics.py`, `XSpect_Visualization.py`) still ship alongside the new package. Decide on a deprecation path.
- Regression coverage exists for ultrafast and static XES. XAS and RIXS paths need equivalent reference-array tests.

# Requirements: XSpect YAML Pipeline Phases 1B-1C (Step Registration + Integration)

## Problem Statement

Phase 1A delivered the pipeline infrastructure (registry, config parser, batch manager, Pipeline class) but only with placeholder steps. Users still cannot run a real analysis from YAML. This phase registers the actual analysis operations as step functions and proves the pipeline can execute real XES and XAS workflows.

## Scope

### Phase 1B: Register analysis steps
Extract each atomic analysis operation from SpectroscopyAnalysis, XESAnalysis, and XASAnalysis as a standalone registered step function. Each step follows the `(run, **kwargs) -> None` contract.

### Phase 1C: Integration
Write YAML configs that replicate the existing controller workflows (XESBatchAnalysis, XASBatchAnalysis) and prove they produce equivalent results.

## User Stories

### US-8: Core analysis steps available in registry
**As a** beamline scientist  
**I want** all standard analysis operations (filter, reduce, bin, normalize) registered as pipeline steps  
**So that** I can compose arbitrary analysis workflows in YAML.

**Acceptance criteria:**
- AC-8.1: `filter_shots` registered and callable from pipeline
- AC-8.2: `filter_detector_adu` registered and callable
- AC-8.3: `union_shots` and `separate_shots` registered
- AC-8.4: `reduce_detector_spatial` registered
- AC-8.5: `reduce_detector_temporal` registered
- AC-8.6: `time_binning` registered
- AC-8.7: `normalize_xes` registered
- AC-8.8: `make_energy_axis` registered
- AC-8.9: `list_steps()` returns all registered step names

### US-9: XES-specific steps
**As a** scientist running XES experiments  
**I want** XES-specific steps (normalize, energy axis) available  
**So that** I can run a complete XES workflow from YAML.

**Acceptance criteria:**
- AC-9.1: XES normalization step works on run.results data
- AC-9.2: Energy axis generation works with vonHamos parameters from YAML

### US-10: XAS-specific steps
**As a** scientist running XAS experiments  
**I want** XAS-specific steps (CCM binning, CCM reduction) available  
**So that** I can run XAS workflows from YAML.

**Acceptance criteria:**
- AC-10.1: `make_ccm_axis` registered
- AC-10.2: `ccm_binning` registered
- AC-10.3: `reduce_detector_ccm` registered
- AC-10.4: `reduce_detector_ccm_temporal` registered (2D reduction)

### US-11: Data loading as pipeline steps
**As a** pipeline user  
**I want** data loading to happen as the first steps in my pipeline  
**So that** the YAML file controls what gets loaded.

**Acceptance criteria:**
- AC-11.1: `load_run_keys` registered - loads keys specified in YAML data section
- AC-11.2: `load_run_key_delayed` registered - loads detector data with ROI support
- AC-11.3: `get_run_shot_properties` registered - sets xray/laser/simultaneous masks

### US-12: End-to-end XES workflow from YAML
**As a** beamline scientist  
**I want** a YAML file that replicates the XESBatchAnalysis workflow  
**So that** I can validate the pipeline produces correct results.

**Acceptance criteria:**
- AC-12.1: A YAML config exists for standard XES time-resolved analysis
- AC-12.2: Pipeline executes without error on synthetic test data
- AC-12.3: Results dict contains expected keys (time_binned, normalized, energy_axis)

### US-13: Reduction steps work across runs
**As a** scientist analyzing multiple runs  
**I want** reduction steps (combine_runs) to aggregate data across all runs  
**So that** multi-run normalization and averaging work.

**Acceptance criteria:**
- AC-13.1: `combine_runs` reduction registered
- AC-13.2: Reduction receives list of analyzed runs
- AC-13.3: Reduction output accessible on pipeline.results

## Out of Scope
- Real LCLS HDF5 data (tests use synthetic numpy arrays)
- S3DF filesystem access
- Visualization steps (deferred)
- Post-processing steps (fitting, kinetics - deferred)

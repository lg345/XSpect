# Requirements: XSpect YAML Pipeline Architecture (Phase 1A)

## Problem Statement

XSpect's analysis controller has grown into 6+ subclasses, each encoding a specific analysis workflow as imperative Python. Adding a new experiment type requires writing a new subclass. Configuration is scattered across notebook cells as attribute assignments. No reproducibility artifact exists: you cannot hand someone a file that fully specifies what analysis was run.

The overhaul replaces this with a single generic Pipeline class that reads a YAML recipe and dispatches registered analysis steps.

## Stakeholders

| Persona | Role | Needs |
|---------|------|-------|
| Beamline scientist | Primary user | Run analysis from a YAML file in a Jupyter notebook on S3DF |
| Instrument developer | Adds new steps | Register a new step with a decorator and have it available in YAML |
| Data analyst | Post-experiment | Reproduce an analysis by re-running the same YAML on the same data |

## User Stories (Phase 1A)

### US-1: Pipeline execution from YAML
**As a** beamline scientist  
**I want to** call `Pipeline.from_yaml("my_analysis.yaml").run(cores=16, batch_size=2000)`  
**So that** my analysis is fully specified in one file and executes without manual attribute setting.

**Acceptance criteria:**
- AC-1.1: `Pipeline.from_yaml(path)` parses a valid YAML and returns a Pipeline instance
- AC-1.2: `.run(cores=N, batch_size=M)` executes all pipeline steps in order on each run
- AC-1.3: Results are accessible via `pipeline.results` after execution
- AC-1.4: Invalid YAML (missing required sections) raises `ConfigValidationError` with a message identifying what's missing

### US-2: Step registry
**As an** instrument developer  
**I want to** decorate a function with `@register_step("my_step")` and have it dispatchable from YAML  
**So that** I can add analysis operations without modifying the pipeline runner.

**Acceptance criteria:**
- AC-2.1: `@register_step("name")` registers the function in a global registry
- AC-2.2: `get_step("name")` returns the registered callable
- AC-2.3: `get_step("nonexistent")` raises `StepNotFoundError`
- AC-2.4: `list_steps()` returns all registered step names
- AC-2.5: `@register_reduction("name")` registers reduction functions separately
- AC-2.6: `get_reduction("name")` returns the registered reduction callable

### US-3: Batch-parallel execution
**As a** beamline scientist  
**I want** the pipeline to automatically split my runs into batches and parallelize across cores  
**So that** I get results faster without managing multiprocessing myself.

**Acceptance criteria:**
- AC-3.1: Batch manager splits a run's shots into contiguous ranges of size `batch_size`
- AC-3.2: Each batch is processed independently via multiprocessing.Pool
- AC-3.3: Batch results are reconverged into a single run-level result
- AC-3.4: `cores` parameter controls Pool size
- AC-3.5: Single-core execution (cores=1) works without spawning a Pool

### US-4: Run results dict
**As a** pipeline step  
**I want to** read from and write to `run.results` using dot-separated keys  
**So that** steps can produce named outputs that downstream steps consume.

**Acceptance criteria:**
- AC-4.1: `spectroscopy_run` has a `self.results = {}` attribute
- AC-4.2: Steps write to `run.results["key"]` (e.g., `run.results["epix_ROI_1.simultaneous_laser"]`)
- AC-4.3: Steps can read from `run.results["key"]` for inputs
- AC-4.4: Existing data loading methods (`load_run_keys`) still work and populate results

### US-5: YAML config validation
**As a** user  
**I want** the config parser to validate my YAML before execution starts  
**So that** I get clear errors upfront rather than failures mid-pipeline.

**Acceptance criteria:**
- AC-5.1: Parser validates presence of required top-level sections: `experiment`, `data`, `pipeline`
- AC-5.2: Parser validates each pipeline step has a `step:` field
- AC-5.3: Parser validates each step name exists in the registry (raises `StepNotFoundError`)
- AC-5.4: Parser returns a structured config object (not raw dict)
- AC-5.5: `output` section is optional (defaults apply)

### US-6: Reduction lifecycle
**As a** beamline scientist  
**I want** reduction steps (combine_runs, normalize) to execute after all runs are processed  
**So that** reductions can aggregate across runs.

**Acceptance criteria:**
- AC-6.1: `reduction` section in YAML is parsed separately from `pipeline`
- AC-6.2: Reduction steps execute after all runs complete their pipeline
- AC-6.3: Reduction steps receive the list of all analyzed run objects
- AC-6.4: Reduction results are accessible on `pipeline.results`

### US-7: Backwards compatibility
**As an** existing user  
**I want** `from XSpect.XSpect_Analysis import spectroscopy_run` to still work  
**So that** my notebooks don't break while the migration is in progress.

**Acceptance criteria:**
- AC-7.1: `from XSpect.XSpect_Analysis import experiment` still imports correctly
- AC-7.2: `from XSpect.XSpect_Analysis import spectroscopy_run` still imports correctly
- AC-7.3: `from XSpect.XSpect_Analysis import SpectroscopyAnalysis` still imports correctly
- AC-7.4: Deprecation warnings emitted on old import paths (optional, not required for 1A)

## Out of Scope (Phase 1A)

- Registering actual analysis steps (Phase 1B)
- End-to-end execution on real LCLS data (Phase 1C)
- Time-resolved analysis (Phase 2)
- XAS analysis (Phase 3)
- Visualization/PostProcessing integration (Phase 4)
- YAML schema versioning
- Pipeline resumability
- Remote execution or job submission

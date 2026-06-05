# Implementation Plan: XSpect YAML Pipeline (Phase 1A)

## Completed Milestones

### M0: Package scaffolding
- Created XSpect/model/, XSpect/analysis/, XSpect/controller/ directories
- Added __init__.py files to all packages
- Created tests/ directory with conftest.py and fixtures/

### M1: Model layer
- XSpect/model/experiment.py: experiment + spectroscopy_experiment (from XSpect_Analysis.py)
- XSpect/model/run.py: spectroscopy_run with self.results = {} added
- XSpect/model/von_hamos.py: vonHamos class moved

### M2: Analysis registry
- XSpect/analysis/registry.py: @register_step, @register_reduction, get_step, get_reduction, list_steps, list_reductions, clear_registry
- StepNotFoundError and ReductionNotFoundError custom exceptions
- Duplicate name detection

### M3: Controller layer
- config_parser.py: parse_yaml() with full validation, _normalize_yaml_keys() for PyYAML boolean key bug
- pipeline_runner.py: run_pipeline() and run_reductions()
- batch_manager.py: split_into_batches(), run_batched(), reconverge_results()

### M4: Pipeline class + integration
- controller/pipeline.py: Pipeline.from_yaml(), .run(), .results
- _MockExperiment fallback for when real data paths unavailable
- Graceful handling of missing HDF5 files

### M5: Backwards compatibility
- XSpect/__init__.py re-exports all key classes
- Old import paths (from XSpect.XSpect_Analysis import ...) still work
- All 7 backwards-compat tests pass

## Test Results

52 tests passing:
- 7 backwards compatibility tests
- 12 batch manager tests
- 11 config parser tests
- 6 pipeline integration tests
- 11 registry tests
- 5 run model tests

## Files Created

| File | Purpose |
|------|---------|
| XSpect/model/__init__.py | Model package exports |
| XSpect/model/experiment.py | experiment classes |
| XSpect/model/run.py | spectroscopy_run with results dict |
| XSpect/model/von_hamos.py | vonHamos geometry |
| XSpect/analysis/__init__.py | Registry exports |
| XSpect/analysis/registry.py | Step/reduction registry |
| XSpect/controller/__init__.py | Controller exports |
| XSpect/controller/config_parser.py | YAML parsing + validation |
| XSpect/controller/pipeline_runner.py | Step dispatch loop |
| XSpect/controller/batch_manager.py | Batch parallelism |
| XSpect/controller/pipeline.py | Pipeline class |
| XSpect/__init__.py | Top-level backwards-compat shim |
| tests/__init__.py | Test package |
| tests/conftest.py | Fixtures + placeholder steps |
| tests/fixtures/test_pipeline.yaml | Integration test YAML |
| tests/fixtures/test_pipeline_no_reduction.yaml | Multi-run test YAML |
| tests/test_registry.py | Registry unit tests |
| tests/test_config_parser.py | Config parser unit tests |
| tests/test_batch_manager.py | Batch manager unit tests |
| tests/test_run.py | Run model unit tests |
| tests/test_pipeline_integration.py | End-to-end integration tests |
| tests/test_backwards_compat.py | Import compatibility tests |

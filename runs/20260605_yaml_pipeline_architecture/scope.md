# Scope: XSpect YAML Pipeline Architecture (Phase 1A)

## MVP Definition (Phase 1A)

Phase 1A delivers the foundational infrastructure that all subsequent phases build on. After 1A, the Pipeline class can load a YAML, dispatch placeholder steps through the registry, manage batches across cores, and produce results. No real analysis logic is ported yet.

### In scope

| Component | Deliverable | Notes |
|-----------|-------------|-------|
| model/experiment.py | experiment, spectroscopy_experiment classes | Moved from XSpect_Analysis.py, minimal changes |
| model/run.py | spectroscopy_run with results dict | Refactored: adds self.results = {}, retains data loading |
| analysis/registry.py | Step + reduction registry with decorators | New file, core dispatch mechanism |
| controller/config_parser.py | YAML parser + validator | New file, returns PipelineConfig dataclass |
| controller/pipeline_runner.py | Step dispatch loop | New file, iterates steps, calls registry |
| controller/batch_manager.py | Shot chunking + Pool + reconvergence | Extracted from BatchAnalysis |
| Pipeline class | from_yaml(), run(), results | New, user-facing entry point |
| __init__.py shim | Backwards-compatible imports | Old import paths still resolve |
| tests/fixtures/ | Synthetic HDF5 fixture + test YAML | For unit/integration tests |

### Out of scope (future phases)

| Phase | What | Why deferred |
|-------|------|-------------|
| 1B | Register 10 static XES steps | Depends on registry existing (1A) |
| 1C | End-to-end on real data | Depends on steps existing (1B) |
| 2 | Time-resolved XES steps + reductions | Depends on 1C validation |
| 3 | XAS steps (CCM, energy binning) | Depends on Phase 2 patterns |
| 4 | Scan, PostProcessing, Visualization | Full migration, delete old code |

## Non-goals

- No web interface or REST API
- No GUI of any kind
- No changes to visualization or diagnostics code
- No changes to XSpect_PostProcessing.py
- No changes to XSpect_Processor/ subpackage internals
- No actual analysis logic ported (that's 1B)
- No real LCLS data dependency for tests
- No YAML schema versioning system
- No CI/CD pipeline setup (future)

## Assumptions

1. Python >=3.9 on all target systems (S3DF, NERSC)
2. PyYAML is an acceptable dependency (already used elsewhere at LCLS)
3. multiprocessing.Pool is the parallelism primitive (not Dask, not Ray)
4. The existing spectroscopy_run data loading interface (load_run_keys, load_run_key_delayed) is preserved
5. Steps are pure functions: `step(run, **kwargs) -> None` (mutate run.results in place)
6. No need for async/await patterns
7. HDF5 smalldata files are the only data source format

## Success criteria

Phase 1A is complete when:
1. `Pipeline.from_yaml("test.yaml").run(cores=4, batch_size=1000)` executes without error using 2+ placeholder steps
2. Registry dispatches decorated steps correctly
3. Batch manager splits 10000 shots into 5 batches of 2000 and reconverges
4. `from XSpect.XSpect_Analysis import spectroscopy_run` still works
5. All unit tests pass with synthetic fixtures (no S3DF access needed)
6. No import errors when the package is loaded

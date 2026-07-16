# Project Plan: XSpect YAML Pipeline Architecture (Phase 1A)

## Milestone Breakdown

### M0: Package scaffolding (first)

**Deliverables:**
- Create directory structure: `XSpect/model/`, `XSpect/analysis/`, `XSpect/controller/`
- Add `__init__.py` files to all new packages
- Add `pyproject.toml` or update existing setup for new subpackages
- Create `tests/` directory with `conftest.py`
- Create `tests/fixtures/` for synthetic test data

**Definition of done:** `import XSpect.model`, `import XSpect.analysis`, `import XSpect.controller` all succeed.

---

### M1: Model layer (depends on M0)

**Deliverables:**
- `XSpect/model/experiment.py`: move `experiment` and `spectroscopy_experiment` from XSpect_Analysis.py
- `XSpect/model/run.py`: refactored `spectroscopy_run` with `self.results = {}`
- `XSpect/model/__init__.py`: exports all model classes
- Unit tests for run.results dict behavior

**Definition of done:** `spectroscopy_run` instantiates, `self.results` is writable, existing data loading methods preserved in signature.

---

### M2: Analysis registry (depends on M0)

**Deliverables:**
- `XSpect/analysis/registry.py`: `@register_step`, `@register_reduction`, `get_step`, `get_reduction`, `list_steps`, `list_reductions`
- Custom exceptions: `StepNotFoundError`, `ReductionNotFoundError`
- Unit tests: register, retrieve, list, error on missing

**Definition of done:** Decorated functions appear in registry and are callable via `get_step("name")`.

---

### M3: Controller layer (depends on M1 + M2)

**Deliverables:**
- `XSpect/controller/config_parser.py`: parse YAML, validate, return `PipelineConfig`
- `XSpect/controller/pipeline_runner.py`: iterate steps, dispatch from registry
- `XSpect/controller/batch_manager.py`: split shots, Pool, reconverge
- Custom exception: `ConfigValidationError`
- Unit tests for each component

**Definition of done:** Config parser rejects bad YAML with clear errors. Pipeline runner dispatches placeholder steps. Batch manager splits 10000 shots into correct ranges.

---

### M4: Pipeline class + integration (depends on M3)

**Deliverables:**
- `Pipeline` class (likely in `XSpect/controller/pipeline.py` or `XSpect/pipeline.py`)
- `Pipeline.from_yaml(path)` class method
- `.run(cores, batch_size)` method
- `.results` attribute
- Integration test: load test YAML, run with placeholder steps, verify results populated
- Test YAML fixture: `tests/fixtures/test_pipeline.yaml`

**Definition of done:** `Pipeline.from_yaml("tests/fixtures/test_pipeline.yaml").run(cores=1, batch_size=100)` completes, results dict is populated by placeholder steps.

---

### M5: Backwards compatibility + cleanup (depends on M4)

**Deliverables:**
- Update `XSpect/__init__.py` to re-export from new locations
- Verify `from XSpect.XSpect_Analysis import spectroscopy_run` still works
- Verify `from XSpect.XSpect_Analysis import experiment` still works
- Keep old files in place (they still contain analysis logic needed in 1B)

**Definition of done:** Existing import patterns do not break. Old XSpect_Analysis.py still importable.

---

## Dependency Graph

```
M0 (scaffolding)
 ├── M1 (model)
 │     └──┐
 └── M2 (registry)
           └── M3 (controller: parser + runner + batch_manager)
                    └── M4 (Pipeline class + integration test)
                              └── M5 (backwards compat shim)
```

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| spectroscopy_run refactor breaks data loading | Medium | High | Keep all existing methods, only add self.results; don't remove attributes yet |
| Multiprocessing pickle failures on run objects | Medium | Medium | Ensure run objects are picklable; test with cores>1 early |
| YAML parsing edge cases (ranges, nested dicts) | Low | Low | Well-defined schema, reject anything ambiguous |
| Import shim creates circular imports | Medium | Medium | Keep shim simple: direct re-exports, no logic |

## Versioning

- This work lives on branch `feature/yaml-pipeline-architecture`
- Phase 1A = one PR merged to master when all milestones pass
- Phases 1B, 1C, 2, 3, 4 are separate PRs

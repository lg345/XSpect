# Test Strategy: XSpect Pipeline Architecture (Phase 1A)

## Test Framework

- **Runner:** pytest
- **Fixtures:** synthetic HDF5 files + YAML configs in `tests/fixtures/`
- **No external dependencies:** all tests run without S3DF access or real LCLS data

## Test Categories

### 1. Unit Tests

#### Registry (tests/test_registry.py)

| Test | Verifies |
|------|----------|
| test_register_step_adds_to_registry | @register_step decorator populates _STEP_REGISTRY |
| test_get_step_returns_callable | get_step("name") returns the decorated function |
| test_get_step_raises_on_missing | get_step("nonexistent") raises StepNotFoundError |
| test_list_steps_returns_all_names | list_steps() returns set of registered names |
| test_register_reduction_separate | Reductions go to _REDUCTION_REGISTRY, not step registry |
| test_duplicate_step_name_raises | Registering same name twice raises ValueError |

#### Config Parser (tests/test_config_parser.py)

| Test | Verifies |
|------|----------|
| test_parse_valid_yaml | Returns PipelineConfig with correct fields |
| test_missing_experiment_section | Raises ConfigValidationError |
| test_missing_pipeline_section | Raises ConfigValidationError |
| test_missing_data_section | Raises ConfigValidationError |
| test_unknown_step_name | Raises StepNotFoundError during validation |
| test_step_missing_step_field | Raises ConfigValidationError |
| test_run_range_expansion | "162-181" expands to [162, 163, ..., 181] |
| test_output_section_defaults | Missing output section gets default format/path |
| test_reduction_section_optional | Missing reduction section results in empty list |

#### Batch Manager (tests/test_batch_manager.py)

| Test | Verifies |
|------|----------|
| test_split_shots_even | 10000 shots / 2000 batch_size = 5 ranges |
| test_split_shots_remainder | 10001 shots / 2000 = 5 full + 1 partial |
| test_split_shots_smaller_than_batch | 500 shots / 2000 = 1 range of 500 |
| test_reconverge_combines_results | Multiple batch results merged into run.results |
| test_single_core_no_pool | cores=1 executes without multiprocessing.Pool |

#### Model - Run (tests/test_run.py)

| Test | Verifies |
|------|----------|
| test_run_has_results_dict | spectroscopy_run().results is {} |
| test_run_results_writable | Can assign run.results["key"] = value |
| test_run_status_logging | update_status appends to status list |
| test_run_preserves_existing_attrs | run_number, experiment, verbose all work |

### 2. Integration Tests

#### Pipeline end-to-end (tests/test_pipeline_integration.py)

| Test | Verifies |
|------|----------|
| test_pipeline_from_yaml_basic | Loads test YAML, returns Pipeline instance |
| test_pipeline_run_placeholder_steps | Runs 2 placeholder steps, results populated |
| test_pipeline_run_multicore | cores=2 produces same results as cores=1 |
| test_pipeline_reduction_phase | Reduction steps execute after pipeline steps |
| test_pipeline_multiple_runs | Pipeline processes multiple run numbers |

### 3. Backwards Compatibility Tests

#### Import shim (tests/test_backwards_compat.py)

| Test | Verifies |
|------|----------|
| test_import_experiment_old_path | from XSpect.XSpect_Analysis import experiment |
| test_import_spectroscopy_run_old_path | from XSpect.XSpect_Analysis import spectroscopy_run |
| test_import_spectroscopy_experiment_old_path | from XSpect.XSpect_Analysis import spectroscopy_experiment |
| test_old_classes_are_same_objects | Old and new imports resolve to the same class |

## Test Fixtures

### Synthetic HDF5 (tests/fixtures/synthetic_run.h5)

Minimal HDF5 file mimicking smalldata structure:
- 100 shots (enough to test batching)
- `lightStatus/xray`: boolean array
- `lightStatus/laser`: boolean array
- `ipm4/sum`: float array (random, some above/below threshold)
- `epix_1/ROI_0_area`: 2D array (100 x 50 pixels, random)
- `enc/lasDelay`: float array

### Test YAML (tests/fixtures/test_pipeline.yaml)

```yaml
experiment:
  hutch: test
  experiment_id: test00000
  lcls_run: 1

data:
  runs: [1]
  keys:
    ipm4/sum: ipm
  detector_keys:
    epix_1/ROI_0_area:
      name: epix
      rois: [[0, -1]]
      combine_rois: true

pipeline:
  - step: placeholder_add
    on: epix
    value: 1.0
  - step: placeholder_multiply
    on: epix
    value: 2.0

output:
  format: hdf5
  path: ./test_results/
```

### Placeholder steps (registered in conftest.py)

```python
@register_step("placeholder_add")
def placeholder_add(run, on=None, value=0, **kwargs):
    key = on or "data"
    run.results[f"{key}.added"] = run.results.get(key, 0) + value

@register_step("placeholder_multiply")
def placeholder_multiply(run, on=None, value=1, **kwargs):
    key = on or "data"
    run.results[f"{key}.multiplied"] = run.results.get(key, 0) * value

@register_reduction("placeholder_sum")
def placeholder_sum(runs, on=None, **kwargs):
    return {"summed": sum(r.results.get(on, 0) for r in runs)}
```

## Data Integrity Checks

- Batch reconvergence must not drop or duplicate shots
- Results dict keys must be strings (enforced by type check or schema)
- Config parser must reject YAML with duplicate step entries at same position (or allow and execute both)

## Performance Considerations

- Batch manager test with 100k shots should complete in <1s (no I/O, just index math)
- Registry lookup should be O(1) dict access
- No performance tests against real HDF5 in Phase 1A (deferred to 1C)

## Coverage Targets

- analysis/registry.py: 95%+
- controller/config_parser.py: 90%+
- controller/batch_manager.py: 85%+
- controller/pipeline_runner.py: 85%+
- model/run.py: 80%+ (data loading methods may not be fully testable without real HDF5)

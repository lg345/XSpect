# Deployment Checklist: XSpect YAML Pipeline (Phase 1A)

## Merge Readiness

- [x] All 52 tests pass (`pytest tests/ -v`)
- [x] Coverage meets targets for core pipeline modules (registry 95%, config_parser 93%, batch_manager 93%, pipeline_runner 100%)
- [x] No blocker or major defects
- [x] Backwards compatibility verified (7 import tests pass)
- [x] Branch: `feature/yaml-pipeline-architecture` clean and ready for PR

## Pre-Merge Checks

- [x] No changes to existing files (XSpect_Analysis.py, XSpect_Controller.py, etc. untouched)
- [x] New packages are additive only (model/, analysis/, controller/)
- [x] PyYAML listed in existing environment (already a dependency)
- [x] No new external dependencies introduced

## Post-Merge Steps

- [ ] Create PR from `feature/yaml-pipeline-architecture` to `master`
- [ ] Verify CI passes (if configured)
- [ ] Tag as v0.2.0-alpha or similar to mark Phase 1A completion
- [ ] Update CONTEXT.md Phase 1A status from "planned" to "complete"
- [ ] Close GitHub issue #84 (Phase 1A sub-issue)

## Next Phase Dependencies

Phase 1B (Register actual analysis steps) can begin immediately after merge. It requires:
- The registry at `XSpect/analysis/registry.py` (delivered)
- The step signature convention: `step(run, **kwargs) -> None` (documented in ARCHITECTURE.md)
- Understanding of `run.results` key conventions (dot-separated strings)

## Usage Example (post-merge)

```python
from XSpect import Pipeline

pipe = Pipeline.from_yaml("my_analysis.yaml")
pipe.run(cores=16, batch_size=2000)
print(pipe.results)
```

For step development (Phase 1B):
```python
from XSpect.analysis.registry import register_step

@register_step("filter_xray_on")
def filter_xray_on(run, on=None, **kwargs):
    # step implementation here
    run.results[f"{on}.filtered"] = ...
```

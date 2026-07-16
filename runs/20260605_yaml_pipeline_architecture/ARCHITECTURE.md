# Architecture: XSpect YAML Pipeline (Phase 1A)

## Module Layout

```
XSpect/
├── __init__.py              # backwards-compat re-exports
├── model/
│   ├── __init__.py          # exports experiment, spectroscopy_run, vonHamos
│   ├── experiment.py        # experiment, spectroscopy_experiment
│   ├── run.py               # spectroscopy_run (with self.results = {})
│   └── von_hamos.py         # vonHamos crystal geometry
├── analysis/
│   ├── __init__.py          # exports registry functions
│   └── registry.py          # @register_step, @register_reduction, dispatch
├── controller/
│   ├── __init__.py          # exports Pipeline, parse_yaml, split_into_batches
│   ├── config_parser.py     # YAML -> PipelineConfig (frozen dataclasses)
│   ├── pipeline_runner.py   # step dispatch loop + reduction execution
│   ├── batch_manager.py     # shot chunking, Pool, reconvergence
│   └── pipeline.py          # Pipeline class (user-facing entry point)
├── XSpect_Analysis.py       # UNCHANGED (still contains analysis logic for Phase 1B)
├── XSpect_Controller.py     # UNCHANGED (still contains controller subclasses)
├── XSpect_PostProcessing.py # UNCHANGED
├── XSpect_Visualization.py  # UNCHANGED
├── XSpect_Diagnostics.py    # UNCHANGED
└── XSpect_Processor/        # UNCHANGED
```

## Core Abstractions

### Pipeline (controller/pipeline.py)
User-facing entry point. Loads YAML, creates experiment/run objects, dispatches steps.

### Registry (analysis/registry.py)
Module-level dicts mapping step names to callables. Decorators for registration.
Steps: `(run, **kwargs) -> None` (mutate run.results in place).
Reductions: `(runs, **kwargs) -> dict`.

### PipelineConfig (controller/config_parser.py)
Frozen dataclass tree representing a parsed YAML. Immutable after construction.
Handles the YAML `on:` keyword (which PyYAML parses as boolean True) via normalization.

### BatchManager (controller/batch_manager.py)
Splits shot ranges, optionally parallelizes via multiprocessing.Pool, reconverges.

## Data Flow

```
YAML -> parse_yaml() -> PipelineConfig
                              |
                    Pipeline.run(cores, batch_size)
                              |
              for each run in config.data.runs:
                  create spectroscopy_run
                  run_pipeline(run, steps)  [or run_batched if total_shots > batch_size]
                              |
              run_reductions(analyzed_runs, reduction_steps)
                              |
                    Pipeline.results populated
```

## Design Decisions

- Steps are stateless functions, not methods on a class.
- `run.results` is a flat dict with dot-separated string keys.
- The `on:` field in YAML is just a string argument; the step decides how to interpret it.
- Batch reconvergence collects per-batch results into lists (downstream decides how to merge).
- Old XSpect_Analysis.py and XSpect_Controller.py are untouched for backwards compat.

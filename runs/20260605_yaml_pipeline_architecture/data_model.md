# Data Model: XSpect Pipeline Architecture

## Overview

XSpect does not use a relational database. The "data model" here describes the in-memory object graph and the YAML configuration schema that drives it.

## Core Objects

### PipelineConfig (new, from YAML)

Returned by `config_parser.parse_yaml(path)`. Immutable after construction.

```python
@dataclass(frozen=True)
class ExperimentConfig:
    hutch: str
    experiment_id: str
    lcls_run: int

@dataclass(frozen=True)
class DataKeyConfig:
    hdf5_path: str        # e.g., "tt/ttCorr"
    friendly_name: str    # e.g., "time_tool_correction"

@dataclass(frozen=True)
class DetectorKeyConfig:
    hdf5_path: str        # e.g., "epix_1/ROI_0_area"
    name: str             # e.g., "epix"
    rois: list[list[int]] # e.g., [[0, -1]]
    combine_rois: bool

@dataclass(frozen=True)
class DataConfig:
    runs: list[int]                       # expanded from ranges
    keys: list[DataKeyConfig]
    detector_keys: list[DetectorKeyConfig]

@dataclass(frozen=True)
class StepConfig:
    step: str             # registered step name
    args: dict            # all other YAML fields (on, filter_key, threshold, etc.)

@dataclass(frozen=True)
class OutputConfig:
    format: str           # "hdf5" default
    path: str             # output directory

@dataclass(frozen=True)
class PipelineConfig:
    experiment: ExperimentConfig
    data: DataConfig
    pipeline: list[StepConfig]
    reduction: list[StepConfig]   # may be empty
    output: OutputConfig
```

### spectroscopy_run (refactored)

Key change: adds `self.results = {}` dict. All step outputs go here instead of as ad-hoc attributes.

```python
class spectroscopy_run:
    def __init__(self, spec_experiment, run, verbose=False, end_index=-1, start_index=0):
        self.experiment = spec_experiment
        self.run_number = run
        self.verbose = verbose
        self.end_index = end_index
        self.start_index = start_index
        
        # NEW: flat results dict with dot-separated keys
        self.results = {}
        
        # RETAINED: status logging
        self.status = []
        self.status_datetime = []
        
        # RETAINED: data loading (populates self.results on load)
        self.run_file = None
        self.run_shots = {}
    
    def update_status(self, message: str) -> None: ...
    def load_run_keys(self, keys, friendly_names) -> None: ...
    def load_run_key_delayed(self, key, friendly_name) -> None: ...
```

### experiment (unchanged)

```python
class experiment:
    def __init__(self, lcls_run, hutch, experiment_id):
        self.lcls_run = lcls_run
        self.hutch = hutch
        self.experiment_id = experiment_id
        self.experiment_directory = None  # resolved by get_experiment_directory()
    
    def get_experiment_directory(self) -> str: ...
```

### Pipeline (new, user-facing)

```python
class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = {}           # final reduction results
        self.analyzed_runs = []     # list of spectroscopy_run after pipeline
        self._status_log = []
    
    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline": ...
    
    def run(self, cores: int = 1, batch_size: int = 2000) -> None: ...
```

## Registry (new)

Not an object model per se, but a module-level dict:

```python
_STEP_REGISTRY: dict[str, Callable] = {}
_REDUCTION_REGISTRY: dict[str, Callable] = {}
```

Step signature: `def step_name(run: spectroscopy_run, **kwargs) -> None`
Reduction signature: `def reduction_name(runs: list[spectroscopy_run], **kwargs) -> dict`

## Data Flow

```
YAML file
    │
    ▼
PipelineConfig (immutable)
    │
    ▼
Pipeline.run(cores, batch_size)
    │
    ├── for each run number in config.data.runs:
    │       │
    │       ▼
    │   spectroscopy_run (loads HDF5 data)
    │       │
    │       ├── BatchManager.split(run, batch_size)
    │       │       │
    │       │       ▼
    │       │   [batch_0, batch_1, ..., batch_N]
    │       │       │
    │       │       ▼  (Pool.map if cores > 1)
    │       │   for each step in config.pipeline:
    │       │       registry.get_step(step.name)(batch_run, **step.args)
    │       │       │
    │       │       ▼
    │       │   batch_run.results populated
    │       │
    │       ▼
    │   BatchManager.reconverge([batch_results]) -> run.results
    │
    ▼
Pipeline.analyzed_runs = [run_0, run_1, ..., run_M]
    │
    ▼
for each step in config.reduction:
    registry.get_reduction(step.name)(analyzed_runs, **step.args)
    │
    ▼
Pipeline.results populated
```

## YAML Schema (reference)

```yaml
# Required sections
experiment:
  hutch: str          # required
  experiment_id: str  # required
  lcls_run: int       # required

data:
  runs: list          # required, supports ranges like "162-181"
  keys:               # required, mapping of hdf5_path -> friendly_name
    "hdf5/path": "friendly_name"
  detector_keys:      # optional
    "hdf5/path":
      name: str
      rois: list[list[int]]
      combine_rois: bool

pipeline:             # required, ordered list
  - step: str         # required, must be in registry
    on: str           # optional, interpreted by step
    # ... arbitrary kwargs passed to step

# Optional sections
reduction:
  - step: str
    on: str/list
    # ... kwargs

output:
  format: str         # default: "hdf5"
  path: str           # default: "./results/"
```

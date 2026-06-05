# XSpect YAML Pipeline

Replaces the controller-subclass pattern (XESBatchAnalysis, XASBatchAnalysis, etc.) with a single `Pipeline` class that reads a YAML recipe and dispatches registered analysis steps. One file specifies an entire analysis: what data to load, what operations to run, and how to reduce across runs.

## Quick Start

```python
from XSpect import Pipeline

pipe = Pipeline.from_yaml("my_analysis.yaml")
pipe.run(cores=16, batch_size=2000)

# Results accessible after execution
print(pipe.results.keys())
```

That's it. The YAML file defines the full workflow. `cores` and `batch_size` control parallelism at runtime without changing the recipe.

## YAML File Structure

A pipeline YAML has five sections. Three are required (`experiment`, `data`, `pipeline`), two are optional (`reduction`, `output`).

```yaml
experiment:       # required - identifies the LCLS experiment
  hutch: mfx
  experiment_id: mfx00000
  lcls_run: 21

data:             # required - what to load from HDF5
  runs: [162, 163, 164]
  keys:
    ipm4/sum: ipm
    lxt/ttc: lxt_ttc
  detector_keys:
    epix_1/ROI_0_area:
      name: epix
      rois: [[0, 80]]
      combine_rois: true

pipeline:         # required - ordered list of analysis steps
  - step: filter_shots
    on: simultaneous
    filter_key: ipm
    threshold: 5000
  - step: reduce_detector_temporal
    on: epix_ROI_1_simultaneous_laser
    timing_bin_key: timing_bin_indices_simultaneous_laser

reduction:        # optional - runs after all per-run pipelines complete
  - step: combine_runs
    detector_key: epix_ROI_1

output:           # optional - where to write results
  format: hdf5
  path: ./results/
```

### Section: `experiment`

| Field | Type | Description |
|-------|------|-------------|
| `hutch` | string | LCLS hutch (mfx, xcs, xpp, etc.) |
| `experiment_id` | string | Experiment identifier (e.g., mfx101338925) |
| `lcls_run` | int | LCLS run number |

Used to locate the smalldata HDF5 files on S3DF.

### Section: `data`

| Field | Type | Description |
|-------|------|-------------|
| `runs` | list | Run numbers to process. Supports ranges: `[162-181]` expands to all integers 162 through 181 |
| `keys` | mapping | HDF5 paths mapped to friendly names. `ipm4/sum: ipm` loads the dataset at `ipm4/sum` and stores it as the attribute `ipm` on the run object |
| `detector_keys` | mapping | HDF5 paths for 3D detector data, with ROI and transpose options |

Detector key options:

```yaml
detector_keys:
  epix_1/ROI_0_area:
    name: epix           # friendly name prefix
    rois: [[0, 80]]      # pixel ranges along last dimension
    combine_rois: true   # merge ROIs into single mask, or keep separate
    transpose: false     # swap rows/cols on load
```

### Section: `pipeline`

Flat ordered list of steps. Each entry is a mapping with a required `step` field and arbitrary keyword arguments passed to that step function.

```yaml
pipeline:
  - step: step_name
    param1: value1
    param2: value2
```

Steps execute top-to-bottom on each run independently. Every step has the signature `step(run, **kwargs)` and mutates the run object in place (setting attributes, populating results).

The `on` field is a convention (not enforced) used by most steps to specify which attribute they operate on.

### Section: `reduction`

Same format as `pipeline`, but these steps run once after all per-run pipelines finish. They receive the full list of analyzed run objects and aggregate across them.

```yaml
reduction:
  - step: combine_runs
    detector_key: epix_ROI_1
```

### Section: `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | string | "hdf5" | Output format |
| `path` | string | "./results/" | Where to write |

## Available Steps

### Data Loading

| Step | Parameters | Description |
|------|-----------|-------------|
| `load_run_keys` | `keys`, `friendly_names` | Load scalar/1D data from HDF5 |
| `load_detector` | `keys`, `friendly_names`, `transpose`, `rois`, `combine_rois` | Load 3D detector data |
| `get_run_shot_properties` | (none) | Set xray/laser/simultaneous boolean masks from lightStatus |

### Shot Filtering

| Step | Parameters | Description |
|------|-----------|-------------|
| `filter_shots` | `on`, `filter_key`, `threshold` | Zero out shots below threshold (or outside range if threshold is [min, max]) |
| `filter_detector_adu` | `on`, `adu_threshold` | Zero detector pixels below ADU threshold |

### Shot Combination

| Step | Parameters | Description |
|------|-----------|-------------|
| `union_shots` | `on`, `filter_keys`, `new_key` | Keep shots where ALL listed masks are true (logical AND). Output: `{on}_{key1}_{key2}` |
| `separate_shots` | `on`, `filter_keys`, `new_key` | Keep shots matching first mask but NOT second. Output: `{on}_{key1}_not_{key2}` |

### Spatial Reduction

| Step | Parameters | Description |
|------|-----------|-------------|
| `reduce_detector_spatial` | `on`, `rois`, `combine_rois`, `reduction`, `purge` | Sum/mean across spatial dimension within ROIs. Produces `{on}_ROI_1`, `_ROI_2`, etc. |
| `apply_roi` | `on`, `rois`, `combine_rois` | Extract ROI without reducing (keep spatial dimension) |

### Temporal Binning and Reduction

| Step | Parameters | Description |
|------|-----------|-------------|
| `time_binning` | `bins`, `lxt_key`, `fast_delay_key`, `tt_correction_key` | Create time bins from laser delay + encoder + timetool. `bins` is [min, max, n_points] |
| `reduce_detector_temporal` | `on`, `timing_bin_key`, `average` | Bin data into time bins. Produces `{on}_time_binned` |
| `reduce_detector_shots` | `on`, `reduction`, `purge` | Collapse shot dimension (sum or mean). Produces `{on}_reduced` |

### XES-Specific

| Step | Parameters | Description |
|------|-----------|-------------|
| `normalize_xes` | `on`, `pixel_range` | Normalize each row by its sum. Produces `{on}_normalized` |
| `make_energy_axis` | `detector_key`, `n_pixels`, `crystal_detector_distance`, `crystal_radius`, `d_spacing`, `mm_per_pixel`, `name` | Convert pixels to energy via vonHamos geometry. Produces `{name}_energy` |
| `patch_pixels` | `on`, `pixels`, `mode` | Repair bad pixels by interpolation or zeroing |

### XAS-Specific

| Step | Parameters | Description |
|------|-----------|-------------|
| `make_ccm_axis` | `energies` | Define CCM energy bins. `energies` is [min, max, n_points] or explicit list |
| `ccm_binning` | `ccm_key`, `ccm_bins_key` | Digitize CCM values into bin indices. Produces `ccm_bin_indices` |
| `reduce_detector_ccm` | `on`, `ccm_bin_key`, `average` | 1D energy binning. Produces `{on}_energy_binned` |
| `reduce_detector_ccm_temporal` | `on`, `timing_bin_key`, `ccm_bin_key`, `average` | 2D time+energy binning. Produces `{on}_time_energy_binned` |

### Utility

| Step | Parameters | Description |
|------|-----------|-------------|
| `bin_uniques` | `on` | Bin unique scan variable values. Produces `scanvar_indices`, `scanvar_bins` |
| `purge_keys` | `keys` | Delete attributes from run to free memory |

### Reductions

| Step | Parameters | Description |
|------|-----------|-------------|
| `combine_runs` | `detector_key`, `laser_on_suffix`, `laser_off_suffix` | Sum time-binned data across runs, compute normalized difference |

## Naming Conventions

Steps build output attribute names from the input key plus a suffix describing the operation:

```
epix_ROI_1                              (after spatial reduction)
epix_ROI_1_simultaneous_laser           (after union_shots)
epix_ROI_1_xray_not_laser              (after separate_shots)
epix_ROI_1_simultaneous_laser_time_binned  (after reduce_detector_temporal)
epix_ROI_1_simultaneous_laser_time_binned_normalized  (after normalize_xes)
```

When you use `union_shots` with `filter_keys: [simultaneous, laser]` on key `epix_ROI_1`, the output lands at `epix_ROI_1_simultaneous_laser`. The next step that needs that data uses it as its `on` parameter.

## Complete Examples

### XES Time-Resolved (replaces XESBatchAnalysis)

```yaml
experiment:
  hutch: mfx
  experiment_id: mfx101338925
  lcls_run: 21

data:
  runs: [162-181]
  keys:
    ipm4/sum: ipm
    lxt/ttc: lxt_ttc
    enc/lasDelay: encoder
  detector_keys:
    epix_1/ROI_0_area:
      name: epix
      rois: [[100, 600]]
      combine_rois: true

pipeline:
  # 1. Filter low-intensity shots
  - step: filter_shots
    on: xray
    filter_key: ipm
    threshold: 5000
  - step: filter_shots
    on: simultaneous
    filter_key: ipm
    threshold: 5000

  # 2. Remove detector noise
  - step: filter_detector_adu
    on: epix_ROI_1
    adu_threshold: 3.0

  # 3. Split by pump condition
  - step: union_shots
    on: epix_ROI_1
    filter_keys: [simultaneous, laser]
  - step: separate_shots
    on: epix_ROI_1
    filter_keys: [xray, laser]

  # 4. Time binning
  - step: time_binning
    bins: [-2, 10, 50]
    lxt_key: lxt_ttc

  # 5. Split timing indices by pump condition
  - step: union_shots
    on: timing_bin_indices
    filter_keys: [simultaneous, laser]
  - step: separate_shots
    on: timing_bin_indices
    filter_keys: [xray, laser]

  # 6. Bin detector into time bins
  - step: reduce_detector_temporal
    on: epix_ROI_1_simultaneous_laser
    timing_bin_key: timing_bin_indices_simultaneous_laser
  - step: reduce_detector_temporal
    on: epix_ROI_1_xray_not_laser
    timing_bin_key: timing_bin_indices_xray_not_laser

  # 7. Normalize spectra
  - step: normalize_xes
    on: epix_ROI_1_simultaneous_laser_time_binned
  - step: normalize_xes
    on: epix_ROI_1_xray_not_laser_time_binned

  # 8. Convert pixels to energy
  - step: make_energy_axis
    detector_key: epix_ROI_1
    crystal_detector_distance: 200.0
    crystal_radius: 250.0
    d_spacing: 1.637
    mm_per_pixel: 0.05
    name: xes

reduction:
  - step: combine_runs
    detector_key: epix_ROI_1

output:
  format: hdf5
  path: ./results/mfx101338925/
```

### XAS Energy Scan (replaces XASBatchAnalysis)

```yaml
experiment:
  hutch: xcs
  experiment_id: xcsl1004021
  lcls_run: 21

data:
  runs: [1-5]
  keys:
    ipm4/sum: ipm
    ccm/energy: ccm
    lxt/ttc: lxt_ttc
    enc/lasDelay: encoder
  detector_keys:
    epix_1/ROI_0_area:
      name: epix
      rois: [[0, 60]]
      combine_rois: true

pipeline:
  # Filter
  - step: filter_shots
    on: simultaneous
    filter_key: ipm
    threshold: 5000
  - step: filter_detector_adu
    on: epix_ROI_1
    adu_threshold: 2.0

  # Split detector by pump condition
  - step: union_shots
    on: epix_ROI_1
    filter_keys: [simultaneous, laser]
  - step: separate_shots
    on: epix_ROI_1
    filter_keys: [xray, laser]

  # Split IPM by pump condition
  - step: union_shots
    on: ipm
    filter_keys: [simultaneous, laser]
  - step: separate_shots
    on: ipm
    filter_keys: [xray, laser]

  # Time binning
  - step: time_binning
    bins: [-1, 5, 15]
    lxt_key: lxt_ttc

  # Energy binning (CCM)
  - step: make_ccm_axis
    energies: [7100, 7200, 50]
  - step: ccm_binning
    ccm_key: ccm
    ccm_bins_key: ccm_bins

  # Split bin indices by condition
  - step: union_shots
    on: timing_bin_indices
    filter_keys: [simultaneous, laser]
  - step: separate_shots
    on: timing_bin_indices
    filter_keys: [xray, laser]
  - step: union_shots
    on: ccm_bin_indices
    filter_keys: [simultaneous, laser]
  - step: separate_shots
    on: ccm_bin_indices
    filter_keys: [xray, laser]

  # 2D reductions (time x energy)
  - step: reduce_detector_ccm_temporal
    on: epix_ROI_1_simultaneous_laser
    timing_bin_key: timing_bin_indices_simultaneous_laser
    ccm_bin_key: ccm_bin_indices_simultaneous_laser
  - step: reduce_detector_ccm_temporal
    on: epix_ROI_1_xray_not_laser
    timing_bin_key: timing_bin_indices_xray_not_laser
    ccm_bin_key: ccm_bin_indices_xray_not_laser
  - step: reduce_detector_ccm_temporal
    on: ipm_simultaneous_laser
    timing_bin_key: timing_bin_indices_simultaneous_laser
    ccm_bin_key: ccm_bin_indices_simultaneous_laser
  - step: reduce_detector_ccm_temporal
    on: ipm_xray_not_laser
    timing_bin_key: timing_bin_indices_xray_not_laser
    ccm_bin_key: ccm_bin_indices_xray_not_laser

output:
  format: hdf5
  path: ./results/xcsl1004021/
```

### Static XES (no time-resolved, replaces XESBatchAnalysisRotation static mode)

```yaml
experiment:
  hutch: mfx
  experiment_id: mfx00000
  lcls_run: 1

data:
  runs: [1-10]
  keys:
    ipm4/sum: ipm
  detector_keys:
    epix_1/ROI_0_area:
      name: epix
      rois: [[100, 300], [400, 600]]
      combine_rois: false

pipeline:
  - step: filter_shots
    on: xray
    filter_key: ipm
    threshold: 2000
  - step: filter_detector_adu
    on: epix_ROI_1
    adu_threshold: 3.0
  - step: filter_detector_adu
    on: epix_ROI_2
    adu_threshold: 3.0
  - step: reduce_detector_shots
    on: epix_ROI_1
    reduction: sum
  - step: reduce_detector_shots
    on: epix_ROI_2
    reduction: sum
  - step: patch_pixels
    on: epix_ROI_1_reduced
    pixels: [45, 112]
    mode: interpolate
  - step: make_energy_axis
    n_pixels: 200
    crystal_detector_distance: 180.0
    crystal_radius: 250.0
    d_spacing: 1.637
    name: roi1
  - step: make_energy_axis
    n_pixels: 200
    crystal_detector_distance: 180.0
    crystal_radius: 250.0
    d_spacing: 1.92
    name: roi2

output:
  format: hdf5
  path: ./results/static_xes/
```

## Writing Custom Steps

Register new operations without touching the pipeline runner:

```python
from XSpect.analysis.registry import register_step

@register_step("my_custom_step")
def my_custom_step(run, **kwargs):
    """Custom analysis step.

    Parameters come from the YAML 'on' and other fields.
    Writes results to run attributes.
    """
    key = kwargs.get("on")
    scale = kwargs.get("scale", 1.0)

    data = getattr(run, key)
    result = data * scale
    setattr(run, f"{key}_scaled", result)
    run.update_status(f"Scaled {key} by {scale}")
```

Then use it in YAML:

```yaml
pipeline:
  - step: my_custom_step
    on: epix_ROI_1
    scale: 2.5
```

The step is importable as soon as the module containing it is imported. Put custom steps in a file and import it before calling `Pipeline.from_yaml()`:

```python
import my_custom_steps  # triggers @register_step decorators
from XSpect import Pipeline

pipe = Pipeline.from_yaml("analysis.yaml")
pipe.run(cores=8)
```

## How Data Flows

```
YAML file
    |
    v
parse_yaml() -> PipelineConfig (frozen dataclasses)
    |
    v
Pipeline.run(cores, batch_size)
    |
    v
For each run number in data.runs:
    1. Create spectroscopy_run (connects to HDF5)
    2. Load keys from data section
    3. Execute pipeline steps top-to-bottom
    4. Each step reads/writes run attributes
    |
    v
After all runs complete:
    Execute reduction steps (receive list of all runs)
    |
    v
Pipeline.results populated
```

## Parallelism

The YAML itself contains no parallelism directives. You control it at runtime:

```python
pipe.run(cores=1, batch_size=2000)    # sequential
pipe.run(cores=16, batch_size=2000)   # 16 parallel workers, each processing 2000 shots
pipe.run(cores=64, batch_size=500)    # more workers, smaller batches
```

When `cores > 1`, each run's shots are split into chunks of `batch_size`, processed in parallel via `multiprocessing.Pool`, and results are reconverged automatically.

## Validation

The config parser validates your YAML before any execution starts:

- Missing required sections (`experiment`, `data`, `pipeline`) raise `ConfigValidationError`
- Each pipeline entry must have a `step` field
- Each step name must exist in the registry (validates at parse time)
- Run ranges are expanded and validated

```python
from XSpect.controller.config_parser import parse_yaml, ConfigValidationError

try:
    config = parse_yaml("my_analysis.yaml")
except ConfigValidationError as e:
    print(f"Invalid config: {e}")
```

## Migration from Controller Subclasses

| Old pattern | New pattern |
|-------------|-------------|
| `controller = XESBatchAnalysis()` | `pipe = Pipeline.from_yaml("xes.yaml")` |
| `controller.runs = [162, 163, ...]` | `data: runs: [162-181]` in YAML |
| `controller.keys = [...]` | `data: keys: {...}` in YAML |
| `controller.rois = [[100, 600]]` | `data: detector_keys: epix: rois: [[100, 600]]` |
| `controller.mintime = -2` | `step: time_binning, bins: [-2, 10, 50]` |
| `controller.adu_cutoff = 3.0` | `step: filter_detector_adu, adu_threshold: 3.0` |
| `controller.crystal_d_space = 1.637` | `step: make_energy_axis, d_spacing: 1.637` |
| `controller.primary_analysis_loop(cores=16)` | `pipe.run(cores=16, batch_size=2000)` |

The old imports still work (`from XSpect.XSpect_Analysis import spectroscopy_run`). Nothing breaks until you decide to switch.

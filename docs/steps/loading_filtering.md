# Loading & shot selection

Steps that pull data off disk and steps that decide which shots survive. See
the [step reference overview](index.md) for the shape vocabulary and the
read/write model.

## Loading

### `load_run_keys`
Load scalar and 1D keys from the smalldata HDF5 into named run attributes.

- **Reads:** nothing on the run; pulls the listed HDF5 paths from `run.run_file`.
- **Writes:** one `(shots,)` attribute per key, named by `friendly_names`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `keys` | `[]` | list of HDF5 dataset paths |
| `friendly_names` | `[]` | attribute names to store them under (parallel to `keys`) |

```yaml
- step: load_run_keys
  keys: [ipm_dg2/sum, enc/lasDelay]
  friendly_names: [ipm, encoder]
```

### `load_detector`
Load a 3D detector stack (one image per shot) with optional crop and transpose.

- **Reads:** nothing on the run; reads the HDF5 paths (delayed).
- **Writes:** a `(shots, rows, cols)` attribute per detector, named by `friendly_names`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `keys` | `[]` | HDF5 detector paths |
| `friendly_names` | `[]` | attribute names |
| `transpose` | `False` | swap rows/cols on load |
| `rois` | `None` | list of `[start, end]` ranges to crop at load |
| `combine_rois` | `True` | merge ROIs into one array vs keep separate |

```yaml
- step: load_detector
  keys: [epix100/ROI_0_area]
  friendly_names: [epix]
  transpose: false
```

### `get_run_shot_properties`
Load the per-shot x-ray / laser status masks from `lightStatus`.

- **Reads:** `lightStatus` (via the run).
- **Writes:** `xray`, `laser`, `simultaneous` — each `(shots,)` bool.
- **Parameters:** none.

```yaml
- step: get_run_shot_properties
```

### `droplet_reconstruction`
Rebuild dense per-shot images from `droplet2photon` sparse photon positions,
reading directly from the source HDF5. Output plugs into every downstream
detector step.

- **Reads:** sparse photon arrays from `run.run_file`. In batch mode the batch
  manager injects `abs_start_index` / `abs_end_index`; otherwise the range comes
  from `run.start_index` / `run.end_index`.
- **Writes:** `new_key` — `(n_shots, rows, cols)`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `det` | required | detector group in the HDF5, e.g. `epix100_0` |
| `new_key` | required | attribute to store the reconstructed stack |
| `roi` | `None` | `[row0, row1, col0, col1]` crop; omit for full panel |
| `panel_shape` | `[704, 768]` | full panel `[rows, cols]` |

```yaml
- step: droplet_reconstruction
  det: epix100_0
  new_key: epix_spec
  roi: [270, 330, 400, 700]   # -> (60, 300) output
```

## Filtering & shot selection

### `filter_shots`
Tighten a shot mask by thresholding on another per-shot key. NaNs in the
filter key are dropped.

- **Reads:** the mask named by `on` `(shots,)`, and `filter_key` `(shots,)`.
- **Writes:** overwrites the mask `on` (fewer `True` entries).
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | shot mask key, e.g. `xray`, `simultaneous` |
| `filter_key` | `"ipm"` | per-shot key to threshold on |
| `threshold` | `1e4` | scalar (keep `> threshold`) or `[min, max]` (keep inside) |

```yaml
- step: filter_shots
  on: xray
  filter_key: ipm
  threshold: [2e4, 8e4]
```

### `filter_detector_adu`
Zero detector pixels outside an ADU window. Shape unchanged.

- **Reads:** detector `on` (any dimensionality).
- **Writes:** overwrites `on`; sub-threshold pixels set to 0.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `adu_threshold` | `3.0` | scalar (keep `> t`) or `[min, max]` (keep inside) |

```yaml
- step: filter_detector_adu
  on: epix
  adu_threshold: 5.0
```

### `filter_detector_variance`
Zero pixels whose value barely changes across shots (dead/hot/constant pixels).
Data-driven alternative to a hand-tuned ADU cut, using sklearn
`VarianceThreshold`.

- **Reads:** 3D detector `on` `(shots, rows, cols)` (flattened to shots × features).
- **Writes:** overwrites `on`; also `<on>_variance_mask` `(rows, cols)` bool of retained pixels.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `variance_threshold` | `0.0` | pixels with variance `<=` this are zeroed; `0.0` removes only constant pixels |

```yaml
- step: filter_detector_variance
  on: epix
  variance_threshold: 0.0
```

### `hitfinding`
Keep only shots whose total detector signal clears a threshold.

- **Reads:** 3D detector `on` `(shots, rows, cols)`.
- **Writes:** overwrites `on` with the surviving shots (first axis shrinks).
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `min_sum` | `None` | absolute per-shot ADU floor; takes precedence when set |
| `cutoff_multiplier` | `1.0` | relative threshold `median - k*std` when `min_sum` is unset |

Use `min_sum: 1.0` to drop shots that are all-zero after ADU filtering. The
relative mode breaks down when most shots are dark (median ≈ 0).

```yaml
- step: hitfinding
  on: epix
  min_sum: 1.0
```

### `union_shots`
Keep shots where ALL listed masks are true (logical AND).

- **Reads:** `on` (shot axis 0) and each mask in `filter_keys` `(shots,)`.
- **Writes:** `new_key` (default `<on>_<key1>_<key2>`), subset along shots.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | data/detector key to filter |
| `filter_keys` | `[]` | mask names to AND together |
| `new_key` | auto | output name; defaults to `<on>_<joined keys>` |

```yaml
- step: union_shots
  on: epix_ROI_1
  filter_keys: [simultaneous, laser]
```

### `separate_shots`
Keep shots matching the first mask but NOT the second (A and not B).

- **Reads:** `on` and the two masks in `filter_keys`.
- **Writes:** `new_key` (default `<on>_<A>_not_<B>`).
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | data/detector key |
| `filter_keys` | `[]` | `[include_mask, exclude_mask]` |
| `new_key` | auto | output name; defaults to `<on>_<A>_not_<B>` |

```yaml
- step: separate_shots
  on: epix_ROI_1
  filter_keys: [xray, laser]   # x-ray on, laser off
```

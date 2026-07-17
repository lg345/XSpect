# Detector correction, geometry & spatial reduction

Steps that clean up detector frames, straighten a tilted dispersion axis, and
collapse the spatial axes into a spectrum. See the
[step reference overview](index.md) for the shape vocabulary and the read/write
model.

## Correction

### `common_mode_correction`
Subtract a per-row, per-column, or per-bank baseline estimated from a
signal-free band. Shape preserved.

- **Reads:** detector `on` — 3D `(shots, rows, cols)` or 2D `(rows, cols)`.
- **Writes:** overwrites `on`; same shape.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `axis` | `"row"` | `row` (per-row offset across columns), `column`, or `bank` |
| `method` | `"median"` | `median` (robust) or `mean` |
| `reference` | full extent | `[start, end]` of the dark band, indexed on the axis orthogonal to `axis` |
| `bank_size` | `128` | column width of an ePix100 bank; only used when `axis: bank` |

The `reference` range is a column range for `axis: row` and a row range for
`axis: column` or `axis: bank`.

```yaml
- step: common_mode_correction
  on: epix
  axis: row
  reference: [0, 40]   # dark columns
```

### `patch_pixels`
Repair bad pixels or columns, either from an explicit list or auto-detected.

- **Reads:** detector `on` — 1D, 2D, or 3D.
- **Writes:** overwrites `on`; same shape.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `pixels` | `None` | explicit list of indices/columns to patch |
| `auto_detect` | `False` | find bad columns from the data instead of a list |
| `threshold` | `5.0` | auto-detect: deviation cut for flagging a column |
| `nsigma` | `5.0` | auto-detect: sigma multiplier for the flag |
| `max_gap_width` | `4` | widest run of bad pixels to bridge |
| `smooth_window` | `31` | window for the smoothed reference profile |
| `mode` | `"polynomial"` | `polynomial`, `interpolate`, or `zero` |
| `axis` | last | axis to patch along |
| `patch_range` | `4` | half-width of the neighborhood sampled around a bad pixel |
| `poly_range` | `6` | half-width of the fit window |
| `deg` | `1` | polynomial degree for `mode: polynomial` |

```yaml
- step: patch_pixels
  on: epix
  pixels: [383, 384]   # dead column pair
  mode: polynomial
```

## Geometry

### `find_rotation_angle`
Auto-detect the tilt of a dispersed signal so `rotate_detector` can straighten
it.

- **Reads:** detector `on` — 3D `(shots, rows, cols)` or 2D.
- **Writes:** `<on>_angle` (override with `angle_key`) — float degrees.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `low_threshold` | `30` | lower ADU bound for edge detection |
| `high_threshold` | `100` | upper ADU bound for edge detection |
| `angle_key` | `<on>_angle` | attribute to store the detected angle |

```yaml
- step: find_rotation_angle
  on: epix
  angle_key: epix_angle
```

### `rotate_detector`
Rotate detector frames by a fixed angle or by a previously detected one.

- **Reads:** detector `on` — 3D or 2D.
- **Writes:** overwrites `on`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `angle` | `0` | rotation in degrees |
| `angle_key` | `None` | read the angle from this attribute; takes precedence over `angle` |
| `axes` | `[1, 2]` 3D / `[0, 1]` 2D | plane to rotate in |
| `reshape` | `False` | grow the output to fit the rotated frame vs keep shape |

```yaml
- step: find_rotation_angle
  on: epix
  angle_key: epix_angle
- step: rotate_detector
  on: epix
  angle_key: epix_angle
```

## Spatial reduction

### `apply_roi`
Crop to one or more regions of interest, keeping the spatial dimension.

- **Reads:** detector `on` — 2D or 3D.
- **Writes:** `<on>_ROI_n` (one per ROI, or combined) — spatial axes retained.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `rois` | required | list of `[row0, row1, col0, col1]` crops |
| `combine_rois` | `True` | merge ROIs into one array vs keep separate `_ROI_n` |

```yaml
- step: apply_roi
  on: epix
  rois: [[270, 330, 400, 700]]
```

### `reduce_detector_spatial`
Crop to ROIs and collapse one spatial axis into a per-shot dispersion trace.

- **Reads:** detector `on` — 3D `(shots, rows, cols)` or 2D.
- **Writes:** `<on>_ROI_n` — reduced along the chosen axis.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `rois` | required | list of `[row0, row1, col0, col1]` crops |
| `combine_rois` | `True` | merge ROIs vs keep separate |
| `reduction` | `"sum"` | `sum` or `mean` over the reduced axis |
| `axis` | `1` (3D) / `-1` | spatial axis to collapse |
| `purge` | `True` | drop the source key after reducing to free memory |

ROI rows are given in absolute detector coordinates; the step translates them
into the loaded crop using the run's `_row_offset`, so the same YAML works
whether the detector was cropped at load or not.

```yaml
- step: reduce_detector_spatial
  on: epix
  rois: [[270, 330, 400, 700]]
  reduction: sum
```

### `reduce_detector_shots`
Collapse the shot axis into a single averaged or summed frame.

- **Reads:** detector `on` — shot axis is axis 0.
- **Writes:** `<on>_reduced` — shot axis removed.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `reduction` | `"sum"` | `sum` or `mean` over shots |
| `purge` | `True` | drop the source key after reducing |

```yaml
- step: reduce_detector_shots
  on: epix
  reduction: mean
```

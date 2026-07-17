# Spectra, utility & cross-run reductions

Steps that finish a binned spectrum (normalize, background-subtract), a utility
step to drop keys, and the one reduction that runs across all completed runs.
See the [step reference overview](index.md) for the shape vocabulary.

## Spectra

### `normalize_xes`
Area-normalize each spectrum so it sums to 1 over a pixel range.

- **Reads:** spectrum `on` — 1D `(pixels,)` or 2D `(bins, pixels)`.
- **Writes:** `<on>_normalized`; also `<on>_normalized_std` when a matching
  `_std` key exists.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | spectrum key (usually `_time_binned`) |
| `pixel_range` | full range | `[start, end]` pixels the sum is taken over |

For 2D input each row (bin) is divided by its own sum; zero-sum rows are left
unchanged. If `<on>` carries a companion `_std` array, it is scaled by the same
factor and written as `<on>_normalized_std`.

```yaml
- step: normalize_xes
  on: epix_ROI_1_time_binned
  pixel_range: [100, 400]
```

### `subtract_polynomial_background`
Fit a polynomial baseline along the spatial axis and subtract it.
Non-destructive.

- **Reads:** spectrum `on` — 1D `(pixels,)` or 2D `(bins, pixels)`.
- **Writes:** `<on>_bkgsub`; same shape.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | spectrum key |
| `axis` | last | spatial axis to fit along |
| `order` | `2` | polynomial degree |
| `background` | `None` | list of `[start, end]` signal-free ranges to fit |
| `peak_mask` | `None` | range(s) to EXCLUDE; single `[start, end]` or a list of ranges |

Give either `background` (the regions to fit) or `peak_mask` (the regions to
skip). `peak_mask` takes a list of ranges so multiple dispersed lines, e.g.
Kalpha and Kbeta on one detector, are masked together. `background` wins if
both are set.

```yaml
- step: subtract_polynomial_background
  on: epix_ROI_1_time_binned
  order: 2
  peak_mask: [[120, 180], [300, 360]]   # two emission lines
```

## Utility

### `purge_keys`
Set the listed run attributes to `None` to free memory mid-pipeline.

- **Reads:** nothing.
- **Writes:** sets each named attribute to `None`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `keys` | `[]` | list of attribute names to drop |

```yaml
- step: purge_keys
  keys: [epix, epix_ROI_1]
```

## Cross-run reduction

### `combine_runs`
A reduction, not a step: runs once after all per-run pipelines finish, summing
laser-on and laser-off binned data across runs and returning a results dict.

- **Signature:** `reduction(runs) -> dict` (operates on the list of completed
  runs, not a single `run`).
- **Reads (per run):** `<detector_key><laser_on_suffix>`,
  `<detector_key><laser_off_suffix>`, and their `_bincount` arrays.
- **Returns:** a dict with `laser_on_summed`, `laser_off_summed`,
  `laser_on_count`, `laser_off_count`, and, when all four are present,
  `laser_on_average`, `laser_off_average`, and `difference`
  `((on - off) / off)`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `detector_key` | `"epix_ROI_1"` | base detector key |
| `laser_on_suffix` | `"_simultaneous_laser_time_binned"` | laser-on data suffix |
| `laser_off_suffix` | `"_xray_not_laser_time_binned"` | laser-off data suffix |

Results land in `pipe.results` under the reduction name.

```yaml
reduction:
  - step: combine_runs
    detector_key: epix_ROI_1
```

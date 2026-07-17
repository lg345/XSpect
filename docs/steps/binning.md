# Axes, bin indices & binned reductions

Steps that build delay/energy axes, assign each shot to a bin, and collapse
per-shot data into a binned spectrum. Axis and index steps run first; the
`reduce_detector_*` steps consume the indices they produce. See the
[step reference overview](index.md) for the shape vocabulary.

## Building axes and bin indices

### `time_binning`
Compute per-shot delays from laser timing keys and lay down delay bins.

- **Reads:** `lxt_key`, `fast_delay_key`, `tt_correction_key` — each `(shots,)`.
  Uses whichever exist.
- **Writes:** `delays` `(shots,)`, `time_bins`, `time_bins_centered`,
  `timing_bin_indices` `(shots,)`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `bins` | required | `"auto"`, an explicit list of centers, or `[min, max, num_points]` |
| `lxt_key` | `"lxt_ttc"` | long-delay stage key; `null` to skip |
| `fast_delay_key` | `"encoder"` | fast-stage key |
| `tt_correction_key` | `"time_tool_correction"` | time-tool jitter correction key |
| `resolution` | `50e-15` | bin width in seconds for `bins: auto` |

### `make_ccm_axis`
Build incident-energy (CCM) bin edges and centers for an XAS scan.

- **Reads:** `ccm_key` `(shots,)` when `energies: auto`.
- **Writes:** `ccm_bins` — `n+1` edges; `ccm_energies` — `n` centers.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `energies` | `"auto"` | `"auto"`, explicit list, or `[min, max, num_points]` |
| `ccm_key` | `"ccm"` | per-shot incident-energy key |
| `resolution` | `0.001` | bin width (keV) for `energies: auto` |

### `ccm_binning`
Assign each shot to a CCM energy bin.

- **Reads:** `ccm_key` `(shots,)`, `ccm_bins_key` (edges).
- **Writes:** `ccm_bin_indices` `(shots,)`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `ccm_key` | `"ccm"` | per-shot incident-energy key |
| `ccm_bins_key` | `"ccm_bins"` | edges produced by `make_ccm_axis` |

### `bin_uniques`
Bin an arbitrary scan variable by its unique values (one bin per value).

- **Reads:** scan variable `on` `(shots,)`.
- **Writes:** `scanvar_indices` `(shots,)`, `scanvar_bins` (unique values).
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | per-shot scan variable to bin |

### `make_energy_axis`
Convert pixel index to emission energy from von Hamos crystal geometry.

- **Reads:** `detector_key` (to read pixel count) or `n_pixels` directly.
- **Writes:** `<name>_energy` `(pixels,)`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `crystal_detector_distance` | required | crystal-to-detector distance A (mm) |
| `crystal_radius` | required | crystal bend radius R (mm) |
| `d_spacing` | required | crystal d-spacing (angstrom) |
| `detector_key` | `None` | key to read pixel count from |
| `n_pixels` | `None` | pixel count; overrides `detector_key` |
| `mm_per_pixel` | `0.05` | pixel pitch (mm) |
| `name` | `"xes"` | output prefix, so the axis is `<name>_energy` |

## Binned reductions

Each reduction sums per-shot data into its bins. Pass `average: True` to divide
by the per-bin count; the raw count is always written as `<on>_..._bincount` for
downstream normalization or cross-run combination.

### `reduce_detector_temporal`
Bin a per-shot spectrum along delay.

- **Reads:** detector `on` — 1D `(shots,)` or 2D `(shots, pixels)`; plus
  `timing_bin_indices` and `time_bins`.
- **Writes:** `<on>_time_binned` `(n_time_bins[, pixels])`, `<on>_bincount`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `timing_bin_key` | `"timing_bin_indices"` | bin-index attribute |
| `average` | `False` | divide each bin by its shot count |

### `reduce_detector_ccm`
Bin a per-shot spectrum along incident energy.

- **Reads:** detector `on` — 1D/2D/3D; plus `ccm_bin_indices`.
- **Writes:** `<on>_energy_binned` `(n_energy[, ...])`, `<on>_energy_bincount`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `ccm_bin_key` | `"ccm_bin_indices"` | bin-index attribute |
| `average` | `False` | divide each bin by its shot count |

### `reduce_detector_ccm_temporal`
Bin a per-shot spectrum along both delay and incident energy (2D map).

- **Reads:** detector `on` — 1D/2D; plus `timing_bin_indices` and
  `ccm_bin_indices`.
- **Writes:** `<on>_time_energy_binned` `(n_time, n_energy[, pixels])`,
  `<on>_time_energy_bincount`.
- **Parameters:**

| name | default | description |
|------|---------|-------------|
| `on` | required | detector key |
| `timing_bin_key` | `"timing_bin_indices"` | delay bin-index attribute |
| `ccm_bin_key` | `"ccm_bin_indices"` | energy bin-index attribute |
| `average` | `False` | divide each bin by its shot count |

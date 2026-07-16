"""
XES (X-ray Emission Spectroscopy) specific pipeline steps.
"""

import numpy as np
from XSpect.analysis.registry import register_step


@register_step("normalize_xes")
def normalize_xes(run, **kwargs):
    """Normalize XES spectra by row sum over a pixel range.

    Divides each time bin's spectrum by its total intensity over the
    specified pixel range, producing area-normalized spectra.

    Parameters from YAML:
        on: detector key (time_binned data, shape: bins x pixels)
        pixel_range: [start, end] pixels for normalization (default: full range)
    """
    detector_key = kwargs.get("on")
    pixel_range = kwargs.get("pixel_range", None)
    if detector_key is None:
        return

    data = getattr(run, detector_key, None)
    if data is None:
        run.update_status(f"normalize_xes: {detector_key} not found")
        return

    if data.ndim == 2:
        if pixel_range:
            norm_slice = data[:, pixel_range[0] : pixel_range[1]]
        else:
            norm_slice = data

        row_sums = np.sum(norm_slice, axis=1)
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        normalized = data / safe_sums[:, np.newaxis]

        setattr(run, f"{detector_key}_normalized", normalized)

        # Propagate std if it exists
        std_key = detector_key.replace("_time_binned", "_std")
        std_data = getattr(run, std_key, None)
        if std_data is not None:
            std_normalized = std_data / safe_sums[:, np.newaxis]
            setattr(run, f"{detector_key}_normalized_std", std_normalized)

    elif data.ndim == 1:
        total = np.sum(data)
        if total > 0:
            normalized = data / total
        else:
            normalized = data
        setattr(run, f"{detector_key}_normalized", normalized)

    run.update_status(f"XES normalization: {detector_key} -> {detector_key}_normalized")


@register_step("make_energy_axis")
def make_energy_axis(run, **kwargs):
    """Generate energy axis from vonHamos spectrometer geometry.

    Uses the same formula as XSpect_Visualization.make_energy_axis:
        gl = pixel_indices * mm_per_pixel
        ll = gl/2 - (max(gl) - min(gl))/4
        energy = 12398.42 / (2 * d * sin(arctan(R / (ll + A))))
        energy = energy[::-1]

    Parameters from YAML:
        detector_key: key to get pixel count from (or use n_pixels directly)
        n_pixels: number of pixels (overrides detector_key shape)
        crystal_detector_distance: A (mm)
        crystal_radius: R (mm)
        d_spacing: d (Angstrom)
        mm_per_pixel: pixel pitch (default 0.05 mm for ePix)
        name: output attribute name prefix (default: "xes")
    """
    detector_key = kwargs.get("detector_key", None)
    n_pixels = kwargs.get("n_pixels", None)
    A = kwargs.get("crystal_detector_distance")
    R = kwargs.get("crystal_radius")
    d = kwargs.get("d_spacing")
    mm_per_pixel = kwargs.get("mm_per_pixel", 0.05)
    name = kwargs.get("name", "xes")

    if A is None or R is None or d is None:
        run.update_status("make_energy_axis: missing geometry parameters")
        return

    if n_pixels is None and detector_key is not None:
        det = getattr(run, detector_key, None)
        if det is not None:
            n_pixels = det.shape[-1] if det.ndim >= 1 else 100
        else:
            n_pixels = 100

    if n_pixels is None:
        n_pixels = 100

    hc = 12398.42  # eV * Angstrom
    gl = np.arange(n_pixels, dtype=np.float64) * mm_per_pixel
    ll = gl / 2.0 - (np.amax(gl) - np.amin(gl)) / 4.0
    energy_axis = hc / (2.0 * d * np.sin(np.arctan(R / (ll + A))))

    setattr(run, f"{name}_energy", energy_axis)
    run.update_status(
        f"Energy axis: {name}_energy ({n_pixels} pixels, center {energy_axis[n_pixels // 2]:.1f} eV)"
    )


@register_step("patch_pixels")
def patch_pixels(run, **kwargs):
    """Repair bad pixels using polynomial fitting from neighbors.

    Replicates XSpect_Analysis.patch_pixel with mode='polynomial':
    fits a weighted polynomial to surrounding pixels (excluding the bad
    pixel region) and evaluates at the bad pixel location.

    Supports automatic detection of ASIC panel-gap spikes via
    ``auto_detect: true``.  Uses a two-method approach:

    **Method 1 — Global ratio:** the column profile (sum over shots and
    rows) is compared to a median-filtered baseline.  Columns with
    ratio > ``threshold`` (bright spike) or ratio < 1/threshold (dead
    gap within signal region) are flagged.  Catches extreme outliers.

    **Method 2 — Z-score + width filter:** computes a robust z-score
    (residual / MAD-based local sigma) for each column.  Bright and dark
    outliers (|z| > ``nsigma``) are separately filtered to keep only
    narrow clusters (≤ ``max_gap_width`` columns).  This catches subtle
    ASIC gap columns (partially reduced intensity) that the ratio method
    misses — the ASIC gap pattern is a dark stripe (the actual gap)
    flanked by bright charge-sharing neighbors.  Filtering bright and
    dark separately avoids merging them into one wide cluster.

    The final set is the union of both methods, merged with any manually
    specified ``pixels``.

    Parameters from YAML:
        on:            detector key
        pixels:        list of pixel indices to patch (merged with auto if both given)
        auto_detect:   bool (default False) — auto-find spike/dead-gap columns
        threshold:     float (default 5.0) — ratio threshold for Method 1
                       (spike: ratio > threshold; dead: ratio < 1/threshold)
        nsigma:        float (default 5.0) — z-score threshold for Method 2
        max_gap_width: int (default 4) — maximum cluster width to keep in
                       Method 2; wider clusters are rejected as signal gradients
        smooth_window: int (default 31) — median-filter window for baseline;
                       must be >= the width of the widest expected spike
        mode:          "polynomial", "interpolate", or "zero" (default "polynomial")
        axis:          which axis the pixel indices refer to
                       (default: 0 for 2D, last axis for 3D)
        patch_range:   pixels on each side of bad pixel to exclude (default 4)
        poly_range:    additional pixels beyond patch_range for fitting (default 6)
        deg:           polynomial degree (default 1)
    """
    from scipy.ndimage import median_filter as _mf

    detector_key = kwargs.get("on")
    pixels = list(kwargs.get("pixels", []))
    mode = kwargs.get("mode", "polynomial")
    axis = kwargs.get("axis", None)
    patch_range = kwargs.get("patch_range", 4)
    poly_range = kwargs.get("poly_range", 6)
    deg = kwargs.get("deg", 1)
    auto_detect = kwargs.get("auto_detect", False)

    if detector_key is None:
        return

    data = getattr(run, detector_key, None)
    if data is None:
        return

    if axis is None:
        # Default: pixel indices refer to columns (last axis)
        # 1D: axis=0; 2D (rows×cols): axis=1; 3D (shots×rows×cols): axis=2
        axis = data.ndim - 1

    # ------------------------------------------------------------------ #
    # Auto-detection: find ASIC panel-gap spikes / dead-gap columns       #
    # ------------------------------------------------------------------ #
    manual_pixels = list(pixels)  # preserve user-specified pixels
    if auto_detect:
        from scipy.ndimage import label as _label

        smooth_win = int(kwargs.get("smooth_window", 31))
        threshold = float(kwargs.get("threshold", 5.0))
        nsigma = float(kwargs.get("nsigma", 5.0))
        max_gap_width = int(kwargs.get("max_gap_width", 4))

        # Collapse shot dimension if 3D → 2D (rows × cols)
        if data.ndim == 3:
            img = np.clip(data, 0, None).sum(axis=0).astype(float)
        elif data.ndim == 2:
            img = np.clip(data, 0, None).astype(float)
        else:
            img = None

        if img is not None and img.ndim == 2:
            col_profile = img.sum(axis=0)
        else:
            col_profile = np.clip(data, 0, None).astype(float)

        n_cols = len(col_profile)
        baseline = _mf(col_profile, size=smooth_win, mode="nearest")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(baseline > 0, col_profile / baseline, 0.0)
        in_signal = baseline > baseline.max() * 0.05

        # ── Method 1: Global ratio (extreme outliers) ────────────────
        ratio_spikes = set(np.where(ratio > threshold)[0])
        ratio_dead = set(np.where((ratio < 1.0 / threshold) & in_signal)[0])

        # ── Method 2: Z-score + width filter (subtle ASIC gaps) ──────
        # The column profile z-score catches columns whose intensity
        # deviates significantly from the local baseline.  Width filtering
        # then keeps only narrow defects (≤ max_gap_width cols), rejecting
        # broad signal gradients.  Bright and dark outliers are filtered
        # separately because ASIC gaps produce a dark stripe flanked by
        # bright charge-sharing neighbors; together they form a wide
        # cluster, but independently each is narrow.
        residual = col_profile - baseline
        abs_residual = np.abs(residual)
        local_mad = _mf(abs_residual, size=smooth_win * 2 + 1, mode="nearest")
        local_sigma = np.maximum(local_mad * 1.4826, baseline * 0.01)

        with np.errstate(divide="ignore", invalid="ignore"):
            z_score = np.where(local_sigma > 0, residual / local_sigma, 0.0)

        def _narrow_clusters(mask, max_w):
            """Keep only contiguous flagged regions ≤ max_w wide."""
            labeled_arr, n = _label(mask)
            out = np.zeros_like(mask)
            for i in range(1, n + 1):
                cluster = np.where(labeled_arr == i)[0]
                if (cluster.max() - cluster.min() + 1) <= max_w:
                    out[cluster] = True
            return out

        bright_mask = (z_score > nsigma) & in_signal
        dark_mask = (z_score < -nsigma) & in_signal

        zscore_spikes = set(np.where(_narrow_clusters(bright_mask, max_gap_width))[0])
        zscore_dead = set(np.where(_narrow_clusters(dark_mask, max_gap_width))[0])

        # ── Combine (union of both methods) ──────────────────────────
        spike_cols = sorted(ratio_spikes | zscore_spikes)
        dead_cols = sorted(ratio_dead | zscore_dead)
        auto_pixels = sorted(set(spike_cols + dead_cols))

        # Merge manual pixels with auto-detected ones
        pixels = sorted(set(manual_pixels + auto_pixels))

        # Store diagnostics on the run object for inspection
        setattr(run, f"{detector_key}_auto_patched_pixels", auto_pixels)
        setattr(run, f"{detector_key}_manual_patched_pixels", manual_pixels)
        setattr(run, f"{detector_key}_auto_spike_cols", spike_cols)
        setattr(run, f"{detector_key}_auto_dead_cols", dead_cols)
        setattr(run, f"{detector_key}_col_profile", col_profile)
        setattr(run, f"{detector_key}_col_baseline", baseline)
        setattr(run, f"{detector_key}_col_zscore", z_score)
        run.update_status(
            f"patch_pixels auto_detect: "
            f"{len(spike_cols)} spikes + {len(dead_cols)} dead = "
            f"{len(auto_pixels)} auto + {len(manual_pixels)} manual = "
            f"{len(pixels)} total on {detector_key} "
            f"(threshold={threshold}, nsigma={nsigma}, "
            f"max_gap_width={max_gap_width})"
        )

    if not pixels:
        return

    n_pixels = data.shape[axis]
    bad_set = set(pixels)  # all detected bad columns (for exclusion)

    for pixel in pixels:
        if pixel < 0 or pixel >= n_pixels:
            continue

        if mode == "zero":
            slc = [slice(None)] * data.ndim
            slc[axis] = pixel
            data[tuple(slc)] = 0
        elif mode in ("polynomial", "interpolate"):
            start = pixel - patch_range - poly_range
            end = pixel + patch_range + poly_range + 1
            actual_start = max(start, 0)
            actual_end = min(end, n_pixels)
            slc_region = [slice(None)] * data.ndim
            slc_region[axis] = slice(actual_start, actual_end)
            region = np.moveaxis(data[tuple(slc_region)], axis, 0)

            patch_x = np.arange(actual_start, actual_end)
            weights = np.ones(len(patch_x))

            # Build exclusion mask: zero-weight the target pixel's
            # neighborhood AND the ±patch_range neighborhood of every
            # other bad pixel in the region.  This ensures the fit
            # anchors only on data well outside the entire ASIC feature
            # (gap + charge-sharing halo), preventing overshoot from
            # elevated neighbors.
            for bad_col in bad_set:
                idx_in_region = bad_col - actual_start
                lo = max(idx_in_region - patch_range, 0)
                hi = min(idx_in_region + patch_range + 1, len(weights))
                if lo < len(weights) and hi > 0:
                    weights[lo:hi] = 0.0

            # If too few non-zero weights remain, expand the fit region
            good_count = np.sum(weights > 0.5)
            if good_count < 4:
                # Expand region symmetrically until we get enough clean pixels
                expand = poly_range
                while good_count < 4 and expand < n_pixels // 2:
                    expand += poly_range
                    exp_start = max(pixel - patch_range - expand, 0)
                    exp_end = min(pixel + patch_range + expand + 1, n_pixels)
                    slc_region = [slice(None)] * data.ndim
                    slc_region[axis] = slice(exp_start, exp_end)
                    region = np.moveaxis(data[tuple(slc_region)], axis, 0)
                    patch_x = np.arange(exp_start, exp_end)
                    weights = np.ones(len(patch_x))
                    # Zero-weight ±patch_range around ALL bad pixels
                    for bad_col in bad_set:
                        idx_in_region = bad_col - exp_start
                        lo = max(idx_in_region - patch_range, 0)
                        hi = min(idx_in_region + patch_range + 1, len(weights))
                        if lo < len(weights) and hi > 0:
                            weights[lo:hi] = 0.0
                    good_count = np.sum(weights > 0.5)
                    actual_start, actual_end = exp_start, exp_end

            if data.ndim == 1:
                coeffs = np.polyfit(patch_x, region, deg, w=weights)
                new_val = np.polyval(coeffs, pixel)
            else:
                # Vectorized weighted polynomial fit across all (shot, row)
                # pairs simultaneously.  The projection vector `proj` maps
                # the region slice to the interpolated value at `pixel` —
                # it only depends on patch_x, weights, and deg (same for
                # all columns of `flat`), so we compute it once and apply
                # via a single matrix-vector multiply.
                other_shape = region.shape[1:]
                flat = region.reshape(region.shape[0], -1).astype(
                    np.float64
                )  # (n_region, N_total)

                # Build Vandermonde matrix and solve for projection vector.
                # Square the weights to match numpy.polyfit convention
                # (polyfit treats w as 1/sigma, minimising sum(w**2 * r**2)).
                X = np.vander(patch_x.astype(np.float64), deg + 1)
                W = (weights**2).astype(np.float64)
                XtW = X.T * W[np.newaxis, :]  # (deg+1, n_region)
                XtWX = XtW @ X  # (deg+1, deg+1)
                XtWX_inv_XtW = np.linalg.solve(XtWX, XtW)  # (deg+1, n_region)
                x_eval = np.vander(np.array([float(pixel)]), deg + 1)[0]  # (deg+1,)
                proj = x_eval @ XtWX_inv_XtW  # (n_region,)

                # Single dot product replaces the per-column polyfit loop
                new_vals = proj @ flat  # (N_total,)
                new_val = new_vals.reshape(other_shape)

            slc = [slice(None)] * data.ndim
            slc[axis] = pixel
            data[tuple(slc)] = new_val

    setattr(run, detector_key, data)
    run.update_status(
        f"Patched {len(pixels)} pixels on {detector_key} (axis={axis}, mode={mode})"
    )

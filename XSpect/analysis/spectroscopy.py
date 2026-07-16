"""
Core spectroscopy pipeline steps.

Each function is registered via @register_step and follows the contract:
    step(run, **kwargs) -> None
All results written to run.results[key].
"""

import numpy as np
from scipy.ndimage import rotate
from XSpect.analysis.registry import register_step, register_reduction


@register_step("load_run_keys")
def load_run_keys(run, **kwargs):
    """Load scalar/1D keys from HDF5 into run attributes.

    Reads the keys defined in the pipeline's data section.
    Expects run to have run_file, start_index, end_index set.
    """
    keys = kwargs.get("keys", [])
    friendly_names = kwargs.get("friendly_names", [])
    if not keys:
        return
    run.load_run_keys(keys, friendly_names)


@register_step("load_detector")
def load_detector(run, **kwargs):
    """Load 3D detector data (shots x rows x cols) with optional ROI and transpose.

    Parameters from YAML:
        keys: list of HDF5 paths
        friendly_names: list of attribute names
        transpose: bool (default False)
        rois: list of [start, end] pixel ranges (default None)
        combine_rois: bool (default True)
    """
    keys = kwargs.get("keys", [])
    friendly_names = kwargs.get("friendly_names", [])
    transpose = kwargs.get("transpose", False)
    rois = kwargs.get("rois", None)
    combine = kwargs.get("combine_rois", True)
    if not keys:
        return
    run.load_run_key_delayed(
        keys, friendly_names, transpose=transpose, rois=rois, combine=combine
    )


@register_step("get_run_shot_properties")
def get_run_shot_properties(run, **kwargs):
    """Load xray/laser/simultaneous boolean masks from lightStatus."""
    run.get_run_shot_properties()


@register_step("filter_shots")
def filter_shots(run, **kwargs):
    """Filter a shot mask by thresholding on another key.

    Parameters from YAML:
        on: shot mask key (e.g., "xray", "simultaneous")
        filter_key: key to threshold on (e.g., "ipm")
        threshold: float or [min, max]
    """
    shot_mask_key = kwargs.get("on")
    filter_key = kwargs.get("filter_key", "ipm")
    threshold = kwargs.get("threshold", 1.0e4)
    if shot_mask_key is None:
        return

    shot_mask = getattr(run, shot_mask_key)
    count_before = np.sum(shot_mask)
    filter_data = getattr(run, filter_key)
    nan_mask = np.isnan(filter_data)

    if isinstance(threshold, (int, float)):
        filtered = shot_mask * (filter_data > threshold) * (~nan_mask)
    elif len(threshold) == 2:
        filtered = (
            shot_mask
            * (filter_data > threshold[0])
            * (filter_data < threshold[1])
            * (~nan_mask)
        )
    else:
        filtered = shot_mask

    setattr(run, shot_mask_key, filtered)
    count_after = np.sum(filtered)
    run.update_status(
        f"Filtered {shot_mask_key} on {filter_key}: {int(count_before - count_after)} shots removed"
    )


@register_step("filter_detector_adu")
def filter_detector_adu(run, **kwargs):
    """Zero out detector pixels below ADU threshold.

    Parameters from YAML:
        on: detector key
        adu_threshold: float or [min, max] (default 3.0)
    """
    detector_key = kwargs.get("on")
    adu_threshold = kwargs.get("adu_threshold", 3.0)
    if detector_key is None:
        return

    detector_images = getattr(run, detector_key, None)
    if detector_images is None:
        return
    if isinstance(adu_threshold, list):
        filtered = detector_images * (detector_images > adu_threshold[0])
        filtered = filtered * (filtered < adu_threshold[1])
    else:
        filtered = detector_images * (detector_images > adu_threshold)

    setattr(run, detector_key, filtered)
    run.update_status(f"ADU filtered {detector_key} with threshold {adu_threshold}")


@register_step("filter_detector_variance")
def filter_detector_variance(run, **kwargs):
    """Zero out low-variance detector pixels using sklearn VarianceThreshold.

    Data-driven alternative to filter_detector_adu: instead of a hand-tuned
    intensity cutoff, pixels whose value barely changes across shots (dead
    pixels, static hot pixels, constant background) are dropped. Signal-bearing
    pixels vary shot to shot and are retained.

    Parameters from YAML:
        on: detector key
        variance_threshold: float (default 0.0). Pixels with variance <= this
            across shots are zeroed. 0.0 removes only constant pixels.

    Detector data is treated as (shots, features): a 3D array
    (shots x rows x cols) is flattened per shot, filtered, then reshaped back.
    Writes the filtered array back to detector_key and stores the retained
    boolean mask as <detector_key>_variance_mask.
    """
    from sklearn.feature_selection import VarianceThreshold

    detector_key = kwargs.get("on")
    variance_threshold = kwargs.get("variance_threshold", 0.0)
    if detector_key is None:
        return

    detector_images = getattr(run, detector_key, None)
    if detector_images is None:
        return

    detector_images = np.asarray(detector_images)
    original_shape = detector_images.shape
    n_shots = original_shape[0]
    flat = detector_images.reshape(n_shots, -1)

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(flat)
    keep_mask = selector.get_support()

    filtered = flat * keep_mask  # broadcast over shots, zero the dropped pixels
    setattr(run, detector_key, filtered.reshape(original_shape))
    setattr(run, f"{detector_key}_variance_mask", keep_mask.reshape(original_shape[1:]))

    n_removed = int(np.sum(~keep_mask))
    run.update_status(
        f"Variance filtered {detector_key} (threshold={variance_threshold}): "
        f"{n_removed}/{keep_mask.size} pixels zeroed"
    )


@register_step("find_rotation_angle")
def find_rotation_angle(run, **kwargs):
    """Auto-detect the tilt angle of a dispersed spectral signal.

    Delegates to XSpectDetectorProcessor which uses:
      1. Canny edge detection on the 8-bit normalised sum image
      2. DBSCAN clustering to isolate distinct signal streaks
      3. PCA per cluster to find each streak's orientation
      4. Mean angle across clusters, clamped to [-90, 90]

    The result (degrees) is stored as ``run.<angle_key>`` for use by
    rotate_detector.

    Parameters from YAML:
        on: detector key (3D shots x rows x cols, or 2D rows x cols)
        low_threshold:  Canny lower hysteresis threshold (default 30)
        high_threshold: Canny upper hysteresis threshold (default 100)
        angle_key: attribute name to store the result (default: <on>_angle)
    """
    from XSpect.XSpect_Processor.XSpectDetectorProcessor import XSpectDetectorProcessor

    detector_key = kwargs.get("on")
    low_threshold = kwargs.get("low_threshold", 30)
    high_threshold = kwargs.get("high_threshold", 100)
    angle_key = kwargs.get("angle_key", f"{detector_key}_angle")

    if detector_key is None:
        return
    data = getattr(run, detector_key, None)
    if data is None:
        return

    # Build 2D sum image, clamp negatives
    img = data.sum(axis=0) if data.ndim == 3 else data.copy()
    img = np.clip(img, 0, None).astype(float)

    try:
        proc = XSpectDetectorProcessor(img)
        proc.detect_edges(low_threshold=low_threshold, high_threshold=high_threshold)
        angle = proc.find_optimal_rotation_angle()
    except Exception as e:
        run.update_status(f"find_rotation_angle: failed for {detector_key}: {e}")
        return

    setattr(run, angle_key, float(angle))
    run.update_status(
        f"find_rotation_angle: {detector_key} -> {angle_key} = {angle:.3f} deg"
    )


@register_step("rotate_detector")
def rotate_detector(run, **kwargs):
    """Rotate detector images by a fixed or auto-detected angle (scipy.ndimage.rotate).

    Applied per-shot. For 3D data (shots x rows x cols), rotates in the
    (rows, cols) plane. For 2D data (rows x cols), rotates directly.
    Skipped when the resolved angle is 0.

    Parameters from YAML:
        on: detector key
        angle: rotation in degrees (positive = CCW). Mutually exclusive with angle_key.
        angle_key: name of a run attribute holding the angle (set by find_rotation_angle).
                   Takes precedence over angle if both are provided.
        axes: rotation plane as [ax1, ax2] (default [1, 2] for 3D, [0, 1] for 2D)
        reshape: bool (default False) — keep original array shape after rotation
    """
    detector_key = kwargs.get("on")
    angle_key = kwargs.get("angle_key", None)
    axes = kwargs.get("axes", None)
    if detector_key is None:
        return

    # Resolve angle: dynamic key takes precedence over static value
    if angle_key is not None:
        angle = getattr(run, angle_key, None)
        if angle is None:
            run.update_status(
                f"rotate_detector: angle_key '{angle_key}' not found, skipping"
            )
            return
    else:
        angle = kwargs.get("angle", 0)

    if angle == 0:
        return

    data = getattr(run, detector_key, None)
    if data is None:
        return

    if axes is None:
        axes = [0, 1] if data.ndim == 2 else [1, 2]

    # Static analysis path (2D) uses axes=[0,1]; per-shot 3D uses [1,2]
    # The old XESBatchAnalysisRotation uses [0,1] for the static case
    if data.ndim == 3 and axes == [0, 1]:
        axes = [1, 2]

    reshape = kwargs.get("reshape", False)
    rotated = rotate(data, angle=angle, axes=axes, reshape=reshape)
    setattr(run, detector_key, rotated)
    run.update_status(
        f"Rotated {detector_key} by {angle:.2f} degrees (axes={axes}, reshape={reshape})"
    )


@register_step("union_shots")
def union_shots(run, **kwargs):
    """Combine shots matching multiple boolean masks (logical AND).

    Parameters from YAML:
        on: detector/data key to filter
        filter_keys: list of mask attribute names to AND together
        new_key: optional output key name (default: auto-generated)
    """
    detector_key = kwargs.get("on")
    filter_keys = kwargs.get("filter_keys", [])
    new_key = kwargs.get("new_key", None)
    if detector_key is None or not filter_keys:
        return

    detector = getattr(run, detector_key, None)
    if detector is None:
        return
    combined_mask = np.ones(detector.shape[0], dtype=bool)
    for fk in filter_keys:
        mask = getattr(run, fk)
        combined_mask = combined_mask * mask.astype(bool)

    if detector.ndim == 1:
        filtered_data = detector[combined_mask]
    elif detector.ndim == 2:
        filtered_data = detector[combined_mask]
    elif detector.ndim == 3:
        filtered_data = detector[combined_mask]
    else:
        filtered_data = detector[combined_mask]

    if new_key is None:
        new_key = f"{detector_key}_{'_'.join(filter_keys)}"
    setattr(run, new_key, filtered_data)
    run.update_status(
        f"Union shots: {detector_key} with masks {filter_keys} -> {new_key} ({int(np.sum(combined_mask))} shots)"
    )


@register_step("separate_shots")
def separate_shots(run, **kwargs):
    """Extract shots matching first mask but NOT second (A and not B).

    Parameters from YAML:
        on: detector/data key
        filter_keys: [include_mask, exclude_mask]
        new_key: optional output key name
    """
    detector_key = kwargs.get("on")
    filter_keys = kwargs.get("filter_keys", [])
    new_key = kwargs.get("new_key", None)
    if detector_key is None or len(filter_keys) < 2:
        return

    detector = getattr(run, detector_key)
    include_mask = getattr(run, filter_keys[0]).astype(bool)
    exclude_mask = getattr(run, filter_keys[1]).astype(bool)
    separation_mask = include_mask & (~exclude_mask)

    filtered_data = detector[separation_mask]

    if new_key is None:
        new_key = f"{detector_key}_{filter_keys[0]}_not_{filter_keys[1]}"
    setattr(run, new_key, filtered_data)
    run.update_status(
        f"Separated shots: {detector_key} ({filter_keys[0]} not {filter_keys[1]}) -> {new_key} ({int(np.sum(separation_mask))} shots)"
    )


@register_step("reduce_detector_spatial")
def reduce_detector_spatial(run, **kwargs):
    """Reduce spatial dimension of detector using ROIs.

    For 3D data (shots x rows x cols), the ROI selects along one spatial axis
    (default: axis 1 = cross-dispersion) and reduces by summing/averaging,
    producing (shots x remaining_spatial).

    Parameters from YAML:
        on: detector key
        rois: list of [start, end] pixel ranges
        combine_rois: bool (default True)
        reduction: "sum" or "mean" (default "sum")
        axis: which spatial axis to apply ROI on (default 1 for 3D, -1 for 2D)
        purge: bool (default True) - delete original after reduction
    """
    detector_key = kwargs.get("on")
    rois = kwargs.get("rois", [[0, None]])
    combine = kwargs.get("combine_rois", True)
    reduction_name = kwargs.get("reduction", "sum")
    roi_axis = kwargs.get("axis", None)
    purge = kwargs.get("purge", True)
    if detector_key is None:
        return

    reduction_fn = np.nansum if reduction_name == "sum" else np.nanmean
    detector = getattr(run, detector_key, None)
    if detector is None:
        return

    if roi_axis is None:
        roi_axis = 1 if detector.ndim == 3 else -1

    axis_size = detector.shape[roi_axis]

    # If the detector was cropped at import (row_range), the ROIs in the YAML
    # are in ABSOLUTE (full-frame) coordinates but the array now starts at the
    # crop origin. Translate ROIs into cropped-array coordinates automatically
    # so the YAML can always use absolute row numbers.
    # The offset is keyed by the loaded detector name (e.g. "epix"); derived
    # keys like "epix_reduced" inherit it by stripping known suffixes.
    row_offset = 0
    offsets = getattr(run, "_row_offset", None)
    if offsets:
        candidate = detector_key
        for suffix in ("_reduced", "_ROI_1", "_ROI_2"):
            candidate = candidate.replace(suffix, "")
        row_offset = offsets.get(detector_key, offsets.get(candidate, 0))
    if row_offset:
        adjusted = []
        for roi in rois:
            start = roi[0] - row_offset
            end = (roi[1] - row_offset) if roi[1] is not None else None
            adjusted.append([max(0, start), end])
        run.update_status(
            f"reduce_detector_spatial: applied row_range offset {row_offset} "
            f"to ROIs {rois} -> {adjusted}"
        )
        rois = adjusted

    if combine:
        mask = np.zeros(axis_size, dtype=bool)
        for roi in rois:
            end = roi[1] if roi[1] is not None else axis_size
            mask[roi[0] : end] = True

        idx = [slice(None)] * detector.ndim
        idx[roi_axis] = mask
        masked_data = detector[tuple(idx)]

        reduced = reduction_fn(masked_data, axis=roi_axis)
        setattr(run, f"{detector_key}_ROI_1", reduced)
        run.update_status(f"Spatial reduction: {detector_key} -> {detector_key}_ROI_1")
    else:
        for roi_idx, roi in enumerate(rois):
            end = roi[1] if roi[1] is not None else axis_size
            idx = [slice(None)] * detector.ndim
            idx[roi_axis] = slice(roi[0], end)
            chunk = detector[tuple(idx)]
            reduced = reduction_fn(chunk, axis=roi_axis)
            setattr(run, f"{detector_key}_ROI_{roi_idx + 1}", reduced)
            run.update_status(
                f"Spatial reduction: {detector_key} -> {detector_key}_ROI_{roi_idx + 1}"
            )

    if purge:
        setattr(run, detector_key, None)


@register_step("apply_roi")
def apply_roi(run, **kwargs):
    """Extract ROI without spatial reduction (keep spatial dimension).

    Parameters from YAML:
        on: detector key
        rois: list of [start, end] pixel ranges
        combine_rois: bool (default True)
    """
    detector_key = kwargs.get("on")
    rois = kwargs.get("rois", [[0, None]])
    combine = kwargs.get("combine_rois", True)
    if detector_key is None:
        return

    detector = getattr(run, detector_key, None)
    if detector is None:
        return

    if combine:
        mask = np.zeros(detector.shape[-1], dtype=bool)
        for roi in rois:
            end = roi[1] if roi[1] is not None else detector.shape[-1]
            mask[roi[0] : end] = True

        if detector.ndim == 3:
            masked = detector[:, :, mask]
        elif detector.ndim == 2:
            masked = detector[:, mask]
        else:
            masked = detector[mask]
        setattr(run, f"{detector_key}_ROI_1", masked)
    else:
        for idx, roi in enumerate(rois):
            end = roi[1] if roi[1] is not None else detector.shape[-1]
            if detector.ndim == 3:
                chunk = detector[:, :, roi[0] : end]
            else:
                chunk = detector[:, roi[0] : end]
            setattr(run, f"{detector_key}_ROI_{idx + 1}", chunk)

    run.update_status(f"Applied ROI to {detector_key}")


@register_step("time_binning")
def time_binning(run, **kwargs):
    """Create time delay bins from laser timing data.

    Parameters from YAML:
        bins: "auto", list, or [min, max, num_points]
        lxt_key: str or null (default "lxt_ttc"; null means use encoder directly)
        fast_delay_key: str (default "encoder")
        tt_correction_key: str (default "time_tool_correction")
        resolution: bin width for auto mode (default 50e-15, i.e. 50 fs)
    """
    bins_spec = kwargs.get("bins")
    lxt_key = kwargs.get("lxt_key", "lxt_ttc")
    fast_delay_key = kwargs.get("fast_delay_key", "encoder")
    tt_correction_key = kwargs.get("tt_correction_key", "time_tool_correction")
    resolution = kwargs.get("resolution", 50e-15)

    if bins_spec is None:
        return

    resolution = float(resolution)

    lxt_raw = getattr(run, lxt_key, None) if lxt_key else None
    fast_delay_raw = getattr(run, fast_delay_key, None) if fast_delay_key else None
    tt_correction_raw = (
        getattr(run, tt_correction_key, None) if tt_correction_key else None
    )

    lxt = np.asarray(lxt_raw, dtype=np.float64) if lxt_raw is not None else None
    fast_delay = (
        np.asarray(fast_delay_raw, dtype=np.float64)
        if fast_delay_raw is not None
        else None
    )
    tt_correction = (
        np.asarray(tt_correction_raw, dtype=np.float64)
        if tt_correction_raw is not None
        else None
    )

    if lxt is not None:
        delays = lxt.copy()
        if fast_delay is not None:
            if np.mean(np.abs(fast_delay)) > 1e-3:
                delays = delays + fast_delay
        if tt_correction is not None:
            delays = delays + tt_correction
    elif fast_delay is not None:
        delays = fast_delay.copy()
        if tt_correction is not None:
            delays = delays + tt_correction
    elif tt_correction is not None:
        delays = tt_correction.copy()
    else:
        run.update_status("time_binning: no timing keys found")
        return

    if bins_spec == "auto":
        rounded = np.round(delays / resolution) * resolution
        bins = np.unique(rounded)
    elif isinstance(bins_spec, list) and len(bins_spec) == 3:
        bins = np.linspace(bins_spec[0], bins_spec[1], int(bins_spec[2]))
    else:
        bins = np.array(bins_spec)

    bin_edges, _ = _center_binning(delays, bins)
    indices = np.digitize(delays, bin_edges)

    run.delays = delays
    run.time_bins = bins
    run.time_bins_centered = (bins[:-1] + bins[1:]) / 2 if len(bins) > 1 else bins
    run.timing_bin_indices = indices
    run.update_status(
        f"Time binning complete: {len(bins)} bins from {bins[0]:.4g} to {bins[-1]:.4g}"
    )


def _center_binning(data, bins):
    """Create centered bin edges from bin centers."""
    bin_edges = np.empty(len(bins) + 1)
    for i in range(len(bins)):
        if i == 0:
            bin_edges[i] = bins[i] - (bins[1] - bins[0]) / 2
        else:
            bin_edges[i] = bins[i] - (bins[i] - bins[i - 1]) / 2
    bin_edges[-1] = bins[-1] + (bins[-1] - bins[-2]) / 2

    indices = np.digitize(data, bin_edges)
    return bin_edges, indices


@register_step("reduce_detector_temporal")
def reduce_detector_temporal(run, **kwargs):
    """Bin detector data along time dimension using timing indices.

    Parameters from YAML:
        on: detector key (2D: shots x pixels)
        timing_bin_key: attribute name for bin indices (default "timing_bin_indices")
        average: bool (default False) - if True, divide by bin count
    """
    detector_key = kwargs.get("on")
    timing_bin_key = kwargs.get("timing_bin_key", "timing_bin_indices")
    average = kwargs.get("average", False)
    if detector_key is None:
        return

    detector = getattr(run, detector_key, None)
    timing_indices = getattr(run, timing_bin_key, None)
    time_bins = getattr(run, "time_bins", None)

    if detector is None or timing_indices is None or time_bins is None:
        run.update_status(f"reduce_detector_temporal: missing data for {detector_key}")
        return

    n_bins = len(time_bins)

    if detector.ndim == 1:
        binned = np.zeros(n_bins)
        bincount = np.zeros(n_bins)
        for i in range(len(detector)):
            idx = timing_indices[i] - 1
            if 0 <= idx < n_bins and not np.isnan(detector[i]):
                binned[idx] += detector[i]
                bincount[idx] += 1
    elif detector.ndim == 2:
        n_pixels = detector.shape[1]
        binned = np.zeros((n_bins, n_pixels))
        bincount = np.zeros(n_bins)
        for i in range(detector.shape[0]):
            idx = timing_indices[i] - 1
            if 0 <= idx < n_bins:
                row = detector[i]
                if not np.all(np.isnan(row)):
                    binned[idx] += np.where(np.isnan(row), 0.0, row)
                    bincount[idx] += 1
    else:
        run.update_status(f"reduce_detector_temporal: unsupported ndim={detector.ndim}")
        return

    if average:
        safe_count = np.where(bincount > 0, bincount, 1)
        if binned.ndim == 2:
            binned = binned / safe_count[:, np.newaxis]
        else:
            binned = binned / safe_count

    setattr(run, f"{detector_key}_time_binned", binned)
    setattr(run, f"{detector_key}_bincount", bincount)
    run.update_status(
        f"Temporal reduction: {detector_key} -> {detector_key}_time_binned ({n_bins} bins)"
    )


@register_step("hitfinding")
def hitfinding(run, **kwargs):
    """Filter shots by total signal intensity (hit detection).

    Keeps shots whose per-shot detector sum exceeds a threshold.
    Two modes (applied after any ADU filtering upstream):

      min_sum (preferred for sparse/XES data):
        Keep shots where sum(detector) > min_sum.
        Use min_sum=1.0 to reject shots that are completely zero after
        the ADU threshold — i.e. true dark shots with no photons.

      cutoff_multiplier (legacy, relative threshold):
        threshold = median(sums) - cutoff_multiplier * std(sums)
        Works well when signal shots are the majority; breaks down when
        most shots are dark (median ≈ 0 → threshold is negative).

    If both are supplied, min_sum takes precedence.

    Parameters from YAML:
        on:                 detector key (3D: shots × rows × cols)
        min_sum:            absolute ADU floor per shot (default None)
        cutoff_multiplier:  std multiplier for relative threshold (default 1.0)
    """
    detector_key = kwargs.get("on")
    min_sum = kwargs.get("min_sum", None)
    cutoff_multiplier = kwargs.get("cutoff_multiplier", 1.0)
    if detector_key is None:
        return

    detector = getattr(run, detector_key, None)
    if detector is None or detector.ndim < 3:
        return

    sum_images = np.sum(detector, axis=(1, 2))

    if min_sum is not None:
        threshold = float(min_sum)
    else:
        threshold = np.median(sum_images) - cutoff_multiplier * np.std(sum_images)

    hits = np.where(sum_images > threshold)[0]

    if len(hits) == 0:
        setattr(
            run, detector_key, np.zeros((0,) + detector.shape[1:], dtype=detector.dtype)
        )
    elif len(hits) < detector.shape[0]:
        setattr(run, detector_key, detector[hits])
    run.update_status(
        f"Hitfinding on {detector_key}: {len(hits)}/{detector.shape[0]} shots kept "
        f"(threshold={threshold:.1f}, mode={'min_sum' if min_sum is not None else 'relative'})"
    )


@register_step("reduce_detector_shots")
def reduce_detector_shots(run, **kwargs):
    """Collapse the shot dimension using sum/mean.

    Parameters from YAML:
        on: detector key
        reduction: "sum" or "mean" (default "sum")
        purge: bool (default True)
    """
    detector_key = kwargs.get("on")
    reduction_name = kwargs.get("reduction", "sum")
    purge = kwargs.get("purge", True)
    if detector_key is None:
        return

    reduction_fn = np.nansum if reduction_name == "sum" else np.nanmean
    detector = getattr(run, detector_key, None)
    if detector is None:
        return

    reduced = reduction_fn(detector, axis=0)
    setattr(run, f"{detector_key}_reduced", reduced)
    if purge:
        setattr(run, detector_key, None)
    run.update_status(f"Shot reduction: {detector_key} -> {detector_key}_reduced")


@register_step("bin_uniques")
def bin_uniques(run, **kwargs):
    """Bin unique scan variable values for scan-based analysis.

    Parameters from YAML:
        on: key with scan variable values
    """
    key = kwargs.get("on")
    if key is None:
        return

    vals = getattr(run, key, None)
    if vals is None:
        return

    bins = np.unique(vals)
    addon = (bins[-1] - bins[-2]) / 2
    bins2 = np.append(bins, bins[-1] + addon)
    bins_center = np.empty_like(bins2)
    for ii in range(len(bins)):
        if ii == 0:
            bins_center[ii] = bins2[ii] - (bins2[ii + 1] - bins2[ii]) / 2
        else:
            bins_center[ii] = bins2[ii] - (bins2[ii] - bins2[ii - 1]) / 2
    bins_center[-1] = bins2[-1]

    run.scanvar_indices = np.digitize(vals, bins_center)
    run.scanvar_bins = bins_center
    run.update_status(f"Binned uniques on {key}: {len(bins)} unique values")


@register_step("purge_keys")
def purge_keys(run, **kwargs):
    """Delete specified attributes from run to free memory.

    Parameters from YAML:
        keys: list of attribute names to purge
    """
    keys = kwargs.get("keys", [])
    for key in keys:
        if hasattr(run, key):
            setattr(run, key, None)
    run.update_status(f"Purged keys: {keys}")


@register_reduction("combine_runs")
def combine_runs(runs, **kwargs):
    """Aggregate time-binned data across multiple runs.

    Sums the time-binned detector data and bin counts from all runs.
    Returns normalized difference (laser_on - laser_off) / laser_off.

    Parameters from YAML:
        detector_key: base detector key (e.g., "epix_ROI_1")
        laser_on_suffix: suffix for laser-on data (default "_simultaneous_laser_time_binned")
        laser_off_suffix: suffix for laser-off data (default "_xray_not_laser_time_binned")
    """
    detector_key = kwargs.get("detector_key", "epix_ROI_1")
    laser_on_suffix = kwargs.get("laser_on_suffix", "_simultaneous_laser_time_binned")
    laser_off_suffix = kwargs.get("laser_off_suffix", "_xray_not_laser_time_binned")

    laser_on_key = f"{detector_key}{laser_on_suffix}"
    laser_off_key = f"{detector_key}{laser_off_suffix}"
    laser_on_count_key = f"{detector_key}_simultaneous_laser_bincount"
    laser_off_count_key = f"{detector_key}_xray_not_laser_bincount"

    sum_on = None
    sum_off = None
    count_on = None
    count_off = None

    for r in runs:
        on_data = getattr(r, laser_on_key, None)
        off_data = getattr(r, laser_off_key, None)
        on_count = getattr(r, laser_on_count_key, None)
        off_count = getattr(r, laser_off_count_key, None)

        if on_data is not None:
            sum_on = on_data if sum_on is None else sum_on + on_data
        if off_data is not None:
            sum_off = off_data if sum_off is None else sum_off + off_data
        if on_count is not None:
            count_on = on_count if count_on is None else count_on + on_count
        if off_count is not None:
            count_off = off_count if count_off is None else count_off + off_count

    results = {
        "laser_on_summed": sum_on,
        "laser_off_summed": sum_off,
        "laser_on_count": count_on,
        "laser_off_count": count_off,
    }

    if (
        sum_on is not None
        and sum_off is not None
        and count_on is not None
        and count_off is not None
    ):
        safe_count_on = np.where(count_on > 0, count_on, 1)
        safe_count_off = np.where(count_off > 0, count_off, 1)
        avg_on = (
            sum_on / safe_count_on[:, np.newaxis]
            if sum_on.ndim == 2
            else sum_on / safe_count_on
        )
        avg_off = (
            sum_off / safe_count_off[:, np.newaxis]
            if sum_off.ndim == 2
            else sum_off / safe_count_off
        )

        safe_off = np.where(np.abs(avg_off) > 1e-10, avg_off, 1e-10)
        difference = (avg_on - avg_off) / safe_off

        results["laser_on_average"] = avg_on
        results["laser_off_average"] = avg_off
        results["difference"] = difference

    return results

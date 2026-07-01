"""
XAS (X-ray Absorption Spectroscopy) specific pipeline steps.
"""

import numpy as np
from XSpect.analysis.registry import register_step


@register_step("make_ccm_axis")
def make_ccm_axis(run, **kwargs):
    """Generate CCM (Channel Cut Monochromator) energy bins.

    Parameters from YAML:
        energies: "auto", explicit list, or [min, max, num_points]
        ccm_key: attribute with CCM values (for auto mode, default "ccm")
        resolution: rounding resolution in same units as ccm data (for auto mode, default 0.001)
    """
    energies_spec = kwargs.get("energies")
    ccm_key = kwargs.get("ccm_key", "ccm")
    resolution = kwargs.get("resolution", 0.001)
    if energies_spec is None:
        return

    # If ccm_bins was pre-computed on the parent run and injected into this
    # batch, skip re-derivation so all batches share the same global axis.
    if getattr(run, "ccm_bins", None) is not None:
        return

    if energies_spec == "auto":
        ccm_data = getattr(run, ccm_key, None)
        if ccm_data is None:
            run.update_status(f"make_ccm_axis: missing {ccm_key} for auto binning")
            return
        rounded = np.round(ccm_data / resolution) * resolution
        energies = np.unique(rounded)
    elif isinstance(energies_spec, list) and len(energies_spec) == 3:
        energies = np.linspace(
            energies_spec[0], energies_spec[1], int(energies_spec[2])
        )
    else:
        energies = np.array(energies_spec)

    if len(energies) < 2:
        addon = resolution / 2
    else:
        addon = (energies[-1] - energies[-2]) / 2
    bins2 = np.append(energies, energies[-1] + addon)
    bins_center = np.empty_like(bins2)
    for ii in range(len(energies)):
        if ii == 0:
            bins_center[ii] = bins2[ii] - (bins2[ii + 1] - bins2[ii]) / 2
        else:
            bins_center[ii] = bins2[ii] - (bins2[ii] - bins2[ii - 1]) / 2
    bins_center[-1] = bins2[-1]

    run.ccm_bins = bins_center
    run.ccm_energies = energies
    run.update_status(
        f"CCM axis: {len(energies)} energy points from {energies[0]:.1f} to {energies[-1]:.1f}"
    )


@register_step("ccm_binning")
def ccm_binning(run, **kwargs):
    """Digitize CCM data into energy bin indices.

    Parameters from YAML:
        ccm_key: attribute with CCM energy values (default "ccm")
        ccm_bins_key: attribute with bin edges (default "ccm_bins")
    """
    ccm_key = kwargs.get("ccm_key", "ccm")
    ccm_bins_key = kwargs.get("ccm_bins_key", "ccm_bins")

    ccm_data = getattr(run, ccm_key, None)
    ccm_bins = getattr(run, ccm_bins_key, None)

    if ccm_data is None or ccm_bins is None:
        run.update_status(f"ccm_binning: missing {ccm_key} or {ccm_bins_key}")
        return

    indices = np.digitize(ccm_data, ccm_bins)
    run.ccm_bin_indices = indices
    run.update_status(f"CCM binning complete: {len(ccm_bins)} bins")


@register_step("reduce_detector_ccm")
def reduce_detector_ccm(run, **kwargs):
    """Bin detector data along energy (CCM) dimension.

    Parameters from YAML:
        on: detector key (1D or 2D)
        ccm_bin_key: bin indices attribute (default "ccm_bin_indices")
        average: bool (default False)
    """
    detector_key = kwargs.get("on")
    ccm_bin_key = kwargs.get("ccm_bin_key", "ccm_bin_indices")
    average = kwargs.get("average", False)
    if detector_key is None:
        return

    detector = getattr(run, detector_key, None)
    ccm_indices = getattr(run, ccm_bin_key, None)
    ccm_bins = getattr(run, "ccm_bins", None)
    ccm_energies = getattr(run, "ccm_energies", None)

    if detector is None or ccm_indices is None or ccm_bins is None:
        run.update_status(f"reduce_detector_ccm: missing data for {detector_key}")
        return

    # ccm_bins has n_energies+1 elements (shifted edge array from make_ccm_axis).
    # Use n_energies so the output axis aligns 1-to-1 with ccm_energies.
    n_bins = len(ccm_energies) if ccm_energies is not None else len(ccm_bins)

    if detector.ndim == 1:
        binned = np.zeros(n_bins)
        bincount = np.zeros(n_bins)
        for i in range(len(detector)):
            idx = ccm_indices[i] - 1
            if 0 <= idx < n_bins and not np.isnan(detector[i]):
                binned[idx] += detector[i]
                bincount[idx] += 1
    elif detector.ndim == 2:
        n_pixels = detector.shape[1]
        binned = np.zeros((n_bins, n_pixels))
        bincount = np.zeros(n_bins)
        for i in range(detector.shape[0]):
            idx = ccm_indices[i] - 1
            if 0 <= idx < n_bins:
                row = detector[i]
                if not np.all(np.isnan(row)):
                    binned[idx] += np.where(np.isnan(row), 0.0, row)
                    bincount[idx] += 1
    elif detector.ndim == 3:
        n_rows = detector.shape[1]
        n_cols = detector.shape[2]
        binned = np.zeros((n_bins, n_rows, n_cols))
        bincount = np.zeros(n_bins)
        for i in range(detector.shape[0]):
            idx = ccm_indices[i] - 1
            if 0 <= idx < n_bins:
                frame = detector[i]
                if not np.all(np.isnan(frame)):
                    binned[idx] += np.where(np.isnan(frame), 0.0, frame)
                    bincount[idx] += 1
    else:
        run.update_status(f"reduce_detector_ccm: unsupported ndim={detector.ndim}")
        return

    if average:
        safe_count = np.where(bincount > 0, bincount, 1)
        if binned.ndim == 2:
            binned = binned / safe_count[:, np.newaxis]
        else:
            binned = binned / safe_count

    setattr(run, f"{detector_key}_energy_binned", binned)
    setattr(run, f"{detector_key}_energy_bincount", bincount)
    run.update_status(
        f"CCM reduction: {detector_key} -> {detector_key}_energy_binned ({n_bins} bins)"
    )


@register_step("reduce_detector_ccm_temporal")
def reduce_detector_ccm_temporal(run, **kwargs):
    """2D binning: both time AND energy (CCM) dimensions.

    Produces a 3D array: (time_bins x energy_bins x pixels) or
    2D: (time_bins x energy_bins) for scalar detectors.

    Parameters from YAML:
        on: detector key
        timing_bin_key: time bin indices (default "timing_bin_indices")
        ccm_bin_key: CCM bin indices (default "ccm_bin_indices")
        average: bool (default False)
    """
    detector_key = kwargs.get("on")
    timing_bin_key = kwargs.get("timing_bin_key", "timing_bin_indices")
    ccm_bin_key = kwargs.get("ccm_bin_key", "ccm_bin_indices")
    average = kwargs.get("average", False)
    if detector_key is None:
        return

    detector = getattr(run, detector_key, None)
    timing_indices = getattr(run, timing_bin_key, None)
    ccm_indices = getattr(run, ccm_bin_key, None)
    time_bins = getattr(run, "time_bins", None)
    ccm_bins = getattr(run, "ccm_bins", None)

    if any(
        x is None for x in [detector, timing_indices, ccm_indices, time_bins, ccm_bins]
    ):
        run.update_status(
            f"reduce_detector_ccm_temporal: missing data for {detector_key}"
        )
        return

    n_time = len(time_bins)
    # Use ccm_energies length (n bins) not ccm_bins length (n+1 edges)
    ccm_energies = getattr(run, "ccm_energies", None)
    n_energy = len(ccm_energies) if ccm_energies is not None else len(ccm_bins) - 1

    if detector.ndim == 1:
        binned = np.zeros((n_time, n_energy))
        bincount = np.zeros((n_time, n_energy))
        for i in range(len(detector)):
            t_idx = timing_indices[i] - 1
            e_idx = ccm_indices[i] - 1
            if (
                0 <= t_idx < n_time
                and 0 <= e_idx < n_energy
                and not np.isnan(detector[i])
            ):
                binned[t_idx, e_idx] += detector[i]
                bincount[t_idx, e_idx] += 1
    elif detector.ndim == 2:
        n_pixels = detector.shape[1]
        binned = np.zeros((n_time, n_energy, n_pixels))
        bincount = np.zeros((n_time, n_energy))
        for i in range(detector.shape[0]):
            t_idx = timing_indices[i] - 1
            e_idx = ccm_indices[i] - 1
            if 0 <= t_idx < n_time and 0 <= e_idx < n_energy:
                row = detector[i]
                if not np.all(np.isnan(row)):
                    binned[t_idx, e_idx] += np.where(np.isnan(row), 0.0, row)
                    bincount[t_idx, e_idx] += 1
    else:
        run.update_status(
            f"reduce_detector_ccm_temporal: unsupported ndim={detector.ndim}"
        )
        return

    if average:
        safe_count = np.where(bincount > 0, bincount, 1)
        if binned.ndim == 3:
            binned = binned / safe_count[:, :, np.newaxis]
        else:
            binned = binned / safe_count

    setattr(run, f"{detector_key}_time_energy_binned", binned)
    setattr(run, f"{detector_key}_time_energy_bincount", bincount)
    run.update_status(
        f"CCM+temporal reduction: {detector_key} -> {detector_key}_time_energy_binned ({n_time}x{n_energy})"
    )

"""
XAS (X-ray Absorption Spectroscopy) specific pipeline steps.
"""

import numpy as np
from XSpect.analysis.registry import register_step


@register_step("make_ccm_axis")
def make_ccm_axis(run, **kwargs):
    """Generate CCM (Channel Cut Monochromator) energy bins.

    Parameters from YAML:
        energies: list of energy values OR [min, max, num_points]
    """
    energies_spec = kwargs.get("energies")
    if energies_spec is None:
        return

    if isinstance(energies_spec, list) and len(energies_spec) == 3:
        energies = np.linspace(energies_spec[0], energies_spec[1], int(energies_spec[2]))
    else:
        energies = np.array(energies_spec)

    addon = (energies[-1] - energies[-2]) / 2
    bins2 = np.append(energies, energies[-1] + addon)
    bins_center = np.empty_like(bins2)
    for ii in range(len(energies)):
        if ii == 0:
            bins_center[ii] = bins2[ii] - (bins2[ii+1] - bins2[ii]) / 2
        else:
            bins_center[ii] = bins2[ii] - (bins2[ii] - bins2[ii-1]) / 2
    bins_center[-1] = bins2[-1]

    run.ccm_bins = bins_center
    run.ccm_energies = energies
    run.update_status(f"CCM axis: {len(energies)} energy points from {energies[0]:.1f} to {energies[-1]:.1f} eV")


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

    if detector is None or ccm_indices is None or ccm_bins is None:
        run.update_status(f"reduce_detector_ccm: missing data for {detector_key}")
        return

    n_bins = len(ccm_bins)

    if detector.ndim == 1:
        binned = np.zeros(n_bins)
        bincount = np.zeros(n_bins)
        for i in range(len(detector)):
            idx = ccm_indices[i] - 1
            if 0 <= idx < n_bins:
                binned[idx] += detector[i]
                bincount[idx] += 1
    elif detector.ndim == 2:
        n_pixels = detector.shape[1]
        binned = np.zeros((n_bins, n_pixels))
        bincount = np.zeros(n_bins)
        for i in range(detector.shape[0]):
            idx = ccm_indices[i] - 1
            if 0 <= idx < n_bins:
                binned[idx] += detector[i]
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
    run.update_status(f"CCM reduction: {detector_key} -> {detector_key}_energy_binned ({n_bins} bins)")


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

    if any(x is None for x in [detector, timing_indices, ccm_indices, time_bins, ccm_bins]):
        run.update_status(f"reduce_detector_ccm_temporal: missing data for {detector_key}")
        return

    n_time = len(time_bins)
    n_energy = len(ccm_bins)

    if detector.ndim == 1:
        binned = np.zeros((n_time, n_energy))
        bincount = np.zeros((n_time, n_energy))
        for i in range(len(detector)):
            t_idx = timing_indices[i] - 1
            e_idx = ccm_indices[i] - 1
            if 0 <= t_idx < n_time and 0 <= e_idx < n_energy:
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
                binned[t_idx, e_idx] += detector[i]
                bincount[t_idx, e_idx] += 1
    else:
        run.update_status(f"reduce_detector_ccm_temporal: unsupported ndim={detector.ndim}")
        return

    if average:
        safe_count = np.where(bincount > 0, bincount, 1)
        if binned.ndim == 3:
            binned = binned / safe_count[:, :, np.newaxis]
        else:
            binned = binned / safe_count

    setattr(run, f"{detector_key}_time_energy_binned", binned)
    setattr(run, f"{detector_key}_time_energy_bincount", bincount)
    run.update_status(f"CCM+temporal reduction: {detector_key} -> {detector_key}_time_energy_binned ({n_time}x{n_energy})")

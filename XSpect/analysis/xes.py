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
            norm_slice = data[:, pixel_range[0]:pixel_range[1]]
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

    Converts pixel positions to energy values using the crystal geometry
    parameters (Bragg angle, crystal radius, d-spacing, pixel size).

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
    pixel_positions = np.arange(n_pixels) * mm_per_pixel
    pixel_center = pixel_positions[n_pixels // 2]
    offsets = pixel_positions - pixel_center

    theta_center = np.arctan(A / R) if R != 0 else np.pi / 4
    theta_array = theta_center + np.arctan(offsets / np.sqrt(A**2 + R**2))

    energy_axis = hc / (2 * d * np.sin(theta_array))

    setattr(run, f"{name}_energy", energy_axis)
    run.update_status(f"Energy axis: {name}_energy ({n_pixels} pixels, center {energy_axis[n_pixels//2]:.1f} eV)")


@register_step("patch_pixels")
def patch_pixels(run, **kwargs):
    """Repair bad pixels by interpolation from neighbors.

    Parameters from YAML:
        on: detector key
        pixels: list of pixel indices to patch
        mode: "interpolate" or "zero" (default "interpolate")
        axis: which axis the pixel indices refer to (default: last axis for 2D, 0 for 1D)
    """
    detector_key = kwargs.get("on")
    pixels = kwargs.get("pixels", [])
    mode = kwargs.get("mode", "interpolate")
    axis = kwargs.get("axis", None)
    if detector_key is None or not pixels:
        return

    data = getattr(run, detector_key, None)
    if data is None:
        return

    if axis is None:
        axis = 0 if data.ndim == 1 else data.ndim - 1

    n_pixels = data.shape[axis]

    for pixel in pixels:
        if pixel < 0 or pixel >= n_pixels:
            continue
        slc = [slice(None)] * data.ndim
        slc[axis] = pixel
        if mode == "interpolate" and 0 < pixel < n_pixels - 1:
            slc_prev = [slice(None)] * data.ndim
            slc_prev[axis] = pixel - 1
            slc_next = [slice(None)] * data.ndim
            slc_next[axis] = pixel + 1
            data[tuple(slc)] = (data[tuple(slc_prev)] + data[tuple(slc_next)]) / 2
        else:
            data[tuple(slc)] = 0

    setattr(run, detector_key, data)
    run.update_status(f"Patched {len(pixels)} pixels on {detector_key} (axis={axis})")

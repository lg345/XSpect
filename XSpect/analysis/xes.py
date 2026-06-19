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
    run.update_status(f"Energy axis: {name}_energy ({n_pixels} pixels, center {energy_axis[n_pixels//2]:.1f} eV)")


@register_step("patch_pixels")
def patch_pixels(run, **kwargs):
    """Repair bad pixels using polynomial fitting from neighbors.

    Replicates XSpect_Analysis.patch_pixel with mode='polynomial':
    fits a weighted polynomial to surrounding pixels (excluding the bad
    pixel region) and evaluates at the bad pixel location.

    Parameters from YAML:
        on: detector key
        pixels: list of pixel indices to patch
        mode: "polynomial", "interpolate", or "zero" (default "polynomial")
        axis: which axis the pixel indices refer to (default: 0 for 2D, last for 3D)
        patch_range: pixels on each side of bad pixel to exclude (default 4)
        poly_range: additional pixels beyond patch_range for fitting (default 6)
        deg: polynomial degree (default 1)
    """
    detector_key = kwargs.get("on")
    pixels = kwargs.get("pixels", [])
    mode = kwargs.get("mode", "polynomial")
    axis = kwargs.get("axis", None)
    patch_range = kwargs.get("patch_range", 4)
    poly_range = kwargs.get("poly_range", 6)
    deg = kwargs.get("deg", 1)
    if detector_key is None or not pixels:
        return

    data = getattr(run, detector_key, None)
    if data is None:
        return

    if axis is None:
        axis = 0 if data.ndim <= 2 else data.ndim - 1

    n_pixels = data.shape[axis]

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
            weights[patch_range:-patch_range] = 0.001

            if data.ndim == 1:
                coeffs = np.polyfit(patch_x, region, deg, w=weights)
                new_val = np.polyval(coeffs, pixel)
            else:
                other_shape = region.shape[1:]
                flat = region.reshape(region.shape[0], -1)
                new_vals = np.empty(flat.shape[1])
                for idx in range(flat.shape[1]):
                    coeffs = np.polyfit(patch_x, flat[:, idx], deg, w=weights)
                    new_vals[idx] = np.polyval(coeffs, pixel)
                new_val = new_vals.reshape(other_shape)

            slc = [slice(None)] * data.ndim
            slc[axis] = pixel
            data[tuple(slc)] = new_val

    setattr(run, detector_key, data)
    run.update_status(f"Patched {len(pixels)} pixels on {detector_key} (axis={axis}, mode={mode})")

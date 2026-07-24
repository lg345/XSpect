"""Droplet-to-photon reconstruction pipeline step.

Reads per-shot sparse photon coordinates from the fixed-length or variable-length
droplet2photon arrays stored in smalldata HDF5 files and scatters them back onto
dense 2D images, producing a (N_shots, rows, cols) stack for downstream steps.

Supported HDF5 layouts
----------------------
Fixed-length  (e.g. epix100_0):
    <det>/droplet_droplet2phot_sparse_{row,col,data,tile}
    shape (Nshots, nData) — zero-padded; valid entries have data != 0

Variable-length (e.g. epix100_1):
    <det>/var_droplet_droplet2phot_sparse/{row,col,data}
    flat arrays + <det>/var_droplet_droplet2phot_sparse_len (per-shot counts)

Batch awareness
---------------
When the pipeline batches a run, the batch manager injects ``abs_start_index``
and ``abs_end_index`` onto each batch run so the step knows which HDF5 rows to
read.  In the non-batched path these attributes are absent and the step falls
back to ``run.start_index`` / ``run.end_index``.

ROI-direct scattering
---------------------
When a ``roi`` is given, photons are filtered to the ROI window and scattered
straight into the small cropped image (``_scatter_roi``) — the full 704x768
panel is never allocated.  This is bit-identical to scattering onto the full
panel and cropping, but uses ~30x less memory and runs ~2x faster for the
typical (60, 300) XES window.
"""

import time

import h5py
import numpy as np

from XSpect.analysis.registry import register_step

# Full ePix100 panel dimensions (rows, cols)
PANEL_SHAPE = (704, 768)


# ---------------------------------------------------------------------------
# Internal scatter helper
# ---------------------------------------------------------------------------


def _scatter(rows, cols, data, shape):
    """Accumulate photon counts at (row, col) into a dense 2D image.

    Sub-pixel float coordinates are floored to the containing pixel.
    Repeated coordinates accumulate (multi-photon pixels add up correctly).
    """
    img = np.zeros(shape, dtype=np.float64)
    if rows.size == 0:
        return img
    ri = np.floor(rows).astype(np.int64)
    ci = np.floor(cols).astype(np.int64)
    inb = (ri >= 0) & (ri < shape[0]) & (ci >= 0) & (ci < shape[1])
    np.add.at(img, (ri[inb], ci[inb]), data[inb])
    return img


def _scatter_roi(rows, cols, data, roi):
    """Scatter photons directly into a cropped ROI image (no full panel).

    Filters photons to the ROI window *before* allocating the output, so memory
    and the np.add.at scatter scale with the ROI size, not the full panel.  This
    is the memory-efficient equivalent of ``_scatter(...)[r0:r1, c0:c1]`` and
    produces bit-identical results.

    Parameters
    ----------
    rows, cols, data : 1D arrays of photon row/col coordinates and counts
    roi : tuple (row0, row1, col0, col1) — half-open window on the full panel

    Returns
    -------
    np.ndarray shape (row1 - row0, col1 - col0)
    """
    r0, r1, c0, c1 = roi[0], roi[1], roi[2], roi[3]
    out_shape = (r1 - r0, c1 - c0)
    img = np.zeros(out_shape, dtype=np.float64)
    if rows.size == 0:
        return img
    ri = np.floor(rows).astype(np.int64)
    ci = np.floor(cols).astype(np.int64)
    # Keep only photons whose floored pixel falls inside the ROI window.
    inb = (ri >= r0) & (ri < r1) & (ci >= c0) & (ci < c1)
    # Offset into ROI-local coordinates before scattering.
    np.add.at(img, (ri[inb] - r0, ci[inb] - c0), data[inb])
    return img


# ---------------------------------------------------------------------------
# Fixed-length format reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_fixed(fh, det, abs_start, abs_end, roi, panel_shape):
    """Reconstruct shots from the fixed-length (zero-padded) sparse format.

    Parameters
    ----------
    fh         : h5py.File, open for reading
    det        : str  — detector group, e.g. "epix100_0"
    abs_start  : int  — first HDF5 shot index (inclusive)
    abs_end    : int  — last HDF5 shot index (exclusive)
    roi        : tuple (row0, row1, col0, col1) or None
    panel_shape: (rows, cols) of the full panel

    Returns
    -------
    np.ndarray shape (n_shots, out_rows, out_cols)
    """
    g = f"{det}/droplet_droplet2phot_sparse"
    row_ds = fh[f"{g}_row"]
    col_ds = fh[f"{g}_col"]
    dat_ds = fh[f"{g}_data"]

    n_shots = abs_end - abs_start
    if roi is not None:
        out_shape = (roi[1] - roi[0], roi[3] - roi[2])
    else:
        out_shape = panel_shape

    out = np.zeros((n_shots,) + out_shape, dtype=np.float64)

    for k in range(n_shots):
        idx = abs_start + k
        row = row_ds[idx].astype(np.float64)
        col = col_ds[idx].astype(np.float64)
        dat = dat_ds[idx].astype(np.float64)

        valid = dat != 0  # zero-padding sentinel
        row, col, dat = row[valid], col[valid], dat[valid]

        if roi is not None:
            # Scatter directly into the ROI — never allocate the full panel.
            out[k] = _scatter_roi(row, col, dat, roi)
        else:
            out[k] = _scatter(row, col, dat, panel_shape)

    return out


# ---------------------------------------------------------------------------
# Variable-length format reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_var(fh, det, abs_start, abs_end, roi, panel_shape):
    """Reconstruct shots from the variable-length sparse format.

    Parameters
    ----------
    fh         : h5py.File, open for reading
    det        : str  — detector group, e.g. "epix100_1"
    abs_start  : int  — first HDF5 shot index (inclusive)
    abs_end    : int  — last HDF5 shot index (exclusive)
    roi        : tuple (row0, row1, col0, col1) or None
    panel_shape: (rows, cols) of the full panel

    Returns
    -------
    np.ndarray shape (n_shots, out_rows, out_cols)
    """
    # Build cumulative offset array (cached on first call per batch)
    lens = fh[f"{det}/var_droplet_droplet2phot_sparse_len"][:]
    offsets = np.concatenate([[0], np.cumsum(lens)])

    g = f"{det}/var_droplet_droplet2phot_sparse"
    row_ds = fh[f"{g}/row"]
    col_ds = fh[f"{g}/col"]
    dat_ds = fh[f"{g}/data"]

    n_shots = abs_end - abs_start
    if roi is not None:
        out_shape = (roi[1] - roi[0], roi[3] - roi[2])
    else:
        out_shape = panel_shape

    out = np.zeros((n_shots,) + out_shape, dtype=np.float64)

    for k in range(n_shots):
        idx = abs_start + k
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        row = row_ds[s:e].astype(np.float64)
        col = col_ds[s:e].astype(np.float64)
        dat = dat_ds[s:e].astype(np.float64)

        if roi is not None:
            # Scatter directly into the ROI — never allocate the full panel.
            out[k] = _scatter_roi(row, col, dat, roi)
        else:
            out[k] = _scatter(row, col, dat, panel_shape)

    return out


# ---------------------------------------------------------------------------
# Registered pipeline step
# ---------------------------------------------------------------------------


@register_step("droplet_reconstruction")
def droplet_reconstruction(run, **kwargs):
    """Reconstruct per-shot 2D images from droplet2photon sparse data.

    Reads sparse photon-position arrays directly from the source smalldata HDF5
    file and scatters them onto dense images.  The resulting (N_shots, rows, cols)
    array is stored on the run object under ``new_key`` and is compatible with
    all downstream steps (``filter_detector_adu``, ``patch_pixels``,
    ``rotate_detector``, ``reduce_detector_spatial``, etc.).

    The step reads from ``run.run_file``.  In batch mode the batch manager
    injects ``abs_start_index`` and ``abs_end_index`` onto the batch run so
    the correct HDF5 rows are read; in the non-batched path these attributes
    are absent and the step derives the range from ``run.start_index`` /
    ``run.end_index``.

    Parameters from YAML
    --------------------
    det         : str   — detector group in the HDF5, e.g. "epix100_0"
    new_key     : str   — attribute name to store the reconstructed stack
    roi         : list  — [row0, row1, col0, col1] crop region (omit for full panel)
    panel_shape : list  — [rows, cols] of the full panel (default [704, 768])

    Example YAML step
    -----------------
    - step: droplet_reconstruction
      det: epix100_0
      new_key: epix_spec
      roi: [270, 330, 400, 700]   # -> (60, 300) output matching ROI_area
    """
    det = kwargs.get("det")
    new_key = kwargs.get("new_key")
    roi_kwarg = kwargs.get("roi", None)
    panel_shape_kwarg = kwargs.get("panel_shape", None)

    if det is None or new_key is None:
        run.update_status("droplet_reconstruction: 'det' and 'new_key' are required")
        return

    run_file = getattr(run, "run_file", None)
    if run_file is None:
        run.update_status("droplet_reconstruction: run_file not set, skipping")
        return

    # ------------------------------------------------------------------
    # Resolve absolute HDF5 shot indices
    # In batch mode the batch manager sets abs_start_index / abs_end_index.
    # In the non-batched path those attrs are absent; fall back to the
    # run's own start_index / end_index.
    # ------------------------------------------------------------------
    abs_start = getattr(run, "abs_start_index", None)
    if abs_start is None:
        abs_start = getattr(run, "start_index", 0)

    abs_end = getattr(run, "abs_end_index", None)
    if abs_end is None:
        end_idx = getattr(run, "end_index", -1)
        if end_idx == -1:
            # Need to look up the total shot count from the HDF5
            try:
                with h5py.File(run_file, "r") as _fh:
                    fixed_key = f"{det}/droplet_droplet2phot_sparse_data"
                    var_key = f"{det}/var_droplet_droplet2phot_sparse_len"
                    if fixed_key in _fh:
                        abs_end = int(_fh[fixed_key].shape[0])
                    elif var_key in _fh:
                        abs_end = int(_fh[var_key].shape[0])
                    else:
                        # Fall back to total_shots if known
                        abs_end = abs_start + int(getattr(run, "total_shots", 0))
            except Exception as exc:
                run.update_status(
                    f"droplet_reconstruction: could not determine shot count: {exc}"
                )
                return
        else:
            abs_end = int(end_idx)

    if abs_end <= abs_start:
        run.update_status(
            f"droplet_reconstruction: empty range [{abs_start}, {abs_end}), skipping"
        )
        return

    panel_shape = (
        tuple(panel_shape_kwarg) if panel_shape_kwarg is not None else PANEL_SHAPE
    )
    roi = tuple(roi_kwarg) if roi_kwarg is not None else None

    n_shots = abs_end - abs_start

    t0 = time.time()
    try:
        with h5py.File(run_file, "r") as fh:
            fixed_key = f"{det}/droplet_droplet2phot_sparse_data"
            var_key = f"{det}/var_droplet_droplet2phot_sparse/data"

            if fixed_key in fh:
                images = _reconstruct_fixed(
                    fh, det, abs_start, abs_end, roi, panel_shape
                )
            elif var_key in fh:
                images = _reconstruct_var(fh, det, abs_start, abs_end, roi, panel_shape)
            else:
                available = list(fh[det].keys()) if det in fh else []
                run.update_status(
                    f"droplet_reconstruction: no sparse data found for '{det}'. "
                    f"Available keys under '{det}': {available}"
                )
                return
    except Exception as exc:
        run.update_status(f"droplet_reconstruction: HDF5 read failed: {exc}")
        return

    elapsed = time.time() - t0
    setattr(run, new_key, images)
    run.update_status(
        f"droplet_reconstruction: '{det}' -> '{new_key}' "
        f"shape={images.shape} roi={roi} "
        f"shots=[{abs_start},{abs_end}) "
        f"elapsed={elapsed:.1f}s"
    )

"""
Per-shot detector image reconstruction from droplet2photon sparse results.

The smd_producer droplet2Photons pipeline stores, for every shot, a sparse list
of reconstructed single-photon positions (row, col) on the FULL epix100 panel
(704 x 768), each with a photon count in `data` (normally 1.0; repeated
coordinates / data>1 mean multiple photons at the same pixel).

Two storage layouts appear in the same file:

  * Fixed-length  (epix100_0):  <det>/droplet_droplet2phot_sparse_{row,col,data,tile}
                                shape (Nshots, nData) zero-padded; valid = data != 0
  * Variable-length (epix100_1): <det>/var_droplet_droplet2phot_sparse/{row,col,data,tile}
                                 flat arrays + <det>/..._sparse_len giving per-shot counts

This module scatters those photons back onto a 2D image. Coordinates are in
full-panel space, so the returned image is the full panel by default; pass
roi=(r0, r1, c0, c1) to crop (e.g. the LUTE getROIs window).

Usage
-----
    from droplet_reconstruct import DropletReconstructor, ROIS, PANEL_SHAPE

    rec = DropletReconstructor(SMD_PATH)
    img = rec.reconstruct_shot('epix100_0', shot_index=30311)        # full panel
    img = rec.reconstruct_shot('epix100_1', 1361, roi=ROIS['epix100_1'])  # cropped
    rec.close()
"""

import h5py
import numpy as np

PANEL_SHAPE = (704, 768)  # full epix100 panel (rows, cols)

# ROI windows (row0, row1, col0, col1) from the LUTE getROIs config for
# mfx101609126 — used only for optional cropping / verification against ROI_area.
ROIS = {
    "epix100_0": (270, 330, 400, 700),  # -> (60, 300)
    "epix100_1": (350, 450, 80, 700),  # -> (100, 620)
}


def _scatter(rows, cols, data, shape):
    """Accumulate photon counts at (row, col) into a dense 2D image.

    Sub-pixel float coordinates are floored to the containing pixel.  Repeated
    coordinates accumulate, so multi-photon pixels add up correctly.
    """
    img = np.zeros(shape, dtype=np.float64)
    if rows.size == 0:
        return img
    ri = np.floor(rows).astype(np.int64)
    ci = np.floor(cols).astype(np.int64)
    inb = (ri >= 0) & (ri < shape[0]) & (ci >= 0) & (ci < shape[1])
    np.add.at(img, (ri[inb], ci[inb]), data[inb])
    return img


class DropletReconstructor:
    """Reconstruct per-shot images from droplet2photon sparse data."""

    def __init__(self, smd_path):
        self.path = smd_path
        self.f = h5py.File(smd_path, "r")
        self._var_offsets = {}  # cache cumulative offsets for variable-length dets

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- format detection ----------------------------------------------------
    def _is_fixed(self, det):
        return f"{det}/droplet_droplet2phot_sparse_data" in self.f

    def _is_var(self, det):
        return f"{det}/var_droplet_droplet2phot_sparse/data" in self.f

    def _offsets(self, det):
        if det not in self._var_offsets:
            lens = self.f[f"{det}/var_droplet_droplet2phot_sparse_len"][:]
            self._var_offsets[det] = np.concatenate([[0], np.cumsum(lens)])
        return self._var_offsets[det]

    # -- per-shot photon coordinates ----------------------------------------
    def photons(self, det, shot_index):
        """Return (rows, cols, data) of valid photons for one shot (full panel)."""
        if self._is_fixed(det):
            g = f"{det}/droplet_droplet2phot_sparse"
            row = self.f[f"{g}_row"][shot_index]
            col = self.f[f"{g}_col"][shot_index]
            dat = self.f[f"{g}_data"][shot_index]
            valid = dat != 0  # zero-padding sentinel
            return (
                row[valid].astype(np.float64),
                col[valid].astype(np.float64),
                dat[valid].astype(np.float64),
            )
        if self._is_var(det):
            off = self._offsets(det)
            s, e = off[shot_index], off[shot_index + 1]
            g = f"{det}/var_droplet_droplet2phot_sparse"
            row = self.f[f"{g}/row"][s:e].astype(np.float64)
            col = self.f[f"{g}/col"][s:e].astype(np.float64)
            dat = self.f[f"{g}/data"][s:e].astype(np.float64)
            return row, col, dat
        # Neither layout matched — list what *is* present to aid debugging
        # (a common cause is a stale module import in a Jupyter kernel: the
        #  on-disk droplet_reconstruct.py supports both layouts, so reload it).
        available = list(self.f[det].keys()) if det in self.f else []
        raise KeyError(
            f"No droplet2phot sparse data found for {det!r}. "
            f"Datasets under {det!r}: {available}. "
            f"If you edited droplet_reconstruct.py, reload it in the kernel "
            f"(e.g. importlib.reload) — Jupyter caches the old import."
        )

    # -- reconstruction ------------------------------------------------------
    def reconstruct_shot(self, det, shot_index, roi=None):
        """Reconstruct the 2D photon-count image for a single shot.

        Parameters
        ----------
        det : str            detector group name, e.g. 'epix100_0'
        shot_index : int     absolute shot index in the run
        roi : tuple or None  (row0, row1, col0, col1) to crop; None -> full panel

        Returns
        -------
        np.ndarray (rows, cols) of photon counts
        """
        rows, cols, dat = self.photons(det, shot_index)
        if roi is None:
            return _scatter(rows, cols, dat, PANEL_SHAPE)
        r0, r1, c0, c1 = roi
        img = _scatter(rows, cols, dat, PANEL_SHAPE)
        return img[r0:r1, c0:c1]

    def reconstruct_many(self, det, shot_indices, roi=None):
        """Reconstruct a stack of shots -> (Nshots, rows, cols)."""
        shape = PANEL_SHAPE if roi is None else (roi[1] - roi[0], roi[3] - roi[2])
        out = np.zeros((len(shot_indices),) + shape, dtype=np.float64)
        for k, idx in enumerate(shot_indices):
            out[k] = self.reconstruct_shot(det, idx, roi=roi)
        return out

    def sum_image(self, det, shot_indices, roi=None):
        """Accumulate all photons across the given shots into one 2D image."""
        shape = PANEL_SHAPE if roi is None else (roi[1] - roi[0], roi[3] - roi[2])
        acc = np.zeros(shape, dtype=np.float64)
        for idx in shot_indices:
            acc += self.reconstruct_shot(det, idx, roi=roi)
        return acc

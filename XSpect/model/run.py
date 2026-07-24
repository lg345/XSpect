import h5py
import numpy as np
import time
from datetime import datetime


class spectroscopy_run:
    """
    A class to represent a run within a spectroscopy experiment. Not an LCLS run.
    """

    def __init__(
        self, spec_experiment, run, verbose=False, end_index=-1, start_index=0
    ):
        """
        Initializes a spectroscopy run instance.

        Parameters
        ----------
        spec_experiment : spectroscopy_experiment
            The parent spectroscopy experiment.
        run : int
            The run number.
        verbose : bool, optional
            Flag for verbose output. Defaults to False.
        end_index : int, optional
            Index to stop processing data. Defaults to -1.
        start_index : int, optional
            Index to start processing data. Defaults to 0.
        """
        self.spec_experiment = spec_experiment
        self.run_number = run
        self.run_file = "%s/%s_Run%04d.h5" % (
            self.spec_experiment.experiment_directory,
            self.spec_experiment.experiment_id,
            self.run_number,
        )
        self.status = [
            "New analysis of run %d located in: %s" % (self.run_number, self.run_file)
        ]
        self.status_datetime = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        self.verbose = verbose
        self.end_index = end_index
        self.start_index = start_index
        self.results = {}

    def update_status(self, update):
        """
        Updates the status log for the run.

        Parameters
        ----------
        update : str
            The status update message.
        """
        self.status.append(update)
        self.status_datetime.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.verbose:
            print(update)

    def get_run_shot_properties(self):
        """
        Retrieves shot properties from the run file.
        """
        with h5py.File(self.run_file, "r") as fh:
            if self.end_index == -1:
                self.total_shots = fh["lightStatus/xray"][self.start_index :].shape[0]
                xray_total = np.sum(fh["lightStatus/xray"][self.start_index :])
                laser_total = np.sum(fh["lightStatus/laser"][self.start_index :])
                self.xray = np.array(fh["lightStatus/xray"][self.start_index :])
                self.laser = np.array(fh["lightStatus/laser"][self.start_index :])
                self.simultaneous = np.logical_and(self.xray, self.laser)
            else:
                self.total_shots = fh["lightStatus/xray"][
                    self.start_index : self.end_index
                ].shape[0]
                xray_total = np.sum(
                    fh["lightStatus/xray"][self.start_index : self.end_index]
                )
                laser_total = np.sum(
                    fh["lightStatus/laser"][self.start_index : self.end_index]
                )
                self.xray = np.array(
                    fh["lightStatus/xray"][self.start_index : self.end_index]
                )
                self.laser = np.array(
                    fh["lightStatus/laser"][self.start_index : self.end_index]
                )
                self.simultaneous = np.logical_and(self.xray, self.laser)

        self.run_shots = {
            "Total": self.total_shots,
            "X-ray Total": xray_total,
            "Laser Total": laser_total,
        }
        self.update_status("Obtained shot properties")

    def get_scan_val(self):
        with h5py.File(self.run_file, "r") as fh:
            self.scan_var = fh["scan/scan_variable"]

    def set_arbitrary_filter(self, key="arbitrary_filter"):
        self.verbose = False
        with h5py.File(self.run_file, "r") as fh:
            self.arbitrary_filter = fh[key][self.start_index : self.end_index]

    def load_run_keys(self, keys, friendly_names):
        """
        Loads specified keys from the run file into memory.

        Parameters
        ----------
        keys : list
            List of keys to load from the hdf5 file
        friendly_names : list
            Corresponding list of friendly names for the keys.
        """
        start = time.time()
        with h5py.File(self.run_file, "r") as fh:
            for key, name in zip(keys, friendly_names):
                try:
                    if self.end_index == -1:
                        setattr(self, name, np.array(fh[key][self.start_index :]))
                    else:
                        setattr(
                            self,
                            name,
                            np.array(fh[key][self.start_index : self.end_index]),
                        )
                except KeyError as e:
                    self.update_status("Key does not exist: %s" % e.args[0])
                except MemoryError:
                    setattr(self, name, fh[key])
                    self.update_status(
                        "Out of memory error while loading key: %s. Not converted to np.array."
                        % key
                    )
        end = time.time()
        self.update_status(
            "HDF5 import of keys completed. Time: %.02f seconds" % (end - start)
        )

    def load_run_key_delayed(
        self,
        keys,
        friendly_names,
        transpose=False,
        rois=None,
        combine=True,
        row_range=None,
    ):
        """
        Loads specified keys from the run file without immediate conversion to numpy arrays.

        Parameters
        ----------
        keys : list
            List of keys to load.
        friendly_names : list
            Corresponding list of friendly names for the keys.
        transpose : bool, optional
            Flag to transpose the loaded data. Defaults to False.
        rois : list of lists, optional
            List of ROIs as pixel ranges along one dimension (default is None).
        combine : bool, optional
            Whether to combine ROIs into a single mask. Defaults to True.
        """
        start = time.time()
        fh = h5py.File(self.run_file, "r")

        for key, name in zip(keys, friendly_names):
            try:
                shot_slice = slice(
                    self.start_index, None if self.end_index == -1 else self.end_index
                )

                if row_range is not None:
                    r0, r1 = int(row_range[0]), int(row_range[1])
                    if transpose:
                        # row_range is in the transposed frame (axis 1 after transpose).
                        # In the raw HDF5 frame the transposed row axis is axis 2.
                        data = np.array(fh[key][shot_slice, :, r0:r1])
                    else:
                        data = np.array(fh[key][shot_slice, r0:r1, :])
                    # Record the crop origin so downstream steps can translate
                    # absolute (full-frame) ROI coordinates into cropped-array
                    # coordinates automatically. Stored per detector name.
                    if not hasattr(self, "_row_offset"):
                        self._row_offset = {}
                    self._row_offset[name] = r0
                else:
                    data = np.array(fh[key][shot_slice, :, :])

                if transpose:
                    data = np.transpose(data, axes=(0, 2, 1))
                    setattr(self, name, data)

                if rois is not None:
                    if combine:
                        mask = np.zeros(data.shape[2], dtype=bool)
                        for roi in rois:
                            start_col, end_col = roi
                            mask[start_col:end_col] = True
                        data = data[:, :, mask]
                        setattr(self, f"{name}_ROI_1", data)
                    else:
                        for idx, roi in enumerate(rois):
                            start_col, end_col = roi
                            roi_data = data[:, :, start_col:end_col]
                            setattr(self, f"{name}_ROI_{idx + 1}", roi_data)

                setattr(self, name, data)

            except KeyError as e:
                self.update_status(f"Key does not exist: {e.args[0]}")
            except MemoryError:
                if self.end_index == -1:
                    setattr(self, name, fh[key][self.start_index :, :, :])
                else:
                    setattr(
                        self, name, fh[key][self.start_index : self.end_index, :, :]
                    )
                self.update_status(
                    f"Out of memory error while loading key: {key}. Not converted to np.array."
                )

        end = time.time()
        self.update_status(
            f"HDF5 import of keys completed. Time: {end - start:.02f} seconds"
        )
        self.h5 = fh

    def load_sum_run_scattering(self, key, low=20, high=80):
        with h5py.File(self.run_file, "r") as fh:
            setattr(
                self,
                "scattering",
                np.nansum(np.nansum(fh[key][:, :, low:high], axis=1), axis=1),
            )

    def close_h5(self):
        self.h5.close()
        del self.h5

    def purge_all_keys(self, keys_to_keep):
        keys_to_keep = set(keys_to_keep)
        new_dict = {
            attr: value for attr, value in self.__dict__.items() if attr in keys_to_keep
        }
        self.__dict__ = new_dict

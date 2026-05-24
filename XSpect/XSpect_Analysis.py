import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import ipywidgets as widgets
from IPython.display import display
from scipy.ndimage import  rotate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit,minimize
from scipy.signal import savgol_filter
import multiprocessing
import os
from functools import partial
import time
import sys
import argparse
from datetime import datetime
import tempfile

class experiment:
    def __init__(self, lcls_run, hutch, experiment_id):
        """
        Initializes an experiment instance.

        Parameters
        ----------
        lcls_run : str
            LCLS run identifier. The LCLS run not the scan/run. Example: 21
        hutch : str
            Hutch name. Example: xcs
        experiment_id : str
            Experiment identifier. Example: xcsl1004021
        """
        self.lcls_run = lcls_run
        self.hutch = hutch
        self.experiment_id = experiment_id
        self.get_experiment_directory()
    def get_experiment_directory(self):
        """
        Determines and returns the directory of the experiment based on the hutch and experiment ID. 
        It attempts the various paths LCLS has had over the years with recent S3DF paths being the first attempt.

        Returns
        -------
        str
            The directory of the experiment.

        Raises
        ------
        Exception
            If the directory cannot be found.
        """
        experiment_directories = [
        '/sdf/data/lcls/ds/%s/%s/hdf5/smalldata',
        '/reg/data/drpsrcf/%s/%s/scratch/hdf5/smalldata',
        '/cds/data/drpsrcf/%s/%s/scratch/hdf5/smalldata',
        '/reg/d/psdm/%s/%s/hdf5/smalldata'
        ]
        for directory in experiment_directories:
            experiment_directory = directory % (self.hutch, self.experiment_id)
            if os.path.exists(experiment_directory) and os.listdir(experiment_directory):
                self.experiment_directory=experiment_directory
                return experiment_directory
        raise Exception("Unable to find experiment directory.")

class spectroscopy_experiment(experiment):
    """
    A class to represent a spectroscopy experiment. 
    Trying to integrate methods that incorporate meta parameters of the experiment but did not follow through.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def add_detector(self, detector_name, detector_dimensions):
        self.detector_name = detector_name
        self.detector_dimensions = detector_dimensions

class spectroscopy_run:
    """
    A class to represent a run within a spectroscopy experiment. Not an LCLS run. 
    """
    def __init__(self,spec_experiment,run,verbose=False,end_index=-1,start_index=0):
        """
        Initializes a spectroscopy run instance.

        Parameters
        ----------
        spec_experiment : spectroscopy_experiment
            The parent spectroscopy experiment.
        run : int
            The run number.
        verbose : bool, optional
            Flag for verbose output used for printing all of the status updates. 
            These statuses are also available in the object itself. Defaults to False.
        end_index : int, optional
            Index to stop processing data. Defaults to -1.
        start_index : int, optional
            Index to start processing data. Defaults to 0.
            These indices are used for batch analysis. 
        """
        self.spec_experiment=spec_experiment
        self.run_number=run
        self.run_file='%s/%s_Run%04d.h5' % (self.spec_experiment.experiment_directory, self.spec_experiment.experiment_id, self.run_number)
        self.status=['New analysis of run %d located in: %s' % (self.run_number,self.run_file)]
        self.status_datetime=[datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        self.verbose=verbose
        self.end_index=end_index
        self.start_index=start_index

    def get_scan_val(self):
        """
        Retrieves the scan variable from the HDF5 file of the run. 
        This is specifically for runengine scans that tag the variable in the hdf5 file. E.g. useful for processing alignment scans
        """
        with h5py.File(self.run_file, 'r') as fh:
            self.scan_var=fh['scan/scan_variable']
            
        
    def update_status(self,update):
        """
        Updates the status log for the run and appends it to the objects status/datetime attibutes.
        If verbose then it prints it.
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
        Retrieves shot properties from the run file, including total shots and simultaneous laser and X-ray shots.
        """
        with h5py.File(self.run_file, 'r') as fh:
            if self.end_index == -1:
                self.total_shots = fh['lightStatus/xray'][self.start_index:].shape[0]
                xray_total = np.sum(fh['lightStatus/xray'][self.start_index:])
                laser_total = np.sum(fh['lightStatus/laser'][self.start_index:])
                self.xray = np.array(fh['lightStatus/xray'][self.start_index:])
                self.laser = np.array(fh['lightStatus/laser'][self.start_index:])
                self.simultaneous=np.logical_and(self.xray,self.laser)
            else:
                self.total_shots = fh['lightStatus/xray'][self.start_index:self.end_index].shape[0]
                xray_total = np.sum(fh['lightStatus/xray'][self.start_index:self.end_index])
                laser_total = np.sum(fh['lightStatus/laser'][self.start_index:self.end_index])
                self.xray = np.array(fh['lightStatus/xray'][self.start_index:self.end_index])
                self.laser = np.array(fh['lightStatus/laser'][self.start_index:self.end_index])
                self.simultaneous=np.logical_and(self.xray,self.laser)
            
        self.run_shots={'Total':self.total_shots,'X-ray Total':xray_total,'Laser Total':laser_total}
        self.update_status('Obtained shot properties')
    def set_arbitrary_filter(self,key='arbitrary_filter'):
        self.verbose=False
        with h5py.File(self.run_file, 'r') as fh:
            self.arbitrary_filter = fh[key][self.start_index:self.end_index]
    
    def load_run_keys(self, keys, friendly_names):
        """
        Loads specified keys from the run file into memory.

        Parameters
        ----------
        keys : list
            List of keys to load from the hdf5 file
        friendly_names : list
            Corresponding list of friendly names for the keys. Some keys are special to the subsequent analyis e.g. epix and ipm. 
        """
        start=time.time()
        with h5py.File(self.run_file, 'r') as fh:
            for key, name in zip(keys, friendly_names):
                
                try:
                    if self.end_index == -1:
                        setattr(self, name, np.array(fh[key][self.start_index:]))
                    else:
                        setattr(self, name, np.array(fh[key][self.start_index:self.end_index]))
                except KeyError as e:
                    self.update_status('Key does not exist: %s' % e.args[0])
                except MemoryError:
                    setattr(self, name, fh[key])
                    self.update_status('Out of memory error while loading key: %s. Not converted to np.array.' % key)
        end=time.time()
        self.update_status('HDF5 import of keys completed. Time: %.02f seconds' % (end-start))
    def load_run_key_delayed(self, keys, friendly_names, transpose=False, rois=None, combine=True):
        """
        Loads specified keys from the run file into memory without immediate conversion to numpy arrays. 
        Supports applying multiple ROIs in one dimension that can be combined into a single mask or handled separately.

        Parameters
        ----------
        keys : list
            List of keys to load.
        friendly_names : list
            Corresponding list of friendly names for the keys.
        transpose : bool, optional
            Flag to transpose the loaded data. Defaults to False.
        rois : list of lists, optional
            List of ROIs (regions of interest) as pixel ranges along one dimension (default is None).
            Each ROI should be in the form [start_col, end_col].
        combine : bool, optional
            Whether to combine ROIs into a single mask. Defaults to True.
        """
        start = time.time()
        fh = h5py.File(self.run_file, 'r')

        for key, name in zip(keys, friendly_names):
            try:
                # Load the data from the file for the given key
                if self.end_index == -1:
                    data = fh[key][self.start_index:, :, :]
                else:
                    data = fh[key][self.start_index:self.end_index, :, :]

                #print(data.shape)

                if transpose:
                    data = np.transpose(data, axes = (0, 2, 1))
                    setattr(self, name, data)

                # Apply one-dimensional ROIs if specified
                if rois is not None:
                    if combine:
                        # Combine multiple ROIs into a single mask
                        mask = np.zeros(data.shape[2], dtype=bool)  # Mask along the third dimension (spatial)
                        for roi in rois:
                            start_col, end_col = roi
                            mask[start_col:end_col] = True
                        # Apply the mask to select the ROI from the third dimension
                        data = data[:, :, mask]
                        setattr(self, f"{name}_ROI_1", data)
                    else:
                        # Handle each ROI separately, storing the results as different attributes
                        for idx, roi in enumerate(rois):
                            start_col, end_col = roi
                            roi_data = data[:, :, start_col:end_col]
                            setattr(self, f"{name}_ROI_{idx+1}", roi_data)

                setattr(self, name, data)

                # if transpose:
                #     setattr(self, name, np.transpose(data, axes=(1, 2)))

            except KeyError as e:
                self.update_status(f'Key does not exist: {e.args[0]}')
            except MemoryError:
                if self.end_index == -1:
                    setattr(self, name, fh[key][self.start_index:, :, :])
                else:
                    setattr(self, name, fh[key][self.start_index:self.end_index, :, :])
                self.update_status(f'Out of memory error while loading key: {key}. Not converted to np.array.')

        end = time.time()
        self.update_status(f'HDF5 import of keys completed. Time: {end - start:.02f} seconds')
        self.h5 = fh



    def load_sum_run_scattering(self,key,low=20,high=80):
        """
        Sums the scattering data across the specified range.

        Parameters
        ----------
        key : str
            The key to sum the scattering data from.
        low : int
            Low index for summing
        high: int 
            high index for summing
            These indices should be chosen over the water ring or some scattering of interest.
        """
        with h5py.File(self.run_file, 'r') as fh:
            setattr(self, 'scattering', np.nansum(np.nansum(fh[key][:,:,low:high],axis=1),axis=1))
        
    def close_h5(self):
        """
        Closes the HDF5 file handle.
        Again, avoiding memory issues.
        """
        self.h5.close()
        del self.h5
    
    def purge_all_keys(self,keys_to_keep):
        """
        Purges all keys from the object except those specified. Again avoid OOM in the analyis object.

        Parameters
        ----------
        keys_to_keep : list
            List of keys to retain.
        """
                
        keys_to_keep = set(keys_to_keep)  # Remove duplicates by converting to a set
        new_dict = {attr: value for attr, value in self.__dict__.items() if attr in keys_to_keep}
        self.__dict__ = new_dict
        
class SpectroscopyAnalysis:
    """
    A class to perform analysis on spectroscopy data.
    """
    def __init__(self):
        pass
    
    def bin_uniques(self,run,key):
        """
        Bins unique values for a given key within a run.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        key : str
            The key for which unique values are to be binned.
        """
        vals = getattr(run,key)
        run.scanvar_bins = np.unique(vals)
        bins_centered, run.scanvar_indices = self.center_binning(vals, run.scanvar_bins)
        run.scanvar_indices = run.scanvar_indices - 1

    def reduce_detector_1D(self, run, detector_key, axis1_key_bins, axis1_key_indices, average = True):
        """
        Reduce detector data over 1 dimension.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        axis1_bins : str
            The key corresponding to 1st dimension bins (e.g., time_bins)
        axis1_key_indices : str
            The key corresponding to the 1st dimension bin indices
        average : bool, optional
            Whether to average the reduced data (default is True)
        """
        detector = getattr(run, detector_key)
        axis1_bins = getattr(run, axis1_key_bins)
        axis1_indices = getattr(run, axis1_key_indices)

        if len(detector.shape) < 2:
            reduced_array = np.zeros((axis1_bins.shape[0]))
        elif len(detector.shape) < 3:
            reduced_array = np.zeros((axis1_bins.shape[0], detector.shape[1]))
        elif len(detector.shape) == 3:
            reduced_array = np.zeros((axis1_bins.shape[0], detector.shape[1], detector.shape[2]))

        reduced_std = np.zeros_like(reduced_array)

        counts = np.bincount(axis1_indices)
        if average:
            np.add.at(reduced_array, axis1_indices, detector)
            reduced_array /= counts[:, None]
        else:
            np.add.at(reduced_array, axis1_indices, detector)

        for i in np.arange(axis1_bins.shape[0]):
            reduced_std[i] = np.nanstd(detector[axis1_indices == i][:], axis = 0)

        setattr(run, detector_key+'_'+axis1_key_bins.split('_')[0]+'_binned', reduced_array)
        setattr(run, detector_key+'_bincount', counts)
        setattr(run, detector_key+'_std',reduced_std)

        run.update_status('Detector %s binned in 1D (%s) into key: %s'%(detector_key, axis1_key_bins.split('_')[0], detector_key+'_'+axis1_key_bins.split('_')[0]+'_binned') )

    def reduce_detector_2D(self, run, detector_key, axis1_key_bins, axis1_key_indices, axis2_key_bins, axis2_key_indices, average=True):
        """
        Reduce detector data over 2 dimensions.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        axis1_bins : str
            The key corresponding to 1st dimension bins (e.g., time_bins)
        axis1_key_indices : str
            The key corresponding to the timing bin indices.
        axis2_bins : str
            The key corresponding to 2nd dimension bins (e.g., energy_bins)
        axis2_key_indices : str
            The key corresponding to the energy bin indices.
        average : bool, optional
            Whether to average the reduced data (default is True).
        """
        detector = getattr(run, detector_key)
        axis1_bins = getattr(run, axis1_key_bins)
        axis1_indices = getattr(run, axis1_key_indices)
        axis2_bins = getattr(run, axis2_key_bins)
        axis2_indices = getattr(run, axis2_key_indices)
        
        if len(detector.shape) < 2:
            reduced_array = np.zeros((axis1_bins.shape[0], axis2_bins.shape[0]))
        elif len(detector.shape) < 3:
            reduced_array = np.zeros((axis1_bins.shape[0], axis2_bins.shape[0], detector.shape[1]))
        elif len(detector.shape) == 3:
            reduced_array = np.zeros((axis1_bins.shape[0], axis2_bins.shape[0], detector.shape[1], detector.shape[2]))
        
        axis1_indices = getattr(run, axis1_key_indices)#digitized indices from detector
        axis2_indices = getattr(run, axis2_key_indices)#digitized indices from detector
        
        reduced_std = np.zeros_like(reduced_array)
        
        unique_indices = np.column_stack((axis1_indices, axis2_indices))

        counts = np.zeros_like(reduced_array)
        for ii in np.arange(reduced_array.shape[0]):
            for jj in np.arange(reduced_array.shape[1]):
                mask = (axis1_indices == ii) & (axis2_indices == jj)
                reduced_std[ii,jj] = np.nanstd(detector[mask], axis = 0)
                counts[ii,jj] = np.nansum(mask)

        np.add.at(reduced_array, (unique_indices[:, 0], unique_indices[:, 1]), detector)
        
        setattr(run, detector_key+'_'+axis1_key_bins.split('_')[0]+'_'+axis2_key_bins.split('_')[0]+'_binned', reduced_array)
        setattr(run, detector_key+'_bincount', counts)
        setattr(run, detector_key+'_std',reduced_std)
        
        run.update_status('Detector %s binned in 2D (%s vs %s) into key: %s'%(detector_key, axis1_key_bins.split('_')[0], axis2_key_bins.split('_')[0], detector_key+'_'+axis1_key_bins.split('_')[0]+'_'+axis2_key_bins.split('_')[0]+'_binned') )
    
    def filter_shots(self, run,shot_mask_key, filter_key='ipm', threshold=1.0E4):
        """
        Filters shots based on a given threshold.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        shot_mask_key : str
            The key corresponding to the shot mask. An example being [xray,simultaneous,laser] for all x-ray shots
        filter_key : str, optional
            The key corresponding to the filter data (default is 'ipm'). 
        threshold : float, optional
            The threshold value for filtering (default is 1.0E4).
        So if we filter: xray,ipm,1E4 then X-ray shots will be filtered out if the ipm is below 1E4.
        """
        shot_mask=getattr(run,shot_mask_key)
        count_before=np.sum(shot_mask)
        filter_mask=getattr(run,filter_key)
        nan_mask = np.isnan(filter_mask)
        if isinstance(threshold, int) or isinstance(threshold, float):
            filtered_shot_mask=shot_mask * (filter_mask>threshold)* (~nan_mask)
        elif len(threshold) == 2:
            filtered_shot_mask=shot_mask * (filter_mask>threshold[0])* (filter_mask<threshold[1])* (~nan_mask)
        count_after=np.sum(filtered_shot_mask)
        setattr(run,shot_mask_key,filtered_shot_mask)
        
        if isinstance(threshold, int) or isinstance(threshold, float):
            run.update_status('Mask: %s has been filtered on %s by minimum threshold: %0.3f\nShots removed: %d' % (shot_mask_key,filter_key,threshold,count_before-count_after))
        elif len(threshold) == 2:
            run.update_status('Mask: %s has been filtered on %s by minimum threshold: %0.3f and maximum threshold: %0.3f\nShots removed: %d' % (shot_mask_key,filter_key,threshold[0], threshold[1],count_before-count_after))
    
    def filter_nan(self, run,shot_mask_key, filter_key='ipm'):
        """
        A specific filtering implementation for Nans due to various DAQ issues. 
        Filters out shots with NaN values in the specified filter.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        shot_mask_key : str
            The key corresponding to the shot mask.
        filter_key : str, optional
            The key corresponding to the filter data (default is 'ipm').
        """
        shot_mask=getattr(run,shot_mask_key)
        count_before=np.sum(shot_mask)
        filter_mask=getattr(run,filter_key)
        filtered_shot_mask=shot_mask * (filter_mask>threshold)
        count_after=np.sum(filtered_shot_mask)
        setattr(run,shot_mask_key,filtered_shot_mask)
        run.update_status('Mask: %s has been filtered on %s by minimum threshold: %0.3f\nShots removed: %d' % (shot_mask_key,filter_key,threshold,count_before-count_after))

    
    def filter_detector_adu(self,run,detector,adu_threshold=3.0):
        """
        Filters is a misnomer compared to the other filter functions. 
        This sets detector pixel values below a threshold to 0.
        Specifically, to remove 0-photon noise from detectors. 

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        detector : str
            The key corresponding to the detector data.
        adu_threshold : float or list of float, optional
            The ADU threshold for filtering. Can be a single value or a range (default is 3.0).
        
        Returns
        -------
        np.ndarray
            The filtered detector data.
        """
        detector_images=getattr(run,detector)
        if isinstance(adu_threshold,list):
            detector_images_adu = detector_images * (detector_images > adu_threshold[0])
            detector_images_adu = detector_images_adu * (detector_images_adu < adu_threshold[1])
            run.update_status('Key: %s has been adu filtered by thresholds: %f,%f' % (detector,adu_threshold[0],adu_threshold[1]))
        else:
            detector_images_adu = detector_images * (detector_images > adu_threshold)
            run.update_status('Key: %s has been adu filtered by threshold: %f' % (detector,adu_threshold))

        setattr(run,detector,detector_images_adu)

        return detector_images_adu
        
    def purge_keys(self,run,keys):
        """
        Purges specific keys from the run to save memory.
        This is specifically to remove the epix key immediately after processing it from the hdf5 file.
        To avoid OOM. This is different than the purge all keys method which is used to purge many of the larger analysis steps.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        keys : list of str
            The list of keys to purge.
        """
        for detector_key in keys:
            setattr(run, detector_key, None)
            run.update_status(f"Purged key to save room: {detector_key}")

    def droplet_reconstruction(self, run, detector_key, detector_friendly_name, rois=None, shot_range = [0, None], transpose = False):
        """
        Will reconstruct detector images per shot from droplet analysis if contained in detector key of hdf5 file.
        If ROIs are specified - will only reconstruct ROI images per shot.
        If no ROIs are specified, will reconstruct full detector image per shot.
        """
        start = time.time()
        
        fh = h5py.File(run.run_file, 'r')
        
        try:
            detector_group = detector_key[0].split('/')[0]
    
            nDroplets_total = np.array(fh[detector_group]['droplet_nDroplets'])
            nDroplets = nDroplets_total[shot_range[0]:shot_range[1]]
    
            indx_start = np.nansum(nDroplets_total[0:shot_range[0]])
            indx_end = np.nansum(nDroplets_total[0:shot_range[1]])

            if transpose:
                cols = np.array(fh[detector_group]['var_droplet_sparse']['row'])[indx_start:indx_end]
                rows = np.array(fh[detector_group]['var_droplet_sparse']['col'])[indx_start:indx_end]
            else:
                cols = np.array(fh[detector_group]['var_droplet_sparse']['col'])[indx_start:indx_end]
                rows = np.array(fh[detector_group]['var_droplet_sparse']['row'])[indx_start:indx_end]
            data = np.array(fh[detector_group]['var_droplet_sparse']['data'])[indx_start:indx_end]

        except:
            pass

        rows_indx, cols_indx = rows - 1, cols - 1
        rows_indx, cols_indx = rows_indx.astype(int), cols_indx.astype(int)
        
        if rois != None:

            ROI_dict = {}

            for i, ROI in enumerate(rois):
                ROIstr = 'ROI_%i' % (i+1)
                ROI_len = ROI[1] - ROI[0]
                ROI_dict[ROIstr] = np.zeros((nDroplets.shape[0], 702, ROI_len))

                ndrops_ROI = np.zeros_like(nDroplets)

                ROImask = (cols_indx >= ROI[0]) & (cols_indx < ROI[1])

                rows_indx_ROI = rows_indx[ROImask]
                cols_indx_ROI = cols_indx[ROImask]
                data_ROI = data[ROImask]

                start_indx = 0
                for ii, ndrops in enumerate(nDroplets):
                    if ii == 0:
                        indices_per_shot = np.arange(ndrops)

                    else:
                        indices_per_shot = np.arange(start_indx, start_indx + ndrops, 1)

                    ndrops_ROI[ii] = np.sum(ROImask[start_indx:(start_indx + ndrops)])

                    if ndrops > 0:
                        start_indx = indices_per_shot[-1]+1

                start_indx = 0
                for iii, ndrops in enumerate(ndrops_ROI):
                    if iii == 0:
                        indices_per_shot = np.arange(ndrops)

                    else:
                        indices_per_shot = np.arange(start_indx, start_indx + ndrops, 1)

                    for j in indices_per_shot:
                        ROI_dict[ROIstr][iii, rows_indx_ROI[j], (cols_indx_ROI[j] - ROI[0])] = data_ROI[j]

                    if ndrops > 0:
                        start_indx = indices_per_shot[-1]+1

                setattr(run, f"{detector_friendly_name[0]}_{ROIstr}", ROI_dict[ROIstr])
                
        else:
            
            data_reconstructed = np.zeros((nDroplets.shape[0], 702, 766))

            start_indx = 0
            for i, ndrops in enumerate(nDroplets):
                if i == 0:
                    indices_per_shot = np.arange(ndrops)

                else:
                    indices_per_shot = np.arange(start_indx, start_indx + ndrops, 1)

                for j in indices_per_shot:
                    data_reconstructed[i, rows_indx[j], cols_indx[j]] = data[j]

                if ndrops > 0:
                    start_indx = indices_per_shot[-1]+1

            setattr(run, detector_friendly_name[0], data_reconstructed)

        end = time.time()
        run.update_status(f'Droplet reconstruction completed. Time: {end - start:.02f} seconds')
        
    
    def reduce_detector_shots(self, run, detector_key,reduction_function=np.sum,  purge=True,new_key=False):
        detector = getattr(run, detector_key)
        reduced_data=reduction_function(detector,axis=0)
        run.update_status(f"Reduced detector by shots: {detector_key} with number of shots: {np.shape(detector)}")
        if new_key:
            target_key=f"{detector_key}_summed"
        else:
            target_key=detector_key
        setattr(run, target_key, reduced_data)
        if purge:
            setattr(run, detector_key,None)
            run.update_status(f"Purged key to save room: {detector_key}")

    def apply_roi(self, run, detector_key, shot_range = [0, None], rois = [[0, None]], combine = True):
        detector = getattr(run, detector_key)
        if combine:
            
            roi_combined = [rois[0][0], rois[-1][1]]  # Combined ROI spanning the first and last ROI
            mask = np.zeros(detector.shape[-1], dtype=bool)
            for roi in rois:
                mask[roi[0]:roi[1]] = True
            if detector.ndim==3:
                masked_data = detector[shot_range[0]:shot_range[1], :, :][:, :, mask]
            elif detector.ndim==2:
                masked_data = detector[:, mask]
            elif detector.ndim==1:
                masked_data = detector[mask]
            roi_indices = ', '.join([f"{roi[0]}-{roi[1]}" for roi in rois])
            run.update_status(f"Applied ROIs to detector: {detector_key} with combined ROI indices: {roi_indices}")
            setattr(run, f"{detector_key}_ROI_1", masked_data)
        else:
            for idx, roi in enumerate(rois):
                # print(roi)
                data_chunk = detector[shot_range[0]:shot_range[1], :, roi[0]:roi[1]]
                
                if roi[1] is None:
                    roi[1] = detector.shape[1] - 1
                    
                run.update_status(f"Applied ROIs to detector: {detector_key} with ROI: {roi[0]}, {roi[1]}")
                # print(data_chunk.shape)
                setattr(run, f"{detector_key}_ROI_{idx+1}", data_chunk)
        
    
    def reduce_detector_spatial(self, run, detector_key, shot_range=[0, None], rois=[[0, None]], reduction_function=np.sum,  purge=True, combine=True):
        """
        Reduces the spatial dimension of detector data based on specified ROIs.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        shot_range : list, optional
            The range of shots to consider (default is [0, None]).
        rois : list of lists, optional
            The list of ROIs (regions of interest) as pixel ranges (default is [[0, None]]).
        reduction_function : function, optional
            The function to apply for reduction (default is np.sum).
        purge : bool, optional
            Whether to purge the original detector data after reduction (default is True).
        combine : bool, optional
            Whether to combine ROIs (default is True).
        """
        detector = getattr(run, detector_key)
        if combine:
            
            roi_combined = [rois[0][0], rois[-1][1]]  # Combined ROI spanning the first and last ROI
            mask = np.zeros(detector.shape[-1], dtype=bool)
            for roi in rois:
                mask[roi[0]:roi[1]] = True
            if detector.ndim==3:
                masked_data = detector[shot_range[0]:shot_range[1], :, :][:, :, mask]
            elif detector.ndim==2:
                masked_data = detector[:, mask]
            elif detector.ndim==1:
                masked_data = detector[mask]
            reduced_data = reduction_function(masked_data, axis=-1)
            roi_indices = ', '.join([f"{roi[0]}-{roi[1]}" for roi in rois])
            run.update_status(f"Spatially reduced detector: {detector_key} with combined ROI indices: {roi_indices}")
            setattr(run, f"{detector_key}_ROI_1", reduced_data)
        else:
            for idx, roi in enumerate(rois):
                data_chunk = detector[shot_range[0]:shot_range[1], roi[0]:roi[1]]
                reduced_data = reduction_function(data_chunk, axis = -1)
            if roi[1] is None:
                roi[1] = detector.shape[1] - 1
                run.update_status(f"Spatially reduced detector: {detector_key} with ROI: {roi[0]}, {roi[1]}")
                setattr(run, f"{detector_key}_ROI_{idx+1}", reduced_data)
        if purge:
            #pass
            setattr(run, detector_key,None)
            #delattr(run, detector_key)
            #del run.detector_key
            run.update_status(f"Purged key after spatial reduction to save room: {detector_key}")

    def time_binning(self,run,bins,lxt_key='lxt_ttc',fast_delay_key='encoder',tt_correction_key='time_tool_correction'):
        """
        Bins data in time based on specified bins. Units in picoseconds.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        bins : array-like
            The bins to use for time binning.
        lxt_key : str, optional
            The key for the laser time delay data (default is 'lxt_ttc').
        fast_delay_key : str, optional
            The key for the fast delay data (default is 'encoder').
        tt_correction_key : str, optional
            The key for the time tool correction data (default is 'time_tool_correction').
        """

        # Check magnitude of timing data by taking mean of absolute value of array and comparing to threshold.
        a = np.nanmean(np.absolute(getattr(run, lxt_key)))
        b = np.nanmean(np.absolute(getattr(run, fast_delay_key)))
        c = np.nanmean(np.absolute(getattr(run, tt_correction_key)))
        if not all(x > 0.001 for x in [a, b, c]):
            run.update_status('------Timing data values are either very small or zero. Confirm the units and keys are correct-----\n-----Mean abs value of: lxt_key: %f, fast_delay: %f, tt_correction: %f -----' % (a, b, c))
        # Generate delays, time_bins and binning
        delays = np.array(getattr(run,lxt_key)*(1e12)).flatten() + np.array(getattr(run,fast_delay_key)).flatten()  + np.array(getattr(run,tt_correction_key)).flatten()

        run.delays=delays
        run.time_bins=bins
        run.time_bins_centered, run.timing_bin_indices = self.center_binning(delays, run.time_bins)
        run.timing_bin_indices = run.timing_bin_indices - 1
        
        run.update_status('Generated timing bins from %f to %f in %d steps.' % (np.min(bins),np.max(bins),len(bins)))

    def center_binning(self, data2bin, binlist):
        """
        np.digitize will take a list of bins and bin an array using the list as bin edges.
        This function takes a list of bins (for example, time or energy) and creates a new set 
        of bin edges such that the given binlist (that will become time or energy axis)
        represents central values of the bins and then bin the desired data that way
        
        Parameters
        ----------
        data2bin : array
            Array of data to bin (delay or energy data)
        binlist : array
            The desired set of bins you want to bin the data over
        """
        bin_addon = (binlist[-1] - binlist[-2])/2
        binlist_expanded = np.append(binlist, binlist[-1] + bin_addon)
        binlist_centered = np.empty_like(binlist_expanded)

        for ii in np.arange(binlist.shape[0]):
            if ii == 0:
                binlist_centered[ii] = binlist_expanded[ii] - (binlist_expanded[ii+1] - binlist_expanded[ii])/2
            else:
                binlist_centered[ii] = binlist_expanded[ii] - (binlist_expanded[ii] - binlist_expanded[ii-1])/2

                binlist_centered[-1] = binlist_expanded[-1]

        data_binned = np.digitize(data2bin, bins = binlist_centered)

        return binlist_centered, data_binned
        
    def union_shots(self, run, detector_key, filter_keys,new_key=True):
        """
        Combines shots across multiple filters into a single array. 
        So union_shots(f,'timing_bin_indices',['simultaneous','laser'])
        means go through the timing_bin_indices and find the ones that correspond to X-rays and laser shots.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        filter_keys : list of str
            The list of filter keys to combine.
        """
        detector = getattr(run, detector_key)
        
        if isinstance(filter_keys, list):
            mask = np.logical_and.reduce([getattr(run, k) for k in filter_keys])
        else:
            mask = getattr(run, filter_keys)
        filtered_detector = detector[mask]
        if new_key:
            target_key=detector_key + '_' + '_'.join(filter_keys)
        else:
            target_key=detector_key
        setattr(run, target_key, filtered_detector)
        run.update_status('Shots (%d) combined for detector %s on filters: %s and %s into %s'%(np.sum(mask), detector_key, filter_keys[0],filter_keys[1],target_key))
        
    def separate_shots(self, run, detector_key, filter_keys):
        """
        Separates shots into different datasets based on filters.
        separate_shots(f,'epix_ROI_1',['xray','laser']) means find me the epix_ROI_1 images in shots that were X-ray but NOT laser.
        If you wanted the inverse you would switch the order of the filter_keys.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        filter_keys : list of str
            The list of filter keys to separate.
        """
        detector = getattr(run, detector_key)
        if isinstance(filter_keys, list):
            mask1 = getattr(run, filter_keys[0])
            mask2 = np.logical_not(getattr(run, filter_keys[1]))
            mask = np.logical_and(mask1, mask2)
        else:
            mask = getattr(run, filter_keys)
        filtered_detector = detector[mask]
        setattr(run, detector_key + '_' +filter_keys[0]+'_not_'+filter_keys[1], filtered_detector)
        run.update_status('Shots (%d) separated for detector %s on filters: %s and %s into %s'%(np.sum(mask),detector_key,filter_keys[0],filter_keys[1],detector_key + '_' +filter_keys[0]+'_not_'+filter_keys[1]))
    
    def reduce_detector_temporal(self, run, detector_key, timing_bin_key_indices,average=False):
        """
        Reduces the temporal dimension of detector data based on timing bins.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        timing_bin_key_indices : str
            The key corresponding to the timing bin indices.
        average : bool, optional
            Whether to average the data within each bin (default is False).
        """
        detector = getattr(run, detector_key)
        indices = getattr(run, timing_bin_key_indices)
        expected_length = len(run.time_bins)
        if len(detector.shape) < 2:
            reduced_array = np.zeros((expected_length))
        elif len(detector.shape) < 3:
            reduced_array = np.zeros((expected_length, detector.shape[1]))
        elif len(detector.shape) == 3:
            reduced_array = np.zeros((expected_length, detector.shape[1], detector.shape[2]))
        reduced_std = np.zeros_like(reduced_array)

        counts = np.bincount(indices)
        if average:
            np.add.at(reduced_array, indices, detector)
            reduced_array /= counts[:, None]
        else:
            np.add.at(reduced_array, indices, detector)

        for i in np.arange(expected_length):
            reduced_std[i] = np.nanstd(detector[indices == i][:], axis = 0)
            
        setattr(run, detector_key+'_time_binned', reduced_array)
        setattr(run, detector_key+'_bincount', counts)
        setattr(run, detector_key+'_std',reduced_std)
        run.update_status('Detector %s binned in time into key: %s from detector shape: %s to reduced shape: %s'%(detector_key,detector_key+'_time_binned', detector.shape,reduced_array.shape) )
    def patch_pixels(self,run,detector_key,  mode='average', patch_range=4, deg=1, poly_range=6,axis=1):
        """
        Patches multiple pixels in detector data.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        mode : str, optional
            The mode of patching ('average', 'polynomial', or 'interpolate').
        patch_range : int, optional
            The range around the pixel to use for patching (default is 4).
        deg : int, optional
            The degree of the polynomial for polynomial patching (default is 1).
        poly_range : int, optional
            The range of pixels to use for polynomial or interpolation patching (default is 6).
        axis : int, optional
            The axis along which to apply the patching (default is 1).
        """
        for pixel in self.pixels_to_patch:
            self.patch_pixel(run,detector_key,pixel,mode,patch_range,deg,poly_range,axis=axis)


    def patch_pixel(self, run, detector_key, pixel, mode='average', patch_range=4, deg=1, poly_range=6, axis=1):
        """
        EPIX detector pixel patching.
        TODO: extend to patch regions instead of per pixel.
        Parameters
        ----------
        data : array_like
            Array of shots
        pixel : integer
            Pixel point to be patched
        mode : string
            Determines which mode to use for patching the pixel. Averaging works well.
        patch_range : integer
            Pixels away from the pixel to be patched to be used for patching. Needed if multiple pixels in a row are an issue.
        deg : integer
            Degree of polynomial if polynomial patching is used.
        poly_range : integer
            Number of pixels to include in the polynomial or interpolation fitting
        Returns
        -------
        float
            The original data with the new patch values.
        """
        data = getattr(run, detector_key)

        def get_neighbor_values(data, pixel, patch_range, axis):
            axis_slice = [slice(None)] * data.ndim
            start_index = max(pixel - patch_range, 0)
            end_index = min(pixel + patch_range + 1, data.shape[axis])
            axis_slice[axis] = slice(start_index, end_index)
            return data[tuple(axis_slice)]

        def patch_value_average(data, pixel, patch_range, axis):
            neighbor_values = get_neighbor_values(data, pixel, patch_range, axis)
            neighbor_values = np.moveaxis(neighbor_values, axis, 0)
            new_val = np.mean(neighbor_values, axis=0)
            return new_val

        def patch_value_polynomial(data, pixel, patch_range, poly_range, deg, axis):
            patch_x = np.arange(pixel - patch_range - poly_range, pixel + patch_range + poly_range + 1)
            patch_range_weights = np.ones(len(patch_x))
            patch_range_weights[patch_range:-patch_range] = 0.001

            neighbor_values = get_neighbor_values(data, pixel, patch_range + poly_range, axis)
            neighbor_values = np.moveaxis(neighbor_values, axis, 0)

            new_vals = []
            for idx in range(neighbor_values.shape[1]): 
                ys = neighbor_values[:, idx]
                coeffs = np.polyfit(patch_x, ys, deg, w=patch_range_weights)
                new_vals.append(np.polyval(coeffs, pixel))
            return np.array(new_vals)

        def patch_value_interpolate(data, pixel, patch_range, poly_range, axis):
            patch_x = np.arange(pixel - patch_range - poly_range, pixel + patch_range + poly_range + 1)
            neighbor_values = get_neighbor_values(data, pixel, patch_range + poly_range, axis)
            neighbor_values = np.moveaxis(neighbor_values, axis, 0)

            new_vals = []
            for idx in range(neighbor_values.shape[1]):
                ys = neighbor_values[:, idx]
                interp_func = interp1d(patch_x, ys, kind='quadratic')
                new_vals.append(interp_func(pixel))
            return np.array(new_vals)

        if mode == 'average':
            new_val = patch_value_average(data, pixel, patch_range, axis)
        elif mode == 'polynomial':
            new_val = patch_value_polynomial(data, pixel, patch_range, poly_range, deg, axis)
        elif mode == 'interpolate':
            new_val = patch_value_interpolate(data, pixel, patch_range, poly_range, axis)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        patch_slice = [slice(None)] * data.ndim
        patch_slice[axis] = pixel
        data[tuple(patch_slice)] = new_val

        setattr(run, detector_key, data)
        run.update_status(f"Detector {detector_key} pixel {pixel} patched. Old value.")
    
    def patch_pixels_1d(self,run,detector_key,  mode='average', patch_range=4, deg=1, poly_range=6):
        """
        Patches multiple pixels in 1D detector data.

        Parameters
        ----------
        run : spectroscopy_run
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        mode : str, optional
            The mode of patching ('average', 'polynomial', or 'interpolate').
        patch_range : int, optional
            The range around the pixel to use for patching (default is 4).
        deg : int, optional
            The degree of the polynomial for polynomial patching (default is 1).
        poly_range : int, optional
            The range of pixels to use for polynomial or interpolation patching (default is 6).
        """
        for pixel in self.pixels_to_patch:
            self.patch_pixel_1d(run,detector_key,pixel,mode,patch_range,deg,poly_range)
    def patch_pixel_1d(self, run, detector_key, pixel, mode='average', patch_range=4, deg=1, poly_range=6):
        """
        EPIX detector pixel patching.
        TODO: extend to patch regions instead of per pixel.
        Parameters
        ----------
        data : array_like
            Array of shots
        pixel : integer
            Pixel point to be patched
        mode : string
            Determined which mode to use for patching the pixel. Averaging works well.
        patch_range : integer
            pixels away from the pixel to be patched to be used for patching. Needed if multiple pixels in a row are an issue.
        deg : integer
            Degree of polynomial if polynomial patching is used.
        poly_range : integer
            Number of pixels to include in the polynomial or interpolation fitting
        Returns
        -------
        float
            The original data with the new patch values.
        """
        data = getattr(run, detector_key)
        if mode == 'average':
            neighbor_values = data[:, pixel - patch_range:pixel + patch_range + 1]
            data[:, pixel] = np.sum(neighbor_values, axis=1) / neighbor_values.shape[1]
        elif mode == 'polynomial':
            patch_x = np.arange(pixel - patch_range - poly_range, pixel + patch_range + poly_range + 1, 1)
            patch_range_weights = np.ones(len(patch_x))
            patch_range_weights[pixel - patch_range - poly_range:pixel + patch_range + poly_range] = 0.001
            coeffs = np.polyfit(patch_x, data[pixel - patch_range - poly_range:pixel + patch_range + poly_range + 1], deg,
                                w=patch_range_weights)
            data[pixel, :] = np.polyval(coeffs, pixel)
        elif mode == 'interpolate':
            patch_x = np.arange(pixel - patch_range - poly_range, pixel + patch_range + poly_range + 1, 1)
            interp = interp1d(patch_x, data[pixel - patch_range - poly_range:pixel + patch_range + poly_range + 1, :],
                              kind='quadratic')
            data[pixel, :] = interp(pixel)
        setattr(run,detector_key,data)
        run.update_status('Detector %s pixel %d patched in mode %s'%(detector_key, pixel,mode ))
        


class XESAnalysis(SpectroscopyAnalysis):
    def __init__(self,xes_line='kbeta'):
        self.xes_line=xes_line
        pass
    def normalize_xes(self,run,detector_key,pixel_range=[0,-1]):
        """
        Normalize XES data by summing the signal over a specified pixel range.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        pixel_range : list of int, optional
            The pixel range to sum over for normalization (default is [300, 550]).
        """
        detector = getattr(run, detector_key)
   
        row_sum = np.nansum(detector[:, pixel_range[0]:pixel_range[1]], axis=1)
        normed_main = np.divide(detector, row_sum[:,np.newaxis])
        setattr(run, detector_key+'_normalized', normed_main)
        try:
            std = getattr(run, detector_key + '_std')
            normed_std = np.divide(std, row_sum[:,np.newaxis])
            setattr(run, detector_key + '_normalized_std', normed_std)
        except:
            pass 
    def make_energy_axis(self, run, energy_axis_length, A, R, mm_per_pixel=0.05, d=0.895, name=None):
        """
        Determination of energy axis by pixels and crystal configuration

        Parameters
        ----------
        A : float
            The detector to vH distance (mm) and can roughly float. This will affect the spectral offset.
        R : float
            The vH crystal radii (mm) and should not float. This will affect the spectral stretch.
        pixel_array : array-like
            Array of pixels to determine the energy of.
        d : float
            Crystal d-spacing. To calculate, visit: spectra.tools/bin/controller.pl?body=Bragg_Angle_Calculator

        """
        pix = mm_per_pixel
        gl = np.arange(energy_axis_length, dtype=np.float64)
        gl *= pix
        ll = gl / 2 - (np.amax(gl) - np.amin(gl)) / 4
        factor = 1.2398e4
        xaxis = factor / (2.0 * d * np.sin(np.arctan(R / (ll + A))))

        if name is not None:
            setattr(run, name+'_energy', xaxis)
        else:
            name = self.xes_line
            setattr(run,name+'_energy',xaxis)
        run.update_status('XES energy axis generated for %s'%(name))

    def reduce_det_scanvar(self, run, detector_key, scanvar_key, scanvar_bins_key):
        """
        Reduce detector data by binning according to an arbitrary scan variable.

        This method bins the detector data based on a specified scan variable and its corresponding bins. 
        The result is stored in the `run` object under a new attribute.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data within the run object.
        scanvar_key : str
            The key corresponding to the scan variable indices.
        scanvar_bins_key : str
            The key corresponding to the scan variable bins.

        Returns
        -------
        None
            The reduced data is stored in the `run` object with the key formatted as `{detector_key}_scanvar_reduced`.
        """
    
        detector = getattr(run, detector_key)
        
        scanvar_indices = getattr(run, scanvar_key)  # Shape: (4509,)
        scanvar_bins=getattr(run, scanvar_bins_key)
        
        n_bins = len(scanvar_bins)  # Number of bins

        # Initialize reduced_array with the correct shape (number of bins, 699, 50)
        reduced_array = np.zeros((n_bins, detector.shape[1], detector.shape[2]))

        # Iterate over the images and accumulate them into reduced_array based on scanvar_indices
        for i in range(detector.shape[0]):
            np.add.at(reduced_array, (scanvar_indices[i],), detector[i])

        # Store the reduced_array in the object, replace 'key_name' with the actual key
        setattr(run,  f"{detector_key}_scanvar_reduced", reduced_array)

        # Update status
        run.update_status(f'Detector binned in time into key: {detector_key}_scanvar_reduced')

    def combine_runs(self, analysis_object, average_laser_off=False):
        roi_list = []
        for i in range(len(analysis_object.rois)):
            roi_list.append('epix_ROI_%i' % (i+1))
        setattr(analysis_object, 'roi_list', roi_list)

        for roi in roi_list:
            label_laser_off = roi + '_xray_not_laser_reduced_time_binned'
            xes = getattr(analysis_object.analyzed_runs[0], label_laser_off)
            label_laser_on = roi + '_simultaneous_laser_reduced_time_binned'
            xes_laser = getattr(analysis_object.analyzed_runs[0], label_laser_on)

            label_laser_off_std = roi + '_xray_not_laser_reduced_std'
            label_laser_on_std = roi + '_simultaneous_laser_reduced_std'

            summed_laser_off_coll = np.empty(((len(analysis_object.analyzed_runs),) + xes.shape))
            summed_laser_on_coll = np.empty(((len(analysis_object.analyzed_runs),) + xes.shape))
            summed_laser_off_var = np.empty(((len(analysis_object.analyzed_runs),) + xes.shape))
            summed_laser_on_var = np.empty(((len(analysis_object.analyzed_runs),) + xes.shape))
            

            for i, run in enumerate(analysis_object.analyzed_runs):
                summed_laser_off_coll[i,:] = getattr(run, label_laser_off)
                summed_laser_on_coll[i,:] = getattr(run, label_laser_on)
                summed_laser_off_var[i,:] = getattr(run, label_laser_off_std)**2
                summed_laser_on_var[i,:] = getattr(run, label_laser_on_std)**2

            summed_laser_off = np.nansum(summed_laser_off_coll, axis = 0)
            summed_laser_on = np.nansum(summed_laser_on_coll, axis = 0)
            summed_laser_off_std = np.sqrt(np.nansum(summed_laser_off_var, axis = 0))
            summed_laser_on_std = np.sqrt(np.nansum(summed_laser_on_var, axis = 0))
           
            if average_laser_off == True:
                summed_laser_off = np.nansum(summed_laser_off, axis = 0)
                summed_laser_off = np.tile(summed_laser_off, (summed_laser_on.shape[0], 1))
                summed_laser_off_var = summed_laser_off_std**2
                summed_laser_off_std = np.sqrt(np.nansum(summed_laser_off_var, axis = 0))
                summed_laser_off_std = np.tile(summed_laser_off_std, (summed_laser_on.shape[0], 1))

            setattr(analysis_object, roi + '_summed_laser_off', summed_laser_off)
            setattr(analysis_object, roi + '_summed_laser_on', summed_laser_on)
            setattr(analysis_object, roi + '_summed_laser_off_std', summed_laser_off_std)
            setattr(analysis_object, roi + '_summed_laser_on_std', summed_laser_on_std)

            self.normalize_xes(analysis_object, roi + '_summed_laser_off', pixel_range = [0,None])
            self.normalize_xes(analysis_object, roi + '_summed_laser_on', pixel_range = [0,None])

            setattr(analysis_object, roi + '_summed_difference_normalized', getattr(analysis_object, roi + '_summed_laser_on_normalized') - getattr(analysis_object, roi + '_summed_laser_off_normalized'))
            setattr(analysis_object, roi + '_summed_difference_normalized_std', np.sqrt(getattr(analysis_object, roi + '_summed_laser_on_normalized_std')**2 + getattr(analysis_object, roi + '_summed_laser_off_normalized_std')**2))

class XASAnalysis(SpectroscopyAnalysis):
    def __init__(self):
        pass;
    def trim_energy(self,run,threshold=120):
        """
        Trim energy values to remove bins with fewer shots than a specified threshold.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        threshold : int, optional
            The minimum number of shots required to keep a energy value (default is 120).
        """
        
        energy_bins=getattr(run,'energy_bins',elist_center)
        energies=getattr(run,'energies',elist)
        counts = np.bincount(bins)
        trimmed_energy=energies[counts[:-1]>120]
        self.make_energy_axis(run,energies)
        
    def make_energy_axis(self,run,energies):
        """
        Generate energy bins and centers from given energy values.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        energies : array-like
            Array of energy values to be used for creating energy bins.
        """
        elist=energies
        addon = (elist[-1] - elist[-2])/2
        elist2 = np.append(elist,elist[-1]+addon)
        elist_center = np.empty_like(elist)

        for ii in np.arange(elist_center.shape[0]):
            if ii == elist_center.shape[0]:
                elist_center[ii] = elist[-1]+addon
            else:
                elist_center[ii] = elist2[ii+1] - (elist2[ii+1] - elist2[ii])/2    
    
        setattr(run,'energy_bins',elist_center)
        setattr(run,'energies',elist)
        
    def reduce_detector_energy_temporal(self, run, detector_key, timing_bin_key_indices,energy_bin_key_indices,average=True):
        """
        Reduce detector data temporally and by energy bins.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        timing_bin_key_indices : str
            The key corresponding to the timing bin indices.
        energy_bin_key_indices : str
            The key corresponding to the energy bin indices.
        average : bool, optional
            Whether to average the reduced data (default is True).
        """
        detector = getattr(run, detector_key)
        timing_indices = getattr(run, timing_bin_key_indices)#digitized indices from detector
        energy_indices = getattr(run, energy_bin_key_indices)#digitized indices from detector
        reduced_array = np.zeros((np.shape(run.time_bins)[0], np.shape(run.energy_bins)[0]))
        reduced_std = np.zeros_like(reduced_array)
        
        unique_indices = np.column_stack((timing_indices, energy_indices))

        counts = np.zeros_like(reduced_array)
        for ii in np.arange(reduced_array.shape[0]):
            for jj in np.arange(reduced_array.shape[1]):
                mask = (timing_indices == ii) & (energy_indices == jj)
                reduced_std[ii,jj] = np.nanstd(detector[mask], axis = 0)
                counts[ii,jj] = np.nansum(mask)

        np.add.at(reduced_array, (unique_indices[:, 0], unique_indices[:, 1]), detector)

        
        setattr(run, detector_key+'_time_energy_binned', reduced_array)
        setattr(run, detector_key+'_bincount', counts)
        setattr(run, detector_key+'_std',reduced_std)
        
        run.update_status('Detector %s binned in time into key: %s'%(detector_key,detector_key+'_time_energy_binned') )

        
    def reduce_detector_energy(self, run, detector_key, energy_bin_key_indices, average = False, not_energy=False):
        """
        Reduce detector data by energy bins.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        energy_bin_key_indices : str
            The key corresponding to the energy bin indices.
        average : bool, optional
            Whether to average the reduced data (default is False).
        not_energy : bool, optional
            Whether to indicate that energy is not being used (default is False).

        """
        detector = getattr(run, detector_key)
        
        indices = getattr(run, energy_bin_key_indices)#digitized indices from detector
        if not_energy:
            reduced_array = np.zeros(np.max(indices)+1 )
        else:
            reduced_array = np.zeros(np.shape(run.energy_bins)[0]) 
        reduced_std = np.zeros_like(reduced_array)
        counts = np.zeros_like(reduced_array)
        
        # np.add.at(reduced_array, energy_indices, detector)
        if average:
            np.add.at(reduced_array, indices, detector)
            reduced_array /= counts[:, None]
        else:
            np.add.at(reduced_array, indices, detector)
            
        # counts = np.bincount(indices)
    
        for i in np.arange(reduced_std.shape[0]):
            mask = (indices == i)
            reduced_std[i] = np.nanstd(detector[mask][:], axis = 0)
            counts[i] = np.nansum(mask)

            
        setattr(run, detector_key+'_energy_binned', reduced_array)
        setattr(run, detector_key+'_bincount', counts)
        setattr(run, detector_key+'_std',reduced_std)
        
        run.update_status('Detector %s binned in energy into key: %s'%(detector_key,detector_key+'_energy_binned') )
        
    def reduce_detector_temporal(self, run, detector_key, timing_bin_key_indices, average=False):
        """
        Reduce detector data temporally. Specifically the 1d detector output for XAS data.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        timing_bin_key_indices : str
            The key corresponding to the timing bin indices.
        average : bool, optional
            Whether to average the reduced data (default is False).
        """
        detector = getattr(run, detector_key)
        time_bins=run.time_bins
        indices = getattr(run, timing_bin_key_indices)#digitized indices from detector
        reduced_array = np.zeros(np.shape(time_bins)[0])
        np.add.at(reduced_array, indices, detector)

        reduced_std = np.zeros_like(reduced_array)
        counts = np.zeros_like(reduced_array)

        # counts = np.bincount(indices)
        if average:
            np.add.at(reduced_array, indices, detector)
            reduced_array /= counts[:, None]
        else:
            np.add.at(reduced_array, indices, detector)

        expected_length = len(run.time_bins)
        for i in np.arange(expected_length):
            mask = (indices == i)
            reduced_std[i] = np.nanstd(detector[indices == i][:], axis = 0)
            counts[i] = np.nansum(mask)

        setattr(run, detector_key+'_time_binned', reduced_array)
        setattr(run, detector_key+'_bincount', counts)
        setattr(run, detector_key+'_std',reduced_std)
        run.update_status('Detector %s binned in time into key: %s'%(detector_key,detector_key+'_time_binned') )
        
    def energy_binning(self,run,energy_bins,energy_key='energy'):
        """
        Generate energy bin indices from energy data and bins.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        energy_bins_key : str
            The key corresponding to the energy bins.
        energy_key : str, optional
            The key corresponding to the energy data (default is 'energy').
        """
        # energy=getattr(run,energy_key)
        # bins=getattr(run,energy_bins_key)
        # run.energy_bin_indices=np.digitize(energy, bins)
        run.energy_bins = energy_bins
        energy = getattr(run, energy_key)
        energy_bins_centered, run.energy_bin_indices = self.center_binning(energy, run.energy_bins)
        run.energy_bin_indices = run.energy_bin_indices - 1
        
        # run.update_status('Generated energy bins from %f to %f in %d steps.' % (np.min(bins),np.max(bins),len(bins)))
        run.update_status('Generated energy bins.')

class vonHamos:
    def __init__(self):
        pass

    def dspacing_cubic(self, a, h, k, l):
        d = a/(np.sqrt(h**2 + k**2 + l**2))
        return d

    def dspacing_hexagonal(self, a, c, h, k, l):
        d = np.sqrt(1/((4/3)*((h**2 + h*k + k**2)/(a**2)) + (l**2)/(c**2)))
        return d
               
    def dspacing(self, crystal, h, k, l):
        if crystal == 'Si':
            a = 5.430986 # Angstrom
            d = self.dspacing_cubic(a, h, k, l)
        elif crystal == 'Ge':
            a = 5.65774 # Angstrom
            d = self.dspacing_cubic(a, h, k, l)
        elif crystal == 'LiNbO3':
            a = 5.148 # Angstrom
            c = 13.863 # Angstrom
            d = self.dspacing_hexagonal(a, c, h, k, l)
        return d

    def bragg2eV(self, bragg_angle, dspacing):
        conversion_factor = 12398.419 # eV - Angstrom
        energy = conversion_factor/(2*dspacing*np.sin(np.deg2rad(bragg_angle))) # eV
        return energy

    def eV2bragg(self, energy, dspacing):
        conversion_factor = 12398.419 # eV - Angstrom
        bragg_angle = np.rad2deg(np.arcsin(conversion_factor/(energy*2*dspacing)))
        return bragg_angle

    def vH_energy_axis(self, avg_detector_distance, spectrum_length, crystal, h, k, l, crystal_radius, pixel_width = 0.05):
        conversion_factor = 12398.419 # eV - Angstrom
        n_pix = np.arange(spectrum_length) # pixel index
        d_rel = n_pix*pixel_width - (np.max(n_pix*pixel_width) - np.min(n_pix*pixel_width))/2 # relative distance from center of detector
        dspacing = self.dspacing(crystal, h, k, l)
        energy = conversion_factor/(2*dspacing*np.sin(np.arctan((2*crystal_radius)/(d_rel + avg_detector_distance))))
        return energy

class SpectrumDerivativeAnalyzer(vonHamos):
    def __init__(self, xtal_dict, y_data, smooth_window=11, poly_order=3, ref_energy = None):
        """
        Interactive spectrum analyzer for finding derivative zero crossings.
        
        Parameters:
        -----------
        x_data : array-like
            X-values of the spectrum
        y_data : array-like
            Y-values of the spectrum
        smooth_window : int
            Window length for Savitzky-Golay smoothing (must be odd)
        poly_order : int
            Polynomial order for Savitzky-Golay smoothing
        """

        self.crystal = xtal_dict['crystal']
        self.h = xtal_dict['h']
        self.k = xtal_dict['k']
        self.l = xtal_dict['l']
        self.d_space = self.dspacing(self.crystal, self.h, self.k, self.l)
        
        self.crystal_radius = xtal_dict['radius']
        detector_distance = xtal_dict['detector_distance']*2

        # conversion_factor = 12398.419 # eV - Angstrom
        conversion_factor = 12398 # eV - Angstrom
        n_pix = np.arange(y_data.shape[0]) # pixel index
        pixel_width = 0.05
        self.d_rel = n_pix*pixel_width - (np.max(n_pix*pixel_width) - np.min(n_pix*pixel_width))/2 # relative distance from center of detector
        x_data = conversion_factor/(2*self.d_space*np.sin(np.arctan((2*self.crystal_radius)/(self.d_rel + detector_distance))))

        if ref_energy is not None:
            self.ref_energy = ref_energy
        else:
            self.ref_energy = None
        
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.smooth_window = smooth_window
        self.poly_order = poly_order
        
        # Store selected region
        self.x_min = None
        self.x_max = None
        
        self.zero_crossing = None
        
        # Setup the figure
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the interactive plot"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Interactive Spectrum Derivative Analyzer', fontsize=14)
        
        # Plot original spectrum
        self.line_spectrum, = self.ax1.plot(self.x_data, self.y_data, 'k-', label='Spectrum')
        if self.ref_energy is not None:
            self.ax1.axvline(self.ref_energy, color = 'r', linestyle = '--', label = 'Ref Energy: %.2f' % (self.ref_energy))
        self.ax1.set_xlim([np.nanmin(self.x_data), np.nanmax(self.x_data)])
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_title('Original Spectrum (Click and Drag to Select Region)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Setup derivative plot
        self.ax2.set_xlim([np.nanmin(self.x_data), np.nanmax(self.x_data)])
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('dY/dX')
        self.ax2.set_title('Derivative (Zero Crossings)')
        self.ax2.grid(True, alpha=0.3)
        
        # Add span selector for region selection
        self.span = SpanSelector(
            self.ax1,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        # Initialize plot elements
        self.line_derivative = None
        self.line_zero = None
        self.scatter_zero = None
        self.vline_spectrum = None
        
        # Info text widget
        self.info_output = widgets.Output()
        
        plt.tight_layout()
        
    def on_select(self, xmin, xmax):
        """Callback for region selection"""
        self.x_min = xmin
        self.x_max = xmax
        
        # Calculate derivative and find zero crossing
        self.calculate_derivative()
        self.find_zero_crossing()
        self.update_plot()
        if self.zero_crossing is not None:
            self.correct_energy_to_ref(self.ref_energy)
        self.display_info()
        
    def calculate_derivative(self):
        """Calculate derivative in selected region"""
        # Get indices for selected region
        mask = (self.x_data >= self.x_min) & (self.x_data <= self.x_max)
        self.x_selected = self.x_data[mask]
        self.y_selected = self.y_data[mask]
        
        if len(self.x_selected) < self.smooth_window:
            self.derivative = np.gradient(self.y_selected, self.x_selected)
        else:
            # Smooth the data using Savitzky-Golay filter
            y_smooth = savgol_filter(self.y_selected, 
                                    min(self.smooth_window, len(self.y_selected) - 1 if len(self.y_selected) % 2 == 0 else len(self.y_selected)),
                                    self.poly_order)
            # Calculate derivative
            self.derivative = np.gradient(y_smooth, self.x_selected)
    
    def find_zero_crossing(self):
        """Find zero crossing point in derivative"""
        self.zero_crossing = None
        self.zero_crossing_y = None
        
        # Find sign changes
        sign_changes = np.where(np.diff(np.sign(self.derivative)))[0]
        
        if len(sign_changes) > 0:
            # Use the first zero crossing (you can modify this logic)
            idx = sign_changes[0]
            
            # Linear interpolation for more accurate zero crossing
            x1, x2 = self.x_selected[idx], self.x_selected[idx + 1]
            y1, y2 = self.derivative[idx], self.derivative[idx + 1]
            
            # Find x where derivative crosses zero
            self.zero_crossing = x1 - y1 * (x2 - x1) / (y2 - y1)
            
            # Get corresponding y-value from original spectrum
            interp_func = interp1d(self.x_selected, self.y_selected, kind='cubic')
            self.zero_crossing_y = interp_func(self.zero_crossing)
    
    def update_plot(self):
        """Update the plots with derivative and zero crossing"""
        # Clear previous derivative plot
        self.ax2.clear()
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('dY/dX')
        self.ax2.set_title('Derivative (Zero Crossings)')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot derivative
        self.ax2.plot(self.x_selected, self.derivative, 'k-', label='Derivative')
        self.ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot zero crossing on derivative
        if self.zero_crossing is not None:
            deriv_interp = interp1d(self.x_selected, self.derivative, kind='linear')
            self.ax2.scatter([self.zero_crossing], [0], 
                           color='red', s=100, zorder=5, 
                           label=f'Zero Crossing: x={self.zero_crossing:.4f}')
            
            # Add vertical line on spectrum plot
            if self.vline_spectrum is not None:
                self.vline_spectrum.remove()
            self.vline_spectrum = self.ax1.axvline(x=self.zero_crossing, 
                                                   color='red', 
                                                   linestyle='--', 
                                                   linewidth=2,
                                                   label=f'Zero at x={self.zero_crossing:.4f}')
            
            # Add point on spectrum
            self.ax1.scatter([self.zero_crossing], [self.zero_crossing_y], 
                           color='red', s=100, zorder=5)
        
        self.ax2.legend()
        self.ax1.legend()
        self.fig.canvas.draw_idle()
    
    def display_info(self):
        """Display information about the analysis"""
        with self.info_output:
            self.info_output.clear_output(wait=True)
            print("="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            print(f"Selected Region: [{self.x_min:.4f}, {self.x_max:.4f}]")
            print(f"Number of points: {len(self.x_selected)}")
            
            if self.zero_crossing is not None:
                print(f"\n🎯 Zero Crossing Found!")
                print(f"   X-value: {self.zero_crossing:.6f}")
                print(f"   Y-value: {self.zero_crossing_y:.6f}")
                print(f"   Corrected detector distance: {self.corrected_distance:.6f}")
            else:
                print("\n❌ No zero crossing found in selected region")
            print("="*50)

    def correct_energy_to_ref(self, reference_energy):
        self.zero_crossing_indx = np.argmin(np.abs(self.x_data - self.zero_crossing))
        # conversion_factor = 12398.419 # eV - Angstrom
        conversion_factor = 12398 # eV - Angstrom
        self.corrected_distance = self.crystal_radius/np.tan(np.arcsin(conversion_factor/(reference_energy*2*self.d_space))) - self.d_rel[self.zero_crossing_indx]/2
        self.corrected_energy_axis = self.vH_energy_axis(self.corrected_distance*2, self.y_data.shape[0], self.crystal, self.h, self.k, self.l, self.crystal_radius, pixel_width = 0.05)

        return self.corrected_distance, self.corrected_energy_axis
    
    def show(self):
        """Display the interactive widget"""
        display(self.info_output)
        plt.show()

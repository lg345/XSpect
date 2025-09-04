import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import  rotate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit,minimize
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
                data = fh[key][self.start_index:self.end_index, :, :]

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
                    else:
                        # Handle each ROI separately, storing the results as different attributes
                        for idx, roi in enumerate(rois):
                            start_col, end_col = roi
                            roi_data = data[:, :, start_col:end_col]
                            setattr(self, f"{name}_ROI_{idx+1}", roi_data)

                setattr(self, name, data)

                if transpose:
                    setattr(self, name, np.transpose(data, axes=(1, 2)))

            except KeyError as e:
                self.update_status(f'Key does not exist: {e.args[0]}')
            except MemoryError:
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
        bins = np.unique(vals)
        addon = (bins[-1] - bins[-2])/2 # add on energy 
        bins2 = np.append(bins,bins[-1]+addon) # elist2 will be elist with dummy value at end
        bins_center = np.empty_like(bins2)
        for ii in np.arange(bins.shape[0]):
            if ii == 0:
                bins_center[ii] = bins2[ii] - (bins2[ii+1] - bins2[ii])/2
            else:
                bins_center[ii] = bins2[ii] - (bins2[ii] - bins2[ii-1])/2
        bins_center[-1] = bins2[-1]

        setattr(run,'scanvar_indices',np.digitize(vals,bins_center))
        setattr(run,'scanvar_bins',bins_center)
    
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
        filtered_shot_mask=shot_mask * (filter_mask>threshold)* (~nan_mask)
        count_after=np.sum(filtered_shot_mask)
        setattr(run,shot_mask_key,filtered_shot_mask)
        run.update_status('Mask: %s has been filtered on %s by minimum threshold: %0.3f\nShots removed: %d' % (shot_mask_key,filter_key,threshold,count_before-count_after))
    
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
                reduced_data = reduction_function(data_chunk, **kwargs)
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
        delays = np.array(getattr(run,lxt_key)).flatten() + np.array(getattr(run,fast_delay_key)).flatten()  + np.array(getattr(run,tt_correction_key)).flatten()
        run.delays=delays
        run.time_bins=bins
        run.timing_bin_indices=np.digitize(run.delays, bins)[:]
        run.update_status('Generated timing bins from %f to %f in %d steps.' % (np.min(bins),np.max(bins),len(bins)))
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
        run.update_status('Shots combined for detector %s on filters: %s and %s into %s'%(detector_key, filter_keys[0],filter_keys[1],target_key))
        
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
        run.update_status('Shots (%d) separated for detector %s on filters: %s and %s into %s'%(np.sum(mask),detector_key,filter_keys[0],filter_keys[1],detector_key + '_' + '_'.join(filter_keys)))
    
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
        expected_length = len(run.time_bins)+1
        if len(detector.shape) < 2:
            reduced_array = np.zeros((expected_length))
        elif len(detector.shape) < 3:
            reduced_array = np.zeros((expected_length, detector.shape[1]))
        elif len(detector.shape) == 3:
            reduced_array = np.zeros((expected_length, detector.shape[1], detector.shape[2]))

        counts = np.bincount(indices)
        if average:
            np.add.at(reduced_array, indices, detector)
            reduced_array /= counts[:, None]
        else:
            np.add.at(reduced_array, indices, detector)
        setattr(run, detector_key+'_time_binned', reduced_array)
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
    def normalize_xes(self,run,detector_key,pixel_range=[300,550]):
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
        row_sum = np.sum(detector[:, pixel_range[0]:pixel_range[1]], axis=1)
        normed_main = np.divide(detector, row_sum[:,np.newaxis])
        setattr(run, detector_key+'_normalized', normed_main)
    def make_energy_axis(self, run,energy_axis_length, A, R,  mm_per_pixel=0.05, d=0.895):
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
        
        setattr(run,self.xes_line+'_energy',xaxis[::-1])
        run.update_status('XES energy axis generated for %s'%(self.xes_line))

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

        # Iterate over the images and accumulate them into reduced_array based on timing_indices
        for i in range(detector.shape[0]):
            np.add.at(reduced_array, (scanvar_indices[i],), detector[i])

        # Store the reduced_array in the object, replace 'key_name' with the actual key
        setattr(run,  f"{detector_key}_scanvar_reduced", reduced_array)

        # Update status
        run.update_status(f'Detector binned in time into key: {detector_key}_scanvar_reduced')

class XASAnalysis(SpectroscopyAnalysis):
    def __init__(self):
        pass;
    def trim_ccm(self,run,threshold=120):
        """
        Trim CCM values to remove bins with fewer shots than a specified threshold.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        threshold : int, optional
            The minimum number of shots required to keep a CCM value (default is 120).
        """
        
        ccm_bins=getattr(run,'ccm_bins',elist_center)
        ccm_energies=getattr(run,'ccm_energies',elist)
        counts = np.bincount(bins)
        trimmed_ccm=ccm_energies[counts[:-1]>120]
        self.make_ccm_axis(run,ccm_energies)
        
    def make_ccm_axis(self,run,energies):
        """
        Generate CCM bins and centers from given energy values.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        energies : array-like
            Array of energy values to be used for creating CCM bins.
        """
        elist=energies
#         addon = (elist[-1] - elist[-2])/2 # add on energy 
#         elist2 = np.append(elist,elist[-1]+addon) # elist2 will be elist with dummy value at end
#         elist_center = np.empty_like(elist2)
#         for ii in np.arange(elist.shape[0]):
#             if ii == 0:
#                 elist_center[ii] = elist2[ii] - (elist2[ii+1] - elist2[ii])/2
#             else:
#                 elist_center[ii] = elist2[ii] - (elist2[ii] - elist2[ii-1])/2
#                 elist_center[-1] = elist2[-1]
        addon = (elist[-1] - elist[-2])/2
        elist2 = np.append(elist,elist[-1]+addon)
        elist_center = np.empty_like(elist)

        for ii in np.arange(elist_center.shape[0]):
            if ii == elist_center.shape[0]:
                elist_center[ii] = elist[-1]+addon
            else:
                elist_center[ii] = elist2[ii+1] - (elist2[ii+1] - elist2[ii])/2    
    
        setattr(run,'ccm_bins',elist_center)
        setattr(run,'ccm_energies',elist)
    def reduce_detector_ccm_temporal(self, run, detector_key, timing_bin_key_indices,ccm_bin_key_indices,average=True):
        """
        Reduce detector data temporally and by CCM bins.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        timing_bin_key_indices : str
            The key corresponding to the timing bin indices.
        ccm_bin_key_indices : str
            The key corresponding to the CCM bin indices.
        average : bool, optional
            Whether to average the reduced data (default is True).
        """
        detector = getattr(run, detector_key)
        timing_indices = getattr(run, timing_bin_key_indices)#digitized indices from detector
        ccm_indices = getattr(run, ccm_bin_key_indices)#digitized indices from detector
        reduced_array = np.zeros((np.shape(run.time_bins)[0]+1, np.shape(run.ccm_bins)[0]))
        unique_indices =np.column_stack((timing_indices, ccm_indices))
        np.add.at(reduced_array, (unique_indices[:, 0], unique_indices[:, 1]), detector)
        reduced_array = reduced_array[:-1,:]
        setattr(run, detector_key+'_time_energy_binned', reduced_array)
        run.update_status('Detector %s binned in time into key: %s'%(detector_key,detector_key+'_time_energy_binned') )
        
    def reduce_detector_ccm(self, run, detector_key, ccm_bin_key_indices, average = False, not_ccm=False):
        """
        Reduce detector data by CCM bins.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        detector_key : str
            The key corresponding to the detector data.
        ccm_bin_key_indices : str
            The key corresponding to the CCM bin indices.
        average : bool, optional
            Whether to average the reduced data (default is False).
        not_ccm : bool, optional
            Whether to indicate that CCM is not being used (default is False).

        """
        detector = getattr(run, detector_key)
        
        ccm_indices = getattr(run, ccm_bin_key_indices)#digitized indices from detector
        if not_ccm:
            reduced_array = np.zeros(np.max(ccm_indices)+1 )
        else:
            reduced_array = np.zeros(np.shape(run.ccm_bins)[0]) 
        np.add.at(reduced_array, ccm_indices, detector)
        setattr(run, detector_key+'_energy_binned', reduced_array)
        
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
        timing_indices = getattr(run, timing_bin_key_indices)#digitized indices from detector
        reduced_array = np.zeros(np.shape(time_bins)[0]+1)
        np.add.at(reduced_array, timing_indices, detector)
        setattr(run, detector_key+'_time_binned', reduced_array)
        run.update_status('Detector %s binned in time into key: %s'%(detector_key,detector_key+'_time_binned') )
        
    def ccm_binning(self,run,ccm_bins_key,ccm_key='ccm'):
        """
        Generate CCM bin indices from CCM data and bins.

        Parameters
        ----------
        run : object
            The spectroscopy run instance.
        ccm_bins_key : str
            The key corresponding to the CCM bins.
        ccm_key : str, optional
            The key corresponding to the CCM data (default is 'ccm').
        """
        ccm=getattr(run,ccm_key)
        bins=getattr(run,ccm_bins_key)
        run.ccm_bin_indices=np.digitize(ccm, bins)
        run.update_status('Generated ccm bins from %f to %f in %d steps.' % (np.min(bins),np.max(bins),len(bins)))

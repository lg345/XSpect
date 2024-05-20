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
        self.lcls_run = lcls_run
        self.hutch = hutch
        self.experiment_id = experiment_id
        self.get_experiment_directory()
    def get_experiment_directory(self):    
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def add_detector(self, detector_name, detector_dimensions):
        self.detector_name = detector_name
        self.detector_dimensions = detector_dimensions

class spectroscopy_run:
    def __init__(self,spec_experiment,run,verbose=False,end_index=-1,start_index=0):
        self.spec_experiment=spec_experiment
        self.run_number=run
        self.run_file='%s/%s_Run%04d.h5' % (self.spec_experiment.experiment_directory, self.spec_experiment.experiment_id, self.run_number)
        self.status=['New analysis of run %d located in: %s' % (self.run_number,self.run_file)]
        self.status_datetime=[datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        self.verbose=verbose
        self.end_index=end_index
        self.start_index=start_index

    def get_scan_val(self):
        with h5py.File(self.run_file, 'r') as fh:
            self.scan_var=fh['scan/scan_variable']
            
        
    def update_status(self,update,):
        self.status.append(update)
        self.status_datetime.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.verbose:
            print(update)

    def get_run_shot_properties(self):
        
        with h5py.File(self.run_file, 'r') as fh:
            self.total_shots = fh['lightStatus/xray'][self.start_index:self.end_index].shape[0]
            xray_total = np.sum(fh['lightStatus/xray'][self.start_index:self.end_index])
            laser_total = np.sum(fh['lightStatus/laser'][self.start_index:self.end_index])
            self.xray = np.array(fh['lightStatus/xray'][self.start_index:self.end_index])
            self.laser = np.array(fh['lightStatus/laser'][self.start_index:self.end_index])
            self.simultaneous=np.logical_and(self.xray,self.laser)
            
        self.run_shots={'Total':self.total_shots,'X-ray Total':xray_total,'Laser Total':laser_total}
        self.update_status('Obtained shot properties')
    
    def load_run_keys(self, keys, friendly_names):
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
    def load_run_key_delayed(self, keys, friendly_names):
        start=time.time()
        fh= h5py.File(self.run_file, 'r')
        for key, name in zip(keys, friendly_names):
            try:
                setattr(self, name, fh[key][self.start_index:self.end_index,:,:])
            except KeyError as e:
                self.update_status('Key does not exist: %s' % e.args[0])
            except MemoryError:
                setattr(self, name, fh[key][self.start_index:self.end_index,:,:])
                self.update_status('Out of memory error while loading key: %s. Not converted to np.array.' % key)
        end=time.time()
        self.update_status('HDF5 import of keys completed kept as hdf5 dataset. Time: %.02f seconds' % (end-start))
        self.h5=fh
    def load_sum_run_scattering(self,key):
        with h5py.File(self.run_file, 'r') as fh:
            setattr(self, 'scattering', np.nansum(np.nansum(fh[key][:,:,20:80],axis=1),axis=1))
        
    def close_h5(self):
        self.h5.close()
        del self.h5
    
    def purge_all_keys(self,keys_to_keep):
        #all_attributes = list(self.__dict__.keys())
        # Remove attributes that are not in the specified list
        #for attribute in all_attributes:
        #    if attribute not in keys_to_keep:
        #        delattr(self, attribute)
        #        self.update_status(f"Purged key to save room: {attribute}")
                
        new_dict = {attr: value for attr, value in self.__dict__.items() if attr in keys_to_keep}
        # Assign the new dictionary to __dict__
        self.__dict__ = new_dict
        
class SpectroscopyAnalysis:
    def __init__(self):
        pass
    
    def bin_uniques(self,run,key):
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
        #filter_mode a is all shots. l is laser+x-ray shots.
        shot_mask=getattr(run,shot_mask_key)
        count_before=np.sum(shot_mask)
        filter_mask=getattr(run,filter_key)
        nan_mask = np.isnan(filter_mask)
        filtered_shot_mask=shot_mask * (filter_mask>threshold)* (~nan_mask)
        count_after=np.sum(filtered_shot_mask)
        setattr(run,shot_mask_key,filtered_shot_mask)
        run.update_status('Mask: %s has been filtered on %s by minimum threshold: %0.3f\nShots removed: %d' % (shot_mask_key,filter_key,threshold,count_before-count_after))
    
    def filter_nan(self, run,shot_mask_key, filter_key='ipm'):
        #filter_mode a is all shots. l is laser+x-ray shots.
        shot_mask=getattr(run,shot_mask_key)
        count_before=np.sum(shot_mask)
        filter_mask=getattr(run,filter_key)
        filtered_shot_mask=shot_mask * (filter_mask>threshold)
        count_after=np.sum(filtered_shot_mask)
        setattr(run,shot_mask_key,filtered_shot_mask)
        run.update_status('Mask: %s has been filtered on %s by minimum threshold: %0.3f\nShots removed: %d' % (shot_mask_key,filter_key,threshold,count_before-count_after))

    
    def filter_detector_adu(self,run,detector,adu_threshold=3.0):
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
        for detector_key in keys:
            setattr(run, detector_key, None)
            run.update_status(f"Purged key to save room: {detector_key}")
            
    
    def reduce_detector_spatial(self, run, detector_key, shot_range=[0, None], rois=[[0, None]], reduction_function=np.sum,  purge=True, combine=True):
        detector = getattr(run, detector_key)
        if combine:
            
            roi_combined = [rois[0][0], rois[-1][1]]  # Combined ROI spanning the first and last ROI
            mask = np.zeros(detector.shape[2], dtype=bool)
            for roi in rois:
                mask[roi[0]:roi[1]] = True
            masked_data = detector[shot_range[0]:shot_range[1], :, :][:, :, mask]
            reduced_data = reduction_function(masked_data, axis=2)
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
            run.update_status(f"Purged key to save room: {detector_key}")

    def time_binning(self,run,bins,lxt_key='lxt_ttc',fast_delay_key='encoder',tt_correction_key='time_tool_correction'):
        run.delays = getattr(run,lxt_key)*1.0e12 + getattr(run,fast_delay_key)  + getattr(run,tt_correction_key)
        run.time_bins=bins
        run.timing_bin_indices=np.digitize(run.delays, bins)[:]
        run.update_status('Generated timing bins from %f to %f in %d steps.' % (np.min(bins),np.max(bins),len(bins)))
    def union_shots(self, run, detector_key, filter_keys):
        detector = getattr(run, detector_key)
        
        if isinstance(filter_keys, list):
            mask = np.logical_and.reduce([getattr(run, k) for k in filter_keys])
        else:
            mask = getattr(run, filter_keys)
        filtered_detector = detector[mask]
        setattr(run, detector_key + '_' + '_'.join(filter_keys), filtered_detector)
        run.update_status('Shots combined for detector %s on filters: %s and %s into %s'%(detector_key, filter_keys[0],filter_keys[1],detector_key + '_' + '_'.join(filter_keys)))
        
    def separate_shots(self, run, detector_key, filter_keys):
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
        detector = getattr(run, detector_key)
        indices = getattr(run, timing_bin_key_indices)
        #print(detector.shape)
        expected_length = len(run.time_bins)+1

        if len(detector.shape) < 3:
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
        run.update_status('Detector %s binned in time into key: %s'%(detector_key,detector_key+'_time_binned') )
    def patch_pixels(self,run,detector_key,  mode='average', patch_range=4, deg=1, poly_range=6,axis=1):
        for pixel in self.pixels_to_patch:
            self.patch_pixel(run,detector_key,pixel,mode,patch_range,deg,poly_range,axis=axis)
    def patch_pixel(self, run, detector_key, pixel, mode='average', patch_range=4, deg=1, poly_range=6,axis=1):
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
            neighbor_values = data[:, pixel - patch_range:pixel + patch_range + 1, :]
            new_val=np.sum(neighbor_values, axis=1) / neighbor_values.shape[1]

            if axis==1:
                data[:, pixel, :] = new_val
            elif axis==2:
                data[:,:,pixel]=new_val
        elif mode == 'polynomial':
            patch_x = np.arange(pixel - patch_range - poly_range, pixel + patch_range + poly_range + 1, 1)
            patch_range_weights = np.ones(len(patch_x))
            patch_range_weights[pixel - patch_range - poly_range:pixel + patch_range + poly_range] = 0.001
            coeffs = np.polyfit(patch_x, data[pixel - patch_range - poly_range:pixel + patch_range + poly_range + 1, :], deg,
                                w=patch_range_weights)
            data[pixel, :] = np.polyval(coeffs, pixel)
        elif mode == 'interpolate':
            patch_x = np.arange(pixel - patch_range - poly_range, pixel + patch_range + poly_range + 1, 1)
            interp = interp1d(patch_x, data[pixel - patch_range - poly_range:pixel + patch_range + poly_range + 1, :],
                              kind='quadratic')
            data[pixel, :] = interp(pixel)
        setattr(run,detector_key,data)
        run.update_status('Detector %s pixel %d patched. Old value.'%(detector_key, pixel ))
    def patch_pixels_1d(self,run,detector_key,  mode='average', patch_range=4, deg=1, poly_range=6):
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
        run.update_status('Detector %s pixel %d patched.'%(detector_key, pixel ))
        


class XESAnalysis(SpectroscopyAnalysis):
    def __init__(self,xes_line='kbeta'):
        self.xes_line=xes_line
        pass
    def normalize_xes(self,run,detector_key,pixel_range=[300,550]):
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
    def reduce_detector_spatial(self, run, detector_key, shot_range=[0, None], rois=[[0, None]], reduction_function=np.sum,  purge=True, combine=True,adu_cutoff=3.0):
        detector = getattr(run, detector_key)
        if combine:
            
            roi_combined = [rois[0][0], rois[-1][1]]  # Combined ROI spanning the first and last ROI
            mask = np.zeros(detector.shape[2], dtype=bool)
            for roi in rois:
                mask[roi[0]:roi[1]] = True
            masked_data = detector[shot_range[0]:shot_range[1], :, :][:, :, mask]
            #masked_data = masked_data * (masked_data > adu_cutoff)
            #Note the adu filtering should be handled at the controller level
            reduced_data = reduction_function(masked_data, axis=2)
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
            run.update_status(f"Purged key to save room: {detector_key}")

class XASAnalysis(SpectroscopyAnalysis):
    def __init__(self):
        pass
    def make_ccm_axis(self,run,energies):
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
        detector = getattr(run, detector_key)
        timing_indices = getattr(run, timing_bin_key_indices)#digitized indices from detector
        ccm_indices = getattr(run, ccm_bin_key_indices)#digitized indices from detector
        reduced_array = np.zeros((np.max(timing_indices)+1, np.max(ccm_indices) + 1))
#         reduced_array = np.zeros((run.time_bins.shape[0], run.ccm_bins.shape[0]))
        #for idx,i in enumerate(detector):
        #    reduced_array[timing_indices[idx],ccm_indices[idx]]=detector[idx]+reduced_array[timing_indices[idx],ccm_indices[idx]]
        unique_indices =np.column_stack((timing_indices, ccm_indices))
        np.add.at(reduced_array, (unique_indices[:, 0], unique_indices[:, 1]), detector)
        reduced_array = reduced_array[:-1,:]
        setattr(run, detector_key+'_time_energy_binned', reduced_array)
        
        run.update_status('Detector %s binned in time into key: %s'%(detector_key,detector_key+'_time_energy_binned') )
        
    def reduce_detector_ccm(self, run, detector_key, ccm_bin_key_indices, average = False):
        detector = getattr(run, detector_key)
        ccm_indices = getattr(run, ccm_bin_key_indices)#digitized indices from detector
        reduced_array = np.zeros(np.max(ccm_indices) + 1)
        np.add.at(reduced_array, ccm_indices, detector)
        setattr(run, detector_key+'_energy_binned', reduced_array)
        
        run.update_status('Detector %s binned in energy into key: %s'%(detector_key,detector_key+'_energy_binned') )
        
    def reduce_detector_temporal(self, run, detector_key, timing_bin_key_indices, average=False):
        detector = getattr(run, detector_key)
        time_bins=run.time_bins
        timing_indices = getattr(run, timing_bin_key_indices)#digitized indices from detector
        reduced_array = np.zeros(np.shape(time_bins)[0]+1)
        np.add.at(reduced_array, timing_indices, detector)
        reduced_array = reduced_array[:-1]
        setattr(run, detector_key+'_time_binned', reduced_array)
        
        run.update_status('Detector %s binned in time into key: %s'%(detector_key,detector_key+'_time_binned') )
        
    def ccm_binning(self,run,ccm_bins_key,ccm_key='ccm'):
        ccm=getattr(run,ccm_key)
        bins=getattr(run,ccm_bins_key)
        run.ccm_bin_indices=np.digitize(ccm, bins)
        run.update_status('Generated ccm bins from %f to %f in %d steps.' % (np.min(bins),np.max(bins),len(bins)))

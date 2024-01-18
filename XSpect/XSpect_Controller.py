import h5py
import psana
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
from XSpect.XSpect_Analysis import *
from XSpect.XSpect_Analysis import spectroscopy_run
class BatchAnalysis:
    def __init__():
        pass
    def run_parser(self,run_array):
        """
        Parses the run input array. This is useful if the user wants to give multiple ranges of runs to analyzer  e.g. '4-6 8 10-12'.
        Parameters
        ----------
        run_array : list of str
            The string array of runs coming directly from argparse. Here we will convert to a single string then parse the string. 

        Returns
        -------
        list of int
            List of individual run values.
        """
        run_string = ' '.join(run_array)
        runs = []
        for run_range in run_string.split():
            if '-' in run_range:
                start, end = map(int, run_range.split('-'))
                runs.extend(range(start, end+1))
            else:
                try:
                    runs.append(int(run_range))
                except ValueError:
                    raise ValueError(f"Invalid input: {run_range}")

        self.runs=runs
        
    def add_filter(self, shot_type, filter_key,threshold):
        if shot_type not in ['xray','laser','simultaneous']:
            raise ValueError('Only options for shot type are xray, laser, or simultaneous.')
        self.filters.append({'FilterType':shot_type,'FilterKey':filter_key,'FilterThreshold':threshold})
        
    def set_key_aliases(self,keys=['tt/ttCorr','epics/lxt_ttc', 'enc/lasDelay' , 'ipm4/sum','tt/AMPL','epix_2/ROI_0_area'], 
names=['time_tool_correction','lxt_ttc'  ,'encoder','ipm', 'time_tool_ampl','epix']):
        self.keys=keys
        self.friendly_names=names
    
    def primary_analysis_loop(self,experiment,verbose=False):
        analyzed_runs=[]
        for run in self.runs:            
            analyzed_runs.append(self.primary_analysis(experiment,run,verbose))
        self.analyzed_runs=analyzed_runs
        
    def primary_analysis_parallel_loop(self,cores,experiment,verbose=False):
        pool = multiprocessing.Pool(processes=cores)
        analyzed_runs = []
        for run in self.runs:
            analyzed_run = pool.apply_async(self.primary_analysis, (experiment, run, verbose))
            analyzed_runs.append((analyzed_run, self.runs.index(run)))
        analyzed_runs = [analyzed_runs[idx][0].get() for _, idx in sorted(analyzed_runs, key=lambda x: x[1])]
        pool.close()
        self.analyzed_runs = analyzed_runs
    
    def primary_analysis(self):
        raise AttributeError('The primaray_analysis must be implemented by the child classes.')

        
class XESBatchAnalysis(BatchAnalysis):
    def __init__(self):
        self.xes_line='kbeta'
        self.pixels_to_patch=[351,352,529,530,531]
        self.crystal_d_space=50.6
        self.crystal_radius=250
        self.adu_cutoff=3.0
        self.rois=[[0,None]]
        self.mintime=-2.0
        self.maxtime=10.0
        self.numpoints=240
        self.time_bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        self.filters=[]
        self.key_epix=['epix_2/ROI_0_area']
        self.friendly_name_epix=['epix']
        self.rotation=0.0
    
    def primary_analysis(self,experiment,run,verbose=False):
        f=spectroscopy_run(experiment,run,verbose=verbose)
        f.get_run_shot_properties()
        f.load_run_keys(self.keys,self.friendly_names)
        f.load_run_key_delayed(self.key_epix,self.friendly_name_epix)
        analysis=XESAnalysis()
        
        analysis.union_shots(f,'epix_ROI_1',['simultaneous','laser'])
        analysis.separate_shots(f,'epix_ROI_1',['xray','laser'])
        self.bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        analysis.time_binning(f,self.bins)
        analysis.union_shots(f,'timing_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'timing_bin_indices',['xray','laser'])
        analysis.reduce_detector_temporal(f,'epix_ROI_1_simultaneous_laser','timing_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_temporal(f,'epix_ROI_1_xray_not_laser','timing_bin_indices_xray_not_laser',average=False)
        analysis.normalize_xes(f,'epix_ROI_1_simultaneous_laser_time_binned')
        analysis.normalize_xes(f,'epix_ROI_1_xray_not_laser_time_binned')
        
        f.epix=rotate(f.epix, angle=self.angle, axes=[1,2])
        
        analysis.pixels_to_patch=self.pixels_to_patch
        analysis.reduce_detector_spatial(f,'epix', rois=self.rois,adu_cutoff=self.adu_cutoff)
        #analysis.patch_pixels_1d(f,'epix_ROI_1')
        f.close_h5()
        analysis.make_energy_axis(f,f.epix_ROI_1.shape[1],self.crystal_d_space,self.crystal_radius)
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold'])                                                                      
        
        return f
    
class XESBatchAnalysisRotation(XESBatchAnalysis):
    def primary_analysis(self,experiment,run,verbose=False):
        f=spectroscopy_run(experiment,run,verbose=verbose)
        f.get_run_shot_properties()
        f.load_run_keys(self.keys,self.friendly_names)
        f.load_run_key_delayed(self.key_epix,self.friendly_name_epix)
        analysis=XESAnalysis()
        analysis.pixels_to_patch=self.pixels_to_patch
#        analysis.reduce_detector_spatial(f,'epix', rois=self.rois,adu_cutoff=self.adu_cutoff)
        #f.close_h5()
        #analysis.make_energy_axis(f,f.epix_ROI_1.shape[1],self.crystal_d_space,self.crystal_radius)
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold'])                                                                  
        analysis.union_shots(f,'epix',['simultaneous','laser'])
        analysis.separate_shots(f,'epix',['xray','laser'])
        self.bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        analysis.time_binning(f,self.bins)
        analysis.union_shots(f,'timing_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'timing_bin_indices',['xray','laser'])
        analysis.reduce_detector_temporal(f,'epix_simultaneous_laser','timing_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_temporal(f,'epix_xray_not_laser','timing_bin_indices_xray_not_laser',average=False)
        
        f.close_h5()
        return f

        
    

class XASBatchAnalysis(BatchAnalysis):
    def __init__(self):
        self.mintime=-2.0
        self.maxtime=10.0
        self.numpoints=240
        self.minccm=7.105
        self.maxccm=7.135
        self.numpoints_ccm=90
        self.time_bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        self.filters=[]
    def primary_analysis(self,experiment,run,verbose=False):
        f=spectroscopy_run(experiment,run,verbose=verbose)
        f.get_run_shot_properties()
        
        f.load_run_keys(self.keys,self.friendly_names)
        analysis=XASAnalysis()
        elist = np.linspace(self.minccm,self.maxccm,self.numpoints_ccm)
        analysis.make_ccm_axis(f,elist)
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold']) 
        analysis.union_shots(f,'epix',['simultaneous','laser'])
        analysis.separate_shots(f,'epix',['xray','laser'])
        analysis.union_shots(f,'ipm',['simultaneous','laser'])
        analysis.separate_shots(f,'ipm',['xray','laser'])
        analysis.union_shots(f,'ccm',['simultaneous','laser'])
        analysis.separate_shots(f,'ccm',['xray','laser'])
        self.time_bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        analysis.time_binning(f,self.time_bins)
        analysis.ccm_binning(f,'ccm_bins','ccm')
        analysis.union_shots(f,'timing_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'timing_bin_indices',['xray','laser'])
        analysis.union_shots(f,'ccm_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'ccm_bin_indices',['xray','laser'])
        analysis.reduce_detector_ccm_temporal(f,'epix_simultaneous_laser','timing_bin_indices_simultaneous_laser','ccm_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_ccm_temporal(f,'epix_xray_not_laser','timing_bin_indices_xray_not_laser','ccm_bin_indices_xray_not_laser',average=False)
        analysis.reduce_detector_ccm_temporal(f,'ipm_simultaneous_laser','timing_bin_indices_simultaneous_laser','ccm_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_ccm_temporal(f,'ipm_xray_not_laser','timing_bin_indices_xray_not_laser','ccm_bin_indices_xray_not_laser',average=False)
        return f

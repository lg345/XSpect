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
from XSpect.XSpect_Controller import *
class SpectroscopyVisualization:                
    def __init__(self):
        pass
    def plot_2d_spectrum(self,run,detector_key):
        spectrum= getattr(run, detector_key)
        vmin, vmax = np.percentile(spectrum, [1,99])
        fig,ax=plt.subplots(1,1)
        im=ax.imshow(spectrum, vmin=vmin, vmax=vmax, origin='lower',aspect='auto')
    
    def plot_2d_difference_spectrum(self,run,detector_keys):
        spectrum_1= getattr(run, detector_keys[0])
        spectrum_2= getattr(run, detector_keys[1])
        spectrum=spectrum_1-spectrum_2
        vmin, vmax = np.percentile(spectrum, [1,99])
        fig,ax=plt.subplots(1,1)
        im=ax.imshow(spectrum, vmin=vmin, vmax=vmax, origin='lower',aspect='auto')
        run.difference_spectrum=spectrum
        
class XESVisualization(SpectroscopyVisualization):
    def __init__(self):
        pass
    def plot_1d_XES(self, run, detector_key, target_key, low=-np.inf,high=np.inf,axis=0):
        target=getattr(run,target_key)
        intensities=getattr(run,detector_key)
        if hasattr(run,run.xes_line+'_energy'):        
            idxlow=np.argmin(np.abs(target-low))
            idxhigh=np.argmin(np.abs(target-high))
            cut=np.nanmean(intensities[idxlow:idxhigh,:],axis=axis)
            plt.plot(getattr(run,run.xes_line+'_energy'),cut)
            run.current_cut=cut
        else:
            raise ValueError('There is no energy axis in this object')
    
    def combine_spectra(self,xes_analysis,xes_key,xes_laser_key):
        xes=getattr(xes_analysis.analyzed_runs[0],xes_key)
        xes_laser=getattr(xes_analysis.analyzed_runs[0],xes_laser_key)
        summed_laser_off=np.zeros_like(xes)
        summed_laser_on=np.zeros_like(xes_laser)
        for run in xes_analysis.analyzed_runs:
            summed_laser_on+=getattr(run,xes_laser_key)
            summed_laser_off+=getattr(run,xes_key)
        self.summed_laser_on=summed_laser_on
        self.summed_laser_off=summed_laser_off
        analysis=XESAnalysis()
        analysis.normalize_xes(self,'summed_laser_on')
        analysis.normalize_xes(self,'summed_laser_off')
        xes_analysis.summed_laser_on_normalized=self.summed_laser_on_normalized
        xes_analysis.summed_laser_off_normalized=self.summed_laser_off_normalized

    def plot_2d_difference_spectrum(self,xes_analysis):
        laser_on_spectrum=xes_analysis.summed_laser_on_normalized
        laser_off_spectrum=xes_analysis.summed_laser_off_normalized
        difference_spectrum=laser_on_spectrum-laser_off_spectrum
        energy=xes_analysis.analyzed_runs[0].kbeta_energy
        vmin, vmax = np.percentile(difference_spectrum, [0,99])
        plt.figure(dpi=100)
        plt.imshow(difference_spectrum.T, cmap='RdBu', vmin=-0.005, vmax=0.005, origin='lower',aspect='auto',extent=[xes_analysis.mintime,xes_analysis.maxtime,np.min(energy),np.max(energy)])
        plt.colorbar()
        plt.xlabel('Time (ps)')
        plt.ylabel('Energy (keV)')
        setattr(xes_analysis,'difference_spectrum',difference_spectrum)
        
class XASVisualization(SpectroscopyVisualization):
    def __init__(self):
        pass
    def plot_XAS(self,run,detector_key,ccm_key):
        det=getattr(run,detector_key)
        ccm=getattr(run,ccm_key)
        plt.plot(ccm,det)
    
    def combine_spectra(self,xas_analysis,xas_laser_key,xas_key,norm_laser_key,norm_key):
        xas=getattr(xas_analysis.analyzed_runs[0],xas_key)
        xas_laser=getattr(xas_analysis.analyzed_runs[0],xas_laser_key)
        norm=getattr(xas_analysis.analyzed_runs[0],norm_key)
        norm_laser=getattr(xas_analysis.analyzed_runs[0],norm_laser_key)
        summed_laser_on=np.zeros_like(xas_laser)
        summed_laser_off=np.zeros_like(xas)
        summed_norm_on=np.zeros_like(norm_laser)
        summed_norm_off=np.zeros_like(norm)
        for run in xas_analysis.analyzed_runs:
            summed_laser_on+=getattr(run,xas_laser_key)
            summed_laser_off+=getattr(run,xas_key)
            summed_norm_on+=getattr(run,norm_laser_key)
            summed_norm_off+=getattr(run,norm_key)
        xas_analysis.summed_laser_on=summed_laser_on
        xas_analysis.summed_laser_off=summed_laser_off
        xas_analysis.summed_norm_on=summed_norm_on
        xas_analysis.summed_norm_off=summed_norm_off
    
    def plot_2d_difference_spectrum(self,xas_analysis):
        laser_on_spectrum=xas_analysis.summed_laser_on/xas_analysis.summed_norm_on
        laser_off_spectrum=np.divide(np.nansum(xas_analysis.summed_laser_off,axis=0),np.nansum(xas_analysis.summed_norm_off,axis=0))
        difference_spectrum=laser_on_spectrum-laser_off_spectrum
        vmin, vmax = np.percentile(difference_spectrum, [0,99])
        plt.figure(dpi=100)
        plt.imshow(difference_spectrum.T, cmap='RdBu', vmin=-0.5, vmax=0.5, origin='lower',aspect='auto',extent=[xas_analysis.mintime,xas_analysis.maxtime,xas_analysis.minccm,xas_analysis.maxccm])
        plt.colorbar()
        plt.xlabel('Time (ps)')
        plt.ylabel('Energy (keV)')
        setattr(xas_analysis,'difference_spectrum',difference_spectrum)
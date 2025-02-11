import h5py
#import psana
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
from scipy.interpolate import interp2d

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

    def make_energy_axis(self, A, R,  mm_per_pixel=0.05, d=0.895):
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
        gl = np.arange(np.shape(self.summed_xes)[0], dtype=np.float64)
        gl *= pix
        ll = gl / 2 - (np.amax(gl) - np.amin(gl)) / 4
        factor = 1.2398e4
        xaxis = factor / (2.0 * d * np.sin(np.arctan(R / (ll + A))))
        
        self.energy=xaxis[:]
        
class XESVisualization(SpectroscopyVisualization):
    def __init__(self):
        self.vmin=-0.1
        self.vmax=0.1
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

    def combine_static_spectra(self,xes_analysis,xes_key):
        xes=getattr(xes_analysis.analyzed_runs[0],xes_key)     
        summed_laser_off=np.zeros_like(xes)       
        for run in xes_analysis.analyzed_runs: 
            current_xes=getattr(run,xes_key)
            if not np.isnan(current_xes).any():
                summed_laser_off+=current_xes
        self.summed_xes=summed_laser_off


           
    def plot_2d_difference_spectrum(self,xes_analysis):
        laser_on_spectrum=xes_analysis.summed_laser_on_normalized
        laser_off_spectrum=xes_analysis.summed_laser_off_normalized
        difference_spectrum=laser_on_spectrum-laser_off_spectrum
        try:
            energy=xes_analysis.analyzed_runs[0].kbeta_energy
        except:
            energy=np.linspace(0,np.shape(laser_on_spectrum),1)
        #vmin, vmax = np.percentile(difference_spectrum, [0,99])
        plt.figure(dpi=100)
        plt.imshow(difference_spectrum.T, cmap='RdBu', vmin=self.vmin, vmax=self.vmax, origin='lower',aspect='auto',extent=[xes_analysis.mintime,xes_analysis.maxtime,energy[0],energy[-1]])
        plt.colorbar()
        plt.xlabel('Time (ps)')
        plt.ylabel('Energy (keV)')
        setattr(xes_analysis,'difference_spectrum',difference_spectrum)

    def normalize_spectrum(self, low, high):
        """
        Normalize the spectrum (x, y) to unity based on the specified range [low, high].

        Parameters:
        x (np.ndarray): Energy values.
        y (np.ndarray): Intensity values.
        low (float): Lower bound of the energy range for normalization.
        high (float): Upper bound of the energy range for normalization.
        
        Returns:
        np.ndarray: Normalized intensity values.
        """
        y=self.background_subtracted
        x=self.energy
        mask = (x >= low) & (x <= high)
        area = np.trapz(y[mask], x[mask])
        if area == 0:
            raise ValueError("The area for normalization is zero, normalization cannot be performed.")
        normalized_y = y / area
        setattr(self,'normalized',normalized_y)
    
    def normalize_peak(self, low, high):
        """
        Normalize the spectrum (x, y) to unity based on the specified range [low, high].

        Parameters:
        x (np.ndarray): Energy values.
        y (np.ndarray): Intensity values.
        low (float): Lower bound of the energy range for normalization.
        high (float): Upper bound of the energy range for normalization.
        
        Returns:
        np.ndarray: Normalized intensity values.
        """
        y=self.background_subtracted
        x=self.energy
        mask = (x >= low) & (x <= high)
        y_peak = np.max(y[mask])
        if y_peak == 0:
            raise ValueError("The peak for normalization is zero, normalization cannot be performed.")
        normalized_peak = y / y_peak
        setattr(self,'normalized_peak',normalized_peak)
        
class XASVisualization(SpectroscopyVisualization):
    def __init__(self):
        self.vmin=-0.1
        self.vmax=0.1
        pass
    def plot_XAS(self,run,detector_key,ccm_key):
        det=getattr(run,detector_key)
        ccm=getattr(run,ccm_key)
        plt.plot(ccm,det)
    


    def combine_spectra(self, xas_analysis, xas_laser_key, xas_key, norm_laser_key, norm_key, interpolate=False):
        xas = getattr(xas_analysis.analyzed_runs[0], xas_key)
        xas_laser = getattr(xas_analysis.analyzed_runs[0], xas_laser_key)
        norm = getattr(xas_analysis.analyzed_runs[0], norm_key)
        norm_laser = getattr(xas_analysis.analyzed_runs[0], norm_laser_key)

        try:
            ccm_bins = getattr(xas_analysis.analyzed_runs[0], 'ccm_energies')
            setattr(xas_analysis, 'ccm_bins', ccm_bins)
        except:
            pass

        summed_laser_on = np.zeros_like(xas_laser)
        summed_laser_off = np.zeros_like(xas)
        summed_norm_on = np.zeros_like(norm_laser)
        summed_norm_off = np.zeros_like(norm)

        for idx, run in enumerate(xas_analysis.analyzed_runs):
            if idx > 0 and interpolate:
                ccm = getattr(run, 'ccm_energies')
                time = np.arange(xas_laser.shape[0])  # assuming time axis is the first dimension

                interp_laser_on = interp2d(ccm, time, getattr(run, xas_laser_key), fill_value=0, bounds_error=False)
                interp_laser_off = interp2d(ccm, time, getattr(run, xas_key), fill_value=0, bounds_error=False)
                interp_norm_on = interp2d(ccm, time, getattr(run, norm_laser_key), fill_value=0, bounds_error=False)
                interp_norm_off = interp2d(ccm, time, getattr(run, norm_key), fill_value=0, bounds_error=False)

                for t in time:
                    summed_laser_on[t, :] += interp_laser_on(ccm_bins, t)
                    summed_laser_off[t, :] += interp_laser_off(ccm_bins, t)
                    summed_norm_on[t, :] += interp_norm_on(ccm_bins, t)
                    summed_norm_off[t, :] += interp_norm_off(ccm_bins, t)
            else:
                summed_laser_on += np.array(getattr(run, xas_laser_key))
                summed_laser_off += np.array(getattr(run, xas_key))
                summed_norm_on += np.array(getattr(run, norm_laser_key))
                summed_norm_off += np.array(getattr(run, norm_key))

        xas_analysis.summed_laser_on = summed_laser_on
        xas_analysis.summed_laser_off = summed_laser_off
        xas_analysis.summed_norm_on = summed_norm_on
        xas_analysis.summed_norm_off = summed_norm_off


    def plot_2d_difference_spectrum(self,xas_analysis,vmin=None,vmax=None):
        laser_on_spectrum=xas_analysis.summed_laser_on/xas_analysis.summed_norm_on
        laser_off_spectrum=np.divide(np.nansum(xas_analysis.summed_laser_off,axis=0),np.nansum(xas_analysis.summed_norm_off,axis=0))
        difference_spectrum=laser_on_spectrum-laser_off_spectrum
        setattr(xas_analysis,'difference_spectrum',difference_spectrum)
#         vmin, vmax = np.nanpercentile(difference_spectrum, [0,99])
        plt.figure(dpi=100)
#         plt.imshow(difference_spectrum.T, cmap='RdBu', vmin=self.vmin, vmax=self.vmax, origin='lower',aspect='auto',extent=[xas_analysis.mintime,xas_analysis.maxtime,xas_analysis.minccm,xas_analysis.maxccm])
        if (vmin==None and vmax==None):
            vmax = np.nanmax(np.abs(difference_spectrum))
            vmin = -vmax
        contlevels = np.linspace(vmin*0.5, vmax*0.5, 20)
        
        plt.contourf(xas_analysis.ccm_bins, xas_analysis.time_bins, difference_spectrum, contlevels, cmap = 'RdBu',extend="max")
        plt.colorbar()
        plt.xlabel('Energy (keV)')
        plt.ylabel('Time (ps)')
        
#     def plot_2d_difference_spectrum_real_axes(self,xas_analysis):
#         laser_on_spectrum=xas_analysis.summed_laser_on/xas_analysis.summed_norm_on
#         laser_off_spectrum=np.divide(np.nansum(xas_analysis.summed_laser_off,axis=0),np.nansum(xas_analysis.summed_norm_off,axis=0))
#         difference_spectrum=laser_on_spectrum-laser_off_spectrum
#         vmin, vmax = np.percentile(difference_spectrum, [0,99])
#         plt.figure(dpi=100)
#         plt.imshow(difference_spectrum.T, cmap='RdBu', vmin=self.vmin, vmax=self.vmax, origin='lower',aspect='auto',extent=[xas_analysis.mintime,xas_analysis.maxtime,xas_analysis.minccm,xas_analysis.maxccm])
#         plt.contourf(
#         plt.colorbar()
#         plt.xlabel('Time (ps)')
#         plt.ylabel('Energy (keV)')
#         setattr(xas_analysis,'difference_spectrum',difference_spectrum)

    def plot_1d_difference_time(self, xas_analysis):
        laser_on = xas_analysis.summed_laser_on/xas_analysis.summed_norm_on
        laser_off = xas_analysis.summed_laser_off/xas_analysis.summed_norm_off
        
        difference = laser_on - laser_off
        setattr(xas_analysis, 'difference_trace', difference)
        plt.figure(dpi=100)
        plt.plot(xas_analysis.time_bins, difference)
        plt.xlabel('Time (ps)')
        plt.ylabel(r'$\Delta XAS$')
        
    def plot_1d_difference_spectrum(self, xas_analysis):
        laser_on = xas_analysis.summed_laser_on/xas_analysis.summed_norm_on
        laser_off = xas_analysis.summed_laser_off/xas_analysis.summed_norm_off
        
        difference = laser_on - laser_off
        setattr(xas_analysis, 'difference_trace', difference)
        plt.figure(dpi=100)
        plt.plot(xas_analysis.ccm_bins, difference)
        plt.xlabel('Energy (keV)')
        plt.ylabel(r'$\Delta XAS$')

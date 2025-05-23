import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from scipy.ndimage import rotate
import inspect

width = 1.5
length = 5

plt.rcParams['figure.figsize'] = (7.5, 4.5)
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.major.width'] = width
plt.rcParams['ytick.major.width'] = width
plt.rcParams['xtick.major.size'] = length
plt.rcParams['ytick.major.size'] = length
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

class utils:
    def __inti__(self):
        pass

    def object_inspector(data_object, verbose=False):
        if verbose==True:
            print("------ ATTRIBUTE LIST ------")
            for x in dir(data_object):
                print(x)
            print("----- METHODS -----")
            for method in inspect.getmembers(data_object, predicate=inspect.ismethod):
                print(method)
            print("----- ATTRIBUTES -----")
            for key, value in vars(data_object).items():
                print(key, ":", value)
        if verbose==False:
            print("----- METHODS -----")
            for method in inspect.getmembers(data_object, predicate=inspect.ismethod):
                print(method)
            print("----- ATTRIBUTES -----")
            for key, value in vars(data_object).items():
                print(key, ":", value)

class plotting:
    def __init__(self):
        pass
    
    def hplot(self, data, thresholds, plt_title, leg_title, xlabel, yscale):
        
        thres = np.array(thresholds)
        
        xmin = 1.2*np.nanmin(data)
        xmax = 1.2*np.nanmax(data)
        binstep = (xmax - xmin)/500
        
        fig, ax = plt.subplots(ncols = 1)
        
        ax.hist(data, bins = np.arange(xmin, xmax, binstep))
        
        if thres.size == 1:
            ax.axvline(thres, color = np.array([255, 0, 0])/255, linewidth = 1.5, label = thres)
        elif thres.size > 1:
            
            for ii in range(thres.size):
                ax.axvline(thres[ii], color = np.array([255, 0, 0])/255, linewidth = 1.5, label = thres[ii])
                
        ax.set_title(plt_title, fontsize = 14, fontweight = 'bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Counts')
        ax.legend(title = leg_title)
        ax.set_yscale(yscale)
        
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(width)
        fig.tight_layout()
        plt.show()
    
    def roiview(self, data, thres, plt_type, energy_dispersive_axis = 'horiz'):
        
        cl = (np.nanpercentile(data, 1), np.nanpercentile(data, 99))
        
        if plt_type == 'xes':
            fig, ax = plt.subplots(ncols = 1, nrows = 2, figsize = (8,8))
            p1 = ax[0].imshow(data, clim = cl)
            ax[0].set_title('XES ROI', fontsize = 14, fontweight = 'bold')
            for lim in thres:
                if energy_dispersive_axis == 'horiz' or energy_dispersive_axis == 'horizontal':
                    if thres[lim]:
                        roisum = np.nansum(data[thres[lim][0]:thres[lim][1],:], axis = 1)
                        p2 = ax[1].plot(roisum, linewidth = 1.5, label = lim)
                        for ii in range(len(thres[lim])):
                            ax[0].axhline(thres[lim][ii], color = 'red', linewidth = 1.5, label = lim + ': {}'.format(thres[lim][ii]))
                else:
                    if thres[lim]:
                        roisum = np.nansum(data[:,thres[lim][0]:thres[lim][1]], axis = 1)
                        p2 = ax[1].plot(roisum, linewidth = 1.5, label = lim)
                        for ii in range(len(thres[lim])):
                            ax[0].axvline(thres[lim][ii], color = 'red', linewidth = 1.5, label = lim + ': {}'.format(thres[lim][ii]))
            ax[1].set_title('ROI Projections', fontsize = 14, fontweight = 'bold')
            ax[1].set_xlabel('Pixel')
            ax[1].set_ylabel('Summed Intensity')
            ax[1].set_xlim([0, data.shape[0]])
            ax[1].legend()
            cb = fig.colorbar(p1, ax = ax[0])
            
            for plot in ax:
                for axis in ['top', 'bottom', 'left', 'right']:
                    plot.spines[axis].set_linewidth(width)
                        
        elif plt_type == 'xas':
            fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (4,4))
            p1 = ax.imshow(data, clim = cl)
            ax.set_title('XAS ROI', fontsize = 14, fontweight = 'bold')
            for lim in thres:
                if thres[lim] and lim == 'horiz':
                    for ii in range(len(thres[lim])):
                        ax.axvline(thres[lim][ii], color = 'red', linewidth = 1.5, label = lim + ': {}'.format(thres[lim][ii]))
                elif thres[lim] and lim == 'vert':
                    for ii in range(len(thres[lim])):
                        ax.axhline(thres[lim][ii], color = 'red', linewidth = 1.5, label = lim + ': {}'.format(thres[lim][ii]))
            cb = fig.colorbar(p1, ax = ax)
            
            for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(width)
                
        #cb.ax.get_children()[7].set_linewidth(width) # cb.ax.get_children()[7] is colorbar spine ##JB this was returning an error
        fig.tight_layout()
        plt.show()
        
class diagnostics(plotting):
    def __init__(self, run, exp, keys, friendly_names):
        
        ## when generating a diagnostics object, run no., exp no., keys, and friendly names are passed to the object
        
        self.run = run
        self.exp = exp
        self.keys = keys
        self.friendly_names = friendly_names
        
        #fpath = '/sdf/data/lcls/ds/{}/{}/hdf5/smalldata/'.format(self.exp[:3], self.exp)
        #f = fpath + '{}_Run{:04d}.h5'.format(self.exp, self.run)
        self.filepath = '/sdf/data/lcls/ds/{}/{}/hdf5/smalldata/'.format(self.exp[:3], self.exp) + '{}_Run{:04d}.h5'.format(self.exp, self.run)
        self.h5 = h5py.File(self.filepath)
        print('Run {} imported'.format(self.run))
        
        ## generate a dictionary for the supplied keys/friendly names
        
        self.datadict = {}
        for key, name in zip(keys, friendly_names):
            self.datadict[name] = key

        ## create list of all group/datasets keys

        self.allgroupsdatasetskeys = []
        def getnames(item):
            self.allgroupsdatasetskeys.append(item)
        self.h5.visit(getnames)    
            
        
    def load_run_keys(self, metadata = False):
        
        ## reads keys from h5 file and stores as arrays in self with the friendly key name
        
        try:
            getattr(self, 'h5')
        except AttributeError:
            print('Error: must run importruns() function first')
            return
        
        if self.h5.__bool__() == False:
            print("The h5 file has closed since initializing the diagnostic object. Attempting to read file.")
            self.h5 = h5py.File(self.filepath)
            print("h5 file open/closed bool state is: " + str(self.h5.__bool__()))
            
        self.meta_data = []
        meta_data_header = [["type", "shape", "memory size (GB)"], ["nan count", "max", "min", "mean"]]
        self.meta_data.append({"header": meta_data_header})
        with self.h5 as fh:
            for key, name in zip(self.keys, self.friendly_names):
                try:
                    setattr(self, name, fh[key][:])
                    if metadata == True:
                        print("meta data set to true, loading meta data")
                        datatype = fh[key].dtype 
                        shape = fh[key].shape
                        sizeGB = fh[key].nbytes / 1e9
                        nancnt = np.count_nonzero(np.isnan(fh[key]))
                        max_value = np.max(np.nan_to_num(fh[key]))
                        min_value = np.min(np.nan_to_num(fh[key]))
                        mean_value = np.mean(np.nan_to_num(fh[key]))
                        meta_data = [[datatype, shape, sizeGB], [nancnt, max_value, min_value, mean_value]]
                        self.meta_data.append({key: meta_data})
                except KeyError as e:
                    print('Key does not exist: %s' % e, '\n  ---> Please check self.alldatasetnames list for all dataset keys')
        print("Finished loading keys, h5 file open/closed bool state is: " + str(self.h5.__bool__()))
                
        
    def adu_histogram(self, nshots, thresholds, ROIopt = False, energy_dispersive_axis = 'vert'):
        
        ## generate linearized array of pixel intensities (ADU) over first nshots of events and plot histogram
        
        pt = 'Pixel Intensity on ePix100 Run {} ({} shots)'.format(self.run, nshots)
        lt = 'ADU Thresholds'
        xl = 'Pixel Intensity (keV)'
        ys = 'log'
        
        if ROIopt:
            if hasattr(self, 'xes_roi_limits'):
                print('Calculating histograms for XES ROIs:')
                thres = self.xes_roi_limits
                for lim in thres:
                    if thres[lim]:
                        if energy_dispersive_axis == 'horiz' or energy_dispersive_axis == 'horizontal':
                            data2plot = self.h5[self.datadict['epix']][0:nshots,thres[lim][0]:thres[lim][1],:].ravel()
                        else:
                            data2plot = self.h5[self.datadict['epix']][0:nshots,:,thres[lim][0]:thres[lim][1]].ravel()
                            
                        pt = 'Pixel Intensity in {} ROI ({} shots)'.format(lim, nshots)
                        self.hplot(data2plot, thresholds, pt, lt, xl, ys)
                        
            elif hasattr(self, 'xas_roi_limits'):
                print('Calculating histograms for XAS ROI:')
                thres = self.xas_roi_limits
                data2plot = self.h5[self.datadict['epix']][0:nshots,thres['vert'][0]:thres['vert'][1],thres['horiz'][0]:thres['horiz'][1]].ravel()
                pt = 'Pixel Intensity in XAS ROI ({} shots)'.format(nshots)
                self.hplot(data2plot, thresholds, pt, lt, xl, ys)
                
            else:
                print('Error: no ROIs have been set, run function xes_ROI or xas_ROI with option setrois = True)')
                return
            
        else:
            data2plot = self.h5[self.datadict['epix']][0:nshots,:,:].ravel()
            pt = 'Pixel Intensity on ePix100 Run {} ({} shots)'.format(self.run, nshots) 
            self.hplot(data2plot, thresholds, pt, lt, xl, ys)
        
    def ipm_histogram(self, thresholds):
        
        ## plot histogram of chosen IPM values (measure of incoming X-ray intensity) over all events
        
        data2plot = self.h5[self.datadict['ipm']][:]
        pt = '{} Histogram'.format(self.datadict['ipm'])
        lt = 'IPM Filters'
        xl = self.datadict['ipm']
        ys = 'linear'
       
        self.hplot(data2plot, thresholds, pt, lt, xl, ys)
        
        
    def ttAMPL_histogram(self, thresholds):
        
        ## plot histogram of time tool amplitude over all events
    
        data2plot = self.h5[self.datadict['time_tool_ampl']][:]
        pt = '{} Histogram'.format(self.datadict['time_tool_ampl'])
        lt = 'ttAMPL Filters'
        xl = self.datadict['time_tool_ampl']
        ys = 'log'
       
        self.hplot(data2plot, thresholds, pt, lt, xl, ys)
        
    def xes_ROI(self, nshots, kb_limits = [], ka_limits = [], setrois = False, energy_dispersive_axis = 'vert', angle=0):
        
        ## plots summed spectroscopy detector image over first nshots events as well as any ROI limits provided
        
        roi_limits = {}
        roi_limits['Ka'] = ka_limits
        roi_limits['Kb'] = kb_limits
        
        if setrois:
            setattr(self, 'xes_roi_limits', roi_limits)
        
        data2plot = np.nansum(self.h5[self.datadict['epix']][0:nshots,:,:], axis = 0)
        data2plot_rot = rotate(data2plot, angle, axes=[0,1])
        
        ptype = 'xes'
        
        self.roiview(data2plot, roi_limits, ptype, energy_dispersive_axis = energy_dispersive_axis)
        
    def xas_ROI(self, nshots, horiz_limits = [], vert_limits = [], setrois = False):
    
        ## plots summed spectroscopy detector image over first nshots events as well as any ROI limits provided
        
        roi_limits = {}
        roi_limits['horiz'] = horiz_limits
        roi_limits['vert'] = vert_limits
        
        if setrois:
            setattr(self, 'xas_roi_limits', roi_limits)
        
        data2plot = np.nansum(self.h5[self.datadict['epix']][0:nshots,:,:], axis = 0)
        
        ptype = 'xas'
        
        self.roiview(data2plot, roi_limits, ptype)

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
import XSpect.XSpect_Analysis
import XSpect.XSpect_Controller
import XSpect.XSpect_Visualization

job_id = os.environ.get('SLURM_JOB_ID')
os.mkdir(str(job_id))
xes_experiment = XSpect.XSpect_Analysis.spectroscopy_experiment(hutch='xcs',experiment_id='xcsp23820',lcls_run=21)
xes=XSpect.XSpect_Controller.XESBatchAnalysisRotation()
xes.end_index=-1
#8000#2000#only import the first 5000 shots
xes.mintime=-2
xes.maxtime=10
xes.numpoints=240
xes.adu_cutoff=[30,9909]#3.0
xes.angle=1.3
xes.pixels_to_patch=[225,350,351,352,353,354,355,356,436,437]
xes.key_epix=['epix_1/ROI_0_area']
xes.set_key_aliases(keys=['tt/ttCorr','epics/lxt_ttc', 'enc/lasDelay' , 'ipm4/sum','tt/AMPL','epix_1/ROI_0_area'], names=['time_tool_correction','lxt_ttc'  ,'encoder','ipm', 'time_tool_ampl','epix'])
xes.rois=[[0,-1]]#[[22,30]]
#xes.add_filter('xray','ipm4',1.0E3)
#xes.add_filter('simultaneous','ipm4',1.0E3)
#xes.add_filter('simultaneous','time_tool_ampl',0.15)
xes.run_parser(['162-181 183-190 193-208 211-224'])

start=time.time()
#xes.primary_analysis_parallel_loop(16,xes_experiment)
xes.primary_analysis_parallel_range(16,xes_experiment,increment=2000)
end=time.time()

np.shape(xes.analyzed_runs[0].epix_simultaneous_laser_time_binned_ROI_1)
v=XSpect.XSpect_Visualization.XESVisualization()
v.combine_spectra(xes_analysis=xes,xes_key='epix_xray_not_laser_time_binned_ROI_1',xes_laser_key='epix_simultaneous_laser_time_binned_ROI_1')
v.vmin=-0.001
v.vmax=0.001

os.chdir(job_id)
np.save('xray_norm.np',xes.summed_laser_on_normalized)
np.save('laser_norm.np',xes.summed_laser_off_normalized)
with open('XSpect.log', 'a') as f:
    for r in xes.analyzed_runs:
        for j,k in zip(r.status_datetime,r.status):
            f.write((j,k))
        f.write('\n')
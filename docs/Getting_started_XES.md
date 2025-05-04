``` py
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
import XSpect.XSpect_Analysis
import XSpect.XSpect_Controller
import XSpect.XSpect_Visualization
```

## Static XES Spectra

``` py
xes_experiment = XSpect.XSpect_Analysis.spectroscopy_experiment(hutch='mfx',experiment_id='mfxx1013623',lcls_run=23)
xes=XSpect.XSpect_Controller.XESBatchAnalysisRotation()
xes.key_epix=['epix_1/ROI_0_area']
#xes.set_key_aliases(keys,names)
xes.import_roi=[[523,535]]
xes.rois=[[0,12]]
xes.adu_cutoff=3.0
xes.angle=0
xes.transpose=True
xes.run_parser(['37'])
start=time.time()
xes.primary_analysis_parallel_range(4,xes_experiment,method=xes.primary_analysis_static,increment=2000,verbose=False)
end=time.time()
v=XSpect.XSpect_Visualization.XESVisualization()
v.combine_static_spectra(xes_analysis=xes,xes_key='epix_ROI_1')
plt.plot(v.summed_xes)
```

    Processing: 100%|██████████| 17/17 [00:43<00:00,  2.55s/Shot_Batch]

    [<matplotlib.lines.Line2D at 0x7f0d8767e820>]

![](media/77615fc07aea654fbf2201cddc214b1008b2608d.png)

## 2D Time-resolved XES Spectra
``` py
xes_experiment = XSpect.XSpect_Analysis.spectroscopy_experiment(hutch='mfx',experiment_id='mfxl1027922',lcls_run=22)
xes=XSpect.XSpect_Controller.XESBatchAnalysisRotation()
keys=['tt/ttCorr','epics/lxt', 'enc/lasDelay' , 'ipm4/sum','tt/AMPL'] 
names=['time_tool_correction','lxt_ttc'  ,'encoder','ipm', 'time_tool_ampl']
#Here we define the epix detector keys separately as they are imported separately to avoid OOM
xes.key_epix=[r'epix_2/ROI_0_area']
xes.friendly_name_epix=['epix']
##
xes.set_key_aliases(keys,names)
#xes.end_index=5000
xes.mintime=-0.9
xes.maxtime=0.9
xes.numpoints=40
xes.time_bins=np.linspace(xes.mintime,xes.maxtime,xes.numpoints)
xes.rois=[[0,50]]
xes.adu_cutoff=3.0
xes.angle=90
xes.lxt_key=None
xes.transpose=True
#xes.add_filter('xray','ipm4',1.0E3)
#xes.add_filter('simultaneous','ipm4',1.0E3)
xes.add_filter('simultaneous','time_tool_ampl',0.05)
xes.run_parser(['44-46'])
```

``` py
start=time.time()
xes.primary_analysis_parallel_range(8,xes_experiment,increment=1000,verbose=False)
end=time.time()
```

    Processing: 100%|██████████| 30/30 [02:25<00:00,  4.85s/Shot_Batch]

``` py
xes.status
```

    ['Setting key aliases.',
     'Adding filter: Shot Type=simultaneous, Filter Key=time_tool_ampl, Threshold=0.05',
     'Parsing run array.',
     'Starting parallel analysis with shot ranges.',
     'Parsing run shots.',
     'Run shots parsed.',
     'Breaking into shot ranges with increment 1000.',
     'Shot ranges broken.',
     'Parallel analysis with shot ranges completed.',
     'Parallel analysis completed.',
     'Total time: 145.59 seconds.',
     'Parallel time (processing): 145.59 seconds.',
     'Time per batch (on average): 4.85 seconds.',
     'Time per core (on average): 18.20 seconds.',
     'Batches per core (on average): 3.75.',
     'Read bytes: 13.14 MB.',
     'Write bytes: 4055.88 MB.',
     'Memory used: 52.87 MB.']

``` py
v=XSpect.XSpect_Visualization.XESVisualization()
v.combine_spectra(xes_analysis=xes,xes_key='epix_xray_not_laser_time_binned_ROI_1',xes_laser_key='epix_simultaneous_laser_time_binned_ROI_1')
v.vmin=-0.006
v.vmax=0.004
v.plot_2d_difference_spectrum(xes)
plt.xlim(-0.8,0.8)
```

    (-0.8, 0.8)

![](media/f22364f0824ce6092523ca08e6a277dd8abc3f46.png)

``` py
```

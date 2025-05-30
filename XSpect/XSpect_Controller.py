import h5py
#import psana
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
import multiprocessing
import os
from functools import partial
import time
import sys
from datetime import datetime
import argparse
from XSpect.XSpect_Analysis import *
from XSpect.XSpect_Analysis import spectroscopy_run
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import psutil
from collections import defaultdict


class BatchAnalysis:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.status = []
        self.status_datetime = []
        self.filters = []
        self.keys = []
        self.friendly_names = []
        self.runs = []
        self.run_shots = {}
        self.run_shot_ranges = {}
        self.analyzed_runs = []
        
        self.xes_line='kbeta'
        self.pixels_to_patch=[351,352,529,530,531]
        self.crystal_detector_distance=50.6
        self.crystal_d_space=0.895
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
        self.angle=0.0
        self.end_index=None
        self.start_index=0
        self.transpose=False
        self.lxt_key='lxt_ttc'
        self.import_roi=None
        self.keys_to_save=['start_index','end_index','run_file','run_number','verbose','status','status_datetime','epix_ROI_1_summed','epix_summed']
        self.patch_mode='average'
        self.arbitrary_filter=False
        self.hitfind=False
        
    def aggregate_statistics(self):
        aggregated_stats = defaultdict(lambda: defaultdict(int))
        
        for run in self.analyzed_runs:
            run_number = run.run_number
            run_shots = run.run_shots
            
            for key, value in run_shots.items():
                aggregated_stats[run_number][key] += value

        # Calculate Percent_XES_Hits after aggregation
        for run_number, stats in aggregated_stats.items():
            total = stats.get('Total', 1)
            xes_hits = stats.get('XES_Hits', 0)
            stats['Percent_XES_Hits'] = (xes_hits / total) * 100
        
        aggregated_stats = {run_number: dict(stats) for run_number, stats in aggregated_stats.items()}
        
        setattr(self, 'run_statistics', dict(aggregated_stats))

    def print_run_statistics(self):
        for run_number, stats in self.run_statistics.items():
            print(f"Run Number: {run_number}")
            for key, value in stats.items():
                if key == 'Percent_XES_Hits':
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value}")
            print()  # Add a newline for better readability
            
    def update_status(self, update):
        self.status.append(update)
        self.status_datetime.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.verbose:
            print(update)

    def run_parser(self, run_array):
        self.update_status("Parsing run array.")
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

        self.runs = runs

    def add_filter(self, shot_type, filter_key, threshold):
        self.update_status(f"Adding filter: Shot Type={shot_type}, Filter Key={filter_key}, Threshold={threshold}")
        if shot_type not in ['xray', 'laser', 'simultaneous']:
            raise ValueError('Only options for shot type are xray, laser, or simultaneous.')
        self.filters.append({'FilterType': shot_type, 'FilterKey': filter_key, 'FilterThreshold': threshold})

    def set_key_aliases(self, keys=['tt/ttCorr', 'epics/lxt_ttc', 'enc/lasDelay', 'ipm4/sum', 'tt/AMPL', 'epix_2/ROI_0_area'],
                        names=['time_tool_correction', 'lxt_ttc', 'encoder', 'ipm', 'time_tool_ampl', 'epix']):
        self.update_status("Setting key aliases.")
        if 'epix' in names:
            warnings.warn('If you plan on using delayed key loading for the epix then please define key_epix and friendly_name_epix. And do not load it here for risk of OOM')
        self.friendly_name_epix=['epix']
        self.keys = keys
        self.friendly_names = names

    def primary_analysis_loop(self, experiment, verbose=False):
        self.update_status(f"Starting primary analysis loop with experiment={experiment}, verbose={verbose}.")
        analyzed_runs = []
        for run in self.runs:
            analyzed_runs.append(self.primary_analysis(experiment, run, verbose))
        self.analyzed_runs = analyzed_runs
        self.update_status("Primary analysis loop completed.")

    def primary_analysis_parallel_loop(self, cores, experiment, verbose=False):
        self.update_status(f"Starting parallel analysis loop with cores={cores}, experiment={experiment}, verbose={verbose}.")
        pool = Pool(processes=cores)
        analyzed_runs = []

        def callback(result):
            analyzed_runs.append(result)

        with tqdm(total=len(self.runs), desc="Processing Runs", unit="Run") as pbar:
            for analyzed_run in pool.imap(partial(self.primary_analysis, experiment=experiment, verbose=verbose), self.runs):
                pbar.update(1)
                analyzed_runs.append(analyzed_run)

        pool.close()
        pool.join()

        analyzed_runs = [analyzed_run for analyzed_run in sorted(analyzed_runs, key=lambda x: (x.run_number, x.end_index))]
        self.analyzed_runs = analyzed_runs
        self.update_status("Parallel analysis loop completed.")
        

    def primary_analysis(self):
        raise AttributeError('The primary_analysis must be implemented by the child classes.')

    def parse_run_shots(self, experiment, verbose=False):
        self.update_status("Parsing run shots.")
        run_shots_dict = {}
        for run in self.runs:
            f = spectroscopy_run(experiment, run, verbose=verbose, end_index=self.end_index)
            f.get_run_shot_properties()
            run_shots_dict[run] = f.total_shots
        self.run_shots = run_shots_dict
        self.update_status("Run shots parsed.")

    def break_into_shot_ranges(self, increment):
        self.update_status(f"Breaking into shot ranges with increment {increment}.")
        run_shot_ranges_dict = {}

        for run, total_shots in self.run_shots.items():
            run_shot_ranges = []
            min_index = 0

            if self.end_index is not None and self.end_index != -1:
                total_shots = min(self.end_index + 1, total_shots)  # Including final shot

            while min_index < total_shots:
                max_index = min(min_index + increment - 1, total_shots - 1)  # Calculate max_index
                run_shot_ranges.append((min_index, max_index))
                min_index = max_index + 1  # Move to the next index right after current max_index

            run_shot_ranges_dict[run] = run_shot_ranges

        self.run_shot_ranges = run_shot_ranges_dict
        self.update_status("Shot ranges broken.")

        flat_list = [(run, (shot_range[0], shot_range[1])) for run, shot_ranges in run_shot_ranges_dict.items() for shot_range in shot_ranges]
        self.run_shot_ranges = np.array(flat_list, dtype=object)



    def primary_analysis_parallel_range(self, cores, experiment, increment, start_index=None, end_index=None, verbose=False, method=None):
        self.update_status("Starting parallel analysis with shot ranges.")
        self.parse_run_shots(experiment, verbose)
        self.break_into_shot_ranges(increment)

        analyzed_runs = []
        total_runs = len(self.run_shot_ranges)

        # Start timing the overall process
        start_time = time.time()

        # Start tracking I/O stats
        io_before = psutil.disk_io_counters()
        mem_before = psutil.virtual_memory().used

        errors = []
        total_tasks = 0

        with Pool(processes=cores) as pool, tqdm(total=total_runs, desc="Processing", unit="Shot_Batch") as pbar:
            run_shot_ranges = self.run_shot_ranges

            def callback(result):
                nonlocal pbar
                if isinstance(result, dict) and "error" in result:
                    self.update_status(f"Error in processing run {result['run']}: {result['error']}")
                    errors.append(result['error'])
                else:
                    analyzed_runs.append(result)
                pbar.update(1)

            def error_callback(e):
                self.update_status(f"Parallel processing error: {str(e)}")
                errors.append(str(e))

            for run_shot in run_shot_ranges:
                run, shot_ranges = run_shot
                pool.apply_async(self.primary_analysis_range, 
                                 (experiment, run, shot_ranges, verbose, method), 
                                 callback=callback, 
                                 error_callback=error_callback)

            pool.close()
            pool.join()

        self.analyzed_runs = analyzed_runs
        analyzed_runs = [analyzed_run for analyzed_run in sorted(analyzed_runs, key=lambda x: (x.run_number, x.end_index))]
        self.analyzed_runs = analyzed_runs
        self.update_status("Parallel analysis with shot ranges completed.")

        # End timing the parallel section
        parallel_end_time = time.time()

        # End timing the overall process
        end_time = time.time()

        # Calculate I/O statistics
        io_after = psutil.disk_io_counters()
        read_bytes = io_after.read_bytes - io_before.read_bytes
        write_bytes = io_after.write_bytes - io_before.write_bytes

        mem_after = psutil.virtual_memory().used
        memory_used = mem_after - mem_before

        # Calculate useful statistics
        total_time = end_time - start_time
        parallel_time = parallel_end_time - start_time
        time_per_run = parallel_time / total_runs if total_runs > 0 else 0
        time_per_core = parallel_time / cores if cores > 0 else 0
        runs_per_core = total_runs / cores if cores > 0 else 0

        # Update status with statistics
        self.update_status(f"Parallel analysis completed.")
        self.update_status(f"Total time: {total_time:.2f} seconds.")
        self.update_status(f"Parallel time (processing): {parallel_time:.2f} seconds.")
        self.update_status(f"Time per batch (on average): {time_per_run:.2f} seconds.")
        self.update_status(f"Time per core (on average): {time_per_core:.2f} seconds.")
        self.update_status(f"Batches per core (on average): {runs_per_core:.2f}.")
        self.update_status(f"Read bytes: {read_bytes / (1024 ** 2):.2f} MB.")
        self.update_status(f"Write bytes: {write_bytes / (1024 ** 2):.2f} MB.")
        self.update_status(f"Memory used: {memory_used / (1024 ** 2):.2f} MB.")

        if errors:
            self.update_status(f"Errors encountered: {len(errors)}")






def analyze_single_run(args):
    obj,experiment, run, shot_ranges, verbose = args
    return obj.primary_analysis_range(expFeriment, run, shot_ranges, verbose)



class XESBatchAnalysis(BatchAnalysis):
    def __init__(self):
        super().__init__()

 
    
    def primary_analysis(self,experiment,run,verbose=False,start_index=None,end_index=None):
        if end_index==None:
            end_index=self.end_index
        if start_index==None:
            try:
                start_index=self.start_index
            except AttributeError:
                start_index=0
        self.time_bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        f=spectroscopy_run(experiment,run,verbose=verbose,start_index=start_index,end_index=end_index)
        f.load_run_keys(self.keys,self.friendly_names)
        f.load_run_key_delayed(self.key_epix,self.friendly_name_epix)
        f.get_run_shot_properties()
        analysis=XESAnalysis()
        analysis.reduce_detector_spatial(f,'epix', rois=self.rois,adu_cutoff=self.adu_cutoff)
        analysis.filter_detector_adu(f,'epix',adu_threshold=self.adu_cutoff)
        analysis.union_shots(f,'epix_ROI_1',['simultaneous','laser'])
        analysis.separate_shots(f,'epix_ROI_1',['xray','laser'])
        self.bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        analysis.time_binning(f,self.bins,lxt_key=self.lxt_key)
        analysis.union_shots(f,'timing_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'timing_bin_indices',['xray','laser'])
        analysis.reduce_detector_temporal(f,'epix_ROI_1_simultaneous_laser','timing_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_temporal(f,'epix_ROI_1_xray_not_laser','timing_bin_indices_xray_not_laser',average=False)
        analysis.normalize_xes(f,'epix_ROI_1_simultaneous_laser_time_binned')
        analysis.normalize_xes(f,'epix_ROI_1_xray_not_laser_time_binned')   
        if self.pixels_to_patch.any()!=None:
            analysis.pixels_to_patch=self.pixels_to_patch
        f.close_h5()
        analysis.make_energy_axis(f,f.epix_ROI_1.shape[1],A=self.crystal_detector_distance,R=self.crystal_radius,d=self.crystal_d_space)
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold'])                                                                      
        
        return f
    
class XESBatchAnalysisRotation(XESBatchAnalysis):
    def __init__(self):
        super().__init__()
    def append_arbitrary_filtering(self,xes_experiment,verbose=False,basepath='.'):
        for run in self.runs:
            f=spectroscopy_run(xes_experiment,run,verbose=verbose,start_index=0,end_index=None)
            f.run_file
            f.get_run_shot_properties()
            target_file=f'{basepath}/{run:03d}_indices.txt'
            arbfil=np.loadtxt(target_file,dtype=int)
            with h5py.File(f.run_file, 'a') as hf:
                if 'arbitrary_filter' in hf:
                    del hf['arbitrary_filter']
                insert_filter = np.zeros(f.total_shots)
                insert_filter[arbfil] = 1
                hf.create_dataset('arbitrary_filter', data=insert_filter)
                self.update_status(f'Arbitrary filter appended to hdf5 file: {f.run_file} using {target_file}')
    def primary_analysis_static_parallel_loop(self, cores, experiment, verbose=False):
        self.update_status(f"Starting parallel analysis loop with cores={cores}, experiment={experiment}, verbose={verbose}.")
        pool = Pool(processes=cores)
        analyzed_runs = []

        def callback(result):
            analyzed_runs.append(result)

        with tqdm(total=len(self.runs), desc="Processing Runs", unit="Run") as pbar:
            for analyzed_run in pool.imap(partial(self.primary_analysis_static, experiment=experiment, verbose=verbose), self.runs):
                pbar.update(1)
                analyzed_runs.append(analyzed_run)

        pool.close()
        pool.join()

        analyzed_runs = [analyzed_run for analyzed_run in sorted(analyzed_runs, key=lambda x: (x.run_number, x.end_index))]
        self.analyzed_runs = analyzed_runs
        self.update_status("Parallel analysis loop completed.")

    def primary_analysis_static(self, run, experiment, verbose=False, start_index=None,end_index=None):
        if end_index==None:
            end_index=self.end_index
        if start_index==None:
            try:
                start_index=self.start_index
            except AttributeError:
                start_index=0
        f=spectroscopy_run(experiment,run,verbose=verbose,start_index=start_index,end_index=end_index)
        f.load_run_keys(self.keys,self.friendly_names)
        f.load_run_key_delayed(self.key_epix,self.friendly_name_epix,rois=self.import_roi)
        f.get_run_shot_properties()
        analysis=XESAnalysis()
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold']) 
        if self.arbitrary_filter:
            f.set_arbitrary_filter()
            analysis.union_shots(f,'epix',['xray','arbitrary_filter'],new_key=False)
        else:
            analysis.union_shots(f,'epix',['xray','xray'],new_key=False)     
        analysis.filter_detector_adu(f,'epix',adu_threshold=self.adu_cutoff)

  
        if self.hitfind:
            from XSpect.XSpect_Processor import HitFinding
            f.update_status(f'Starting hit finding')
            hits,mean_sum,std_sum,threshold,sum_images=HitFinding.basic_detect(f.epix,cutoff_multiplier=1)
            f.update_status(f'Hit finding on epix. Hits found: {str(len(hits))}, median: {str(mean_sum)}, std: {std_sum}, threshold: {threshold}')
            f.epix=f.epix[hits]
            f.update_status(f'Applying Hits to ePix detector. New size {str(np.shape(f.epix)[0])}')
            f.run_shots['XES_Hits']=len(hits)
            f.sum_images=sum_images
        analysis.reduce_detector_shots(f,'epix',purge=False,new_key=False)
        if self.transpose:
            f.epix=np.transpose(f.epix)
        analysis.pixels_to_patch=self.pixels_to_patch     
        analysis.patch_pixels(f,'epix',axis=0,mode=self.patch_mode)
        if self.angle!=0:
            #f.epix=rotate(f.epix, angle=self.angle, axes=[1,2])
            f.epix=rotate(f.epix, angle=self.angle, axes=[0,1])
        analysis.reduce_detector_spatial(f,'epix', rois=self.rois,combine=True,purge=False)
        self.keys_to_save.extend(['start_index','end_index','run_file','run_number','verbose','status','status_datetime','epix_ROI_1','sum_images','epix','all_epix','run_shots'])

        f.purge_all_keys(self.keys_to_save)
        #analysis.make_energy_axis(f,f.epix.shape[1],d=self.crystal_d_space,R=self.crystal_radius,A=self.crystal_detector_distance)
        return f
    
    def primary_analysis_static_laser(self, run, experiment, verbose=False, start_index=None, end_index=None):
        if end_index is None:
            end_index = self.end_index
        if start_index is None:
            try:
                start_index = self.start_index
            except AttributeError:
                start_index = 0
    
        f = spectroscopy_run(experiment, run, verbose=verbose, start_index=start_index, end_index=end_index)
        f.load_run_keys(self.keys, self.friendly_names)
        f.load_run_key_delayed(self.key_epix, self.friendly_name_epix, rois=self.import_roi)
        f.get_run_shot_properties()
    
        analysis = XESAnalysis()
        for fil in self.filters:
            analysis.filter_shots(f, fil['FilterType'], fil['FilterKey'], fil['FilterThreshold'])
        ##arbitrary filters can't work because it calls union shots. which lowers the size of the epix shot array. then mismatches xray laser array
        #if self.arbitrary_filter:
        #    f.set_arbitrary_filter()
        #    analysis.union_shots(f, 'epix', ['xray', 'arbitrary_filter'], new_key=False)
        #else:
            #analysis.union_shots(f, 'epix', ['xray', 'xray'], new_key=False)
        analysis.filter_detector_adu(f, 'epix', adu_threshold=self.adu_cutoff)
        
        # Laser specific logic
        analysis.union_shots(f, 'epix', ['simultaneous', 'laser'])
        analysis.separate_shots(f, 'epix', ['xray', 'laser'])
    
        self.keys_to_save.extend(['epix_simultaneous_laser_ROI_1', 'epix_xray_not_laser_ROI_1'])
    

        # Ensure 'epix', 'epix_simultaneous_laser', and 'epix_xray_not_laser' go through the same steps
        for key in ['epix', 'epix_simultaneous_laser', 'epix_xray_not_laser']:
            if self.hitfind:
                from XSpect.XSpect_Processor import HitFinding
                f.update_status(f'Starting hit finding')
                hits, mean_sum, std_sum, threshold, sum_images = HitFinding.basic_detect(getattr(f, key), cutoff_multiplier=1, absolute_threshold=100)
                f.update_status(f'Hit finding on {key}. Hits found: {str(len(hits))}, median: {str(mean_sum)}, std: {std_sum}, threshold: {threshold}')
                setattr(f, key, getattr(f, key)[hits])
                f.update_status(f'Applying Hits to {key} detector. New size {str(np.shape(getattr(f, key))[0])}')
                f.run_shots[f'XES_Hits_{key}'] = len(hits)
                f.sum_images = sum_images

            analysis.reduce_detector_shots(f, key, purge=False, new_key=False)
    
            if self.transpose:
                setattr(f, key, np.transpose(getattr(f, key)))
    
            analysis.pixels_to_patch = self.pixels_to_patch
            analysis.patch_pixels(f, key, axis=0, mode=self.patch_mode)
    
            if self.angle != 0:
                setattr(f, key, rotate(getattr(f, key), angle=self.angle, axes=[0, 1]))
            analysis.reduce_detector_spatial(f, key, rois=self.rois, combine=True, purge=False)
    
        self.keys_to_save.extend(['start_index', 'end_index', 'run_file', 'run_number', 'verbose', 'status', 'status_datetime', 'epix_ROI_1', 'sum_images', 'epix', 'all_epix', 'run_shots'])
        f.purge_all_keys(self.keys_to_save)
        return f
        
    def primary_analysis(self,experiment,run,verbose=False,start_index=None,end_index=None):
        if end_index==None:
            end_index=self.end_index
        if start_index==None:
            try:
                start_index=self.start_index
            except AttributeError:
                start_index=0
        self.time_bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        self.end_index=end_index
        self.start_index=start_index
        f=spectroscopy_run(experiment,run,verbose=verbose,start_index=start_index,end_index=end_index)
        f.load_run_keys(self.keys,self.friendly_names)
        f.load_run_key_delayed(self.key_epix,self.friendly_name_epix,rois=self.import_roi)
        f.get_run_shot_properties()
        analysis=XESAnalysis()
        analysis.pixels_to_patch=self.pixels_to_patch
        analysis.filter_detector_adu(f,'epix',adu_threshold=self.adu_cutoff)

        if self.hitfind:
            from XSpect.XSpect_Processor import HitFinding
            f.update_status(f'Starting hit finding')
            hits,mean_sum,std_sum,threshold,sum_images=HitFinding.basic_detect(f.epix,cutoff_multiplier=1,absolute_threshold=100)
            f.update_status(f'Hit finding on epix. Hits found: {str(len(hits))}, median intensity: {str(mean_sum)}, std: {std_sum}, threshold: {threshold}')
            f.epix=f.epix[hits]
            f.xray=f.xray[hits]
            f.laser=f.laser[hits]
            f.simultaneous=np.logical_and(f.xray,f.laser)
            f.update_status(f'Applying Hits to ePix detector. New size {str(np.shape(f.epix)[0])}')
            for name in self.friendly_names:
                setattr(f, name,getattr(f,name)[hits])
            f.run_shots['XES_Hits']=len(hits)
            f.sum_images=sum_images




        #analysis.patch_pixels(f,'epix',axis=1)
        if self.angle!=0:
            f.epix=rotate(f.epix, angle=self.angle, axes=[1,2])
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold'])    
        analysis.union_shots(f,'epix',['simultaneous','laser'])
        analysis.separate_shots(f,'epix',['xray','laser'])
        self.bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        analysis.time_binning(f,self.bins,lxt_key=self.lxt_key)
        analysis.union_shots(f,'timing_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'timing_bin_indices',['xray','laser'])
        analysis.reduce_detector_temporal(f,'epix_simultaneous_laser','timing_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_temporal(f,'epix_xray_not_laser','timing_bin_indices_xray_not_laser',average=False)
        analysis.reduce_detector_spatial(f,'epix_simultaneous_laser_time_binned', rois=self.rois)
        analysis.reduce_detector_spatial(f,'epix_xray_not_laser_time_binned', rois=self.rois)
        analysis.make_energy_axis(f,f.epix_xray_not_laser_time_binned_ROI_1.shape[1],d=self.crystal_d_space,R=self.crystal_radius,A=self.crystal_detector_distance)
        keys_to_save=['start_index','end_index','run_file','run_number','run_shots','verbose','status','status_datetime','epix_xray_not_laser_time_binned_ROI_1','epix_simultaneous_laser_time_binned_ROI_1']
        f.purge_all_keys(keys_to_save)
        analysis.make_energy_axis(f,f.epix_xray_not_laser_time_binned_ROI_1.shape[1],d=self.crystal_d_space,R=self.crystal_radius,A=self.crystal_detector_distance)
        return f
    def primary_analysis_range(self, experiment, run, shot_ranges, verbose=False, method=None):
        try:
            if method is None:
                method = self.primary_analysis 
            start, end = shot_ranges
            return method(run=run, experiment=experiment, start_index=start, end_index=end, verbose=verbose)
        except Exception as e:
            # Return or log the exception with enough details
            return {"error": str(e), "run": run, "shot_ranges": shot_ranges}


    
    def hit_find(self,experiment,run,verbose=False,start_index=None,end_index=None):
        if end_index==None:
            end_index=self.end_index
        if start_index==None:
            try:
                start_index=self.start_index
            except AttributeError:
                start_index=0
        f=spectroscopy_run(experiment,run,verbose=verbose,start_index=start_index,end_index=end_index)
        f.load_run_keys(self.keys,self.friendly_names)
        f.load_run_key_delayed(self.key_epix,self.friendly_name_epix)
        f.get_run_shot_properties()

        analysis=XESAnalysis()
        analysis.pixels_to_patch=self.pixels_to_patch
        analysis.filter_detector_adu(f,'epix',adu_threshold=self.adu_cutoff)
        f.epix=np.nansum(np.nansum(f.epix,axis=1),axis=1)
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold'])  
        return f
            
        
            

        
    

class XASBatchAnalysis(BatchAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if self.scattering==True:
            f.load_sum_run_scattering('epix10k2M/azav_azav')
            f.ipm=f.scattering[:-1]
        analysis=XASAnalysis()
        try:
            ccm_val = getattr(f, 'ccm_E_setpoint')
            elist = np.unique(ccm_val)
        except KeyError as e:
            self.update_status('Key does not exist: %s' % e.args[0])
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
        analysis.reduce_detector_ccm_temporal(f,'epix_simultaneous_laser','timing_bin_indices_simultaneous_laser','ccm_bin_indices_simultaneous_laser',average=True)
        analysis.reduce_detector_ccm_temporal(f,'epix_xray_not_laser','timing_bin_indices_xray_not_laser','ccm_bin_indices_xray_not_laser',average=True)
        analysis.reduce_detector_ccm_temporal(f,'ipm_simultaneous_laser','timing_bin_indices_simultaneous_laser','ccm_bin_indices_simultaneous_laser',average=True)
        analysis.reduce_detector_ccm_temporal(f,'ipm_xray_not_laser','timing_bin_indices_xray_not_laser','ccm_bin_indices_xray_not_laser',average=True)
        return f

class XASBatchAnalysis_1D_ccm(BatchAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.minccm=7.105
        self.maxccm=7.135
        self.numpoints_ccm=100
        self.filters=[]
    def primary_analysis(self,experiment,run,verbose=False):
        f=spectroscopy_run(experiment,run,verbose=verbose)
        f.get_run_shot_properties()
        
        f.load_run_keys(self.keys,self.friendly_names)
        analysis=XASAnalysis()
        try:
            ccm_val = getattr(f, 'ccm_E_setpoint')
            elist = np.unique(ccm_val)
        except KeyError as e:
            self.update_status('Key does not exist: %s' % e.args[0])
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
#         self.time_bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
#         analysis.time_binning(f,self.time_bins)
        analysis.ccm_binning(f,'ccm_bins','ccm')
#         analysis.union_shots(f,'timing_bin_indices',['simultaneous','laser'])
#         analysis.separate_shots(f,'timing_bin_indices',['xray','laser'])
        analysis.union_shots(f,'ccm_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'ccm_bin_indices',['xray','laser'])
        analysis.reduce_detector_ccm(f,'epix_simultaneous_laser','ccm_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_ccm(f,'epix_xray_not_laser','ccm_bin_indices_xray_not_laser',average=False)
        analysis.reduce_detector_ccm(f,'ipm_simultaneous_laser','ccm_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_ccm(f,'ipm_xray_not_laser','ccm_bin_indices_xray_not_laser',average=False)
        return f

class XASBatchAnalysis_1D_time(BatchAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
#         try:
#             ccm_val = getattr(f, 'ccm_E_setpoint')
#             elist = np.unique(ccm_val)
#         except KeyError as e:
#             self.update_status('Key does not exist: %s' % e.args[0])
#             elist = np.linspace(self.minccm,self.maxccm,self.numpoints_ccm)
#         analysis.make_ccm_axis(f,elist)
        self.time_bins=np.linspace(self.mintime,self.maxtime,self.numpoints)
        analysis.time_binning(f,self.time_bins)
        for fil in self.filters:
            analysis.filter_shots(f,fil['FilterType'],fil['FilterKey'],fil['FilterThreshold']) 
        analysis.union_shots(f,'epix',['simultaneous','laser'])
        analysis.separate_shots(f,'epix',['xray','laser'])
        analysis.union_shots(f,'ipm',['simultaneous','laser'])
        analysis.separate_shots(f,'ipm',['xray','laser'])
#         analysis.union_shots(f,'ccm',['simultaneous','laser'])
#         analysis.separate_shots(f,'ccm',['xray','laser'])


#         analysis.ccm_binning(f,'ccm_bins','ccm')
        analysis.union_shots(f,'timing_bin_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'timing_bin_indices',['xray','laser'])
#         analysis.union_shots(f,'ccm_bin_indices',['simultaneous','laser'])
#         analysis.separate_shots(f,'ccm_bin_indices',['xray','laser'])
        analysis.reduce_detector_temporal(f,'epix_simultaneous_laser','timing_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_temporal(f,'epix_xray_not_laser','timing_bin_indices_xray_not_laser',average=False)
        analysis.reduce_detector_temporal(f,'ipm_simultaneous_laser','timing_bin_indices_simultaneous_laser',average=False)
        analysis.reduce_detector_temporal(f,'ipm_xray_not_laser','timing_bin_indices_xray_not_laser',average=False)
        return f

class ScanAnalysis_1D(BatchAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    def primary_analysis(self,experiment,run,verbose=False):
        f=spectroscopy_run(experiment,run=run,verbose=True)
        analysis=XASAnalysis()
        f.get_run_shot_properties()
        f.load_run_keys(self.keys,self.friendly_names)
        #f.ccm_bins=f.ccm
        analysis.bin_uniques(f,'scan')
        analysis.union_shots(f,'epix',['simultaneous','laser'])
        analysis.separate_shots(f,'epix',['xray','laser'])
        analysis.union_shots(f,'ipm',['simultaneous','laser'])
        analysis.separate_shots(f,'ipm',['xray','laser'])
        analysis.union_shots(f,'scan',['simultaneous','laser'])
        analysis.separate_shots(f,'scan',['xray','laser'])
        analysis.union_shots(f,'scanvar_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'scanvar_indices',['xray','laser'])
        analysis.reduce_detector_ccm(f,'epix_simultaneous_laser','scanvar_indices_simultaneous_laser',average=False,not_ccm=True)
        analysis.reduce_detector_ccm(f,'epix_xray_not_laser','scanvar_indices_xray_not_laser',average=False,not_ccm=True)
        analysis.reduce_detector_ccm(f,'ipm_simultaneous_laser','scanvar_indices_simultaneous_laser',average=False,not_ccm=True)
        analysis.reduce_detector_ccm(f,'ipm_xray_not_laser','scanvar_indices_xray_not_laser',average=False,not_ccm=True)
        return f
class ScanAnalysis_1D_XES(BatchAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def primary_analysis(self,experiment,run,verbose=False):
        f=spectroscopy_run(experiment,run=run,verbose=True)
        analysis=XESAnalysis()
        f.get_run_shot_properties()
        f.load_run_keys(self.keys,self.friendly_names)
        #f.ccm_bins=f.ccm
        analysis.bin_uniques(f,'scan')
        analysis.union_shots(f,'epix',['simultaneous','laser'])
        analysis.separate_shots(f,'epix',['xray','laser'])
        analysis.union_shots(f,'ipm',['simultaneous','laser'])
        analysis.separate_shots(f,'ipm',['xray','laser'])
        analysis.union_shots(f,'scan',['simultaneous','laser'])
        analysis.separate_shots(f,'scan',['xray','laser'])
        analysis.union_shots(f,'scanvar_indices',['simultaneous','laser'])
        analysis.separate_shots(f,'scanvar_indices',['xray','laser'])

        analysis.reduce_det_scanvar(f,'epix_simultaneous_laser','scanvar_indices_simultaneous_laser','scanvar_bins')
        analysis.reduce_det_scanvar(f,'epix_xray_not_laser','scanvar_indices_xray_not_laser','scanvar_bins')


        return f

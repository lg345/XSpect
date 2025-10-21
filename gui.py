

import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from io import StringIO
import json

# Import your analysis modules
import XSpect.XSpect_Analysis as XSpect_Analysis
import XSpect.XSpect_Controller as XSpect_Controller
import XSpect.XSpect_Visualization as XSpect_Visualization

# Initialize Panel
pn.extension('tabulator',design='material', sizing_mode='stretch_width')
pn.config.raw_css.append("""
.panel-widget-box {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 10px;
}
""")
class XSpectGUI:
    """Main GUI class for XFEL XAS Analysis"""
    
    def __init__(self):
        # Initialize default key aliases
        self.default_key_aliases = pd.DataFrame({
            'Key': ['epics/ccm_E', 'epicsUser/ccm_E_setpoint', 'tt/ttCorr', 
                    'epics/lxt_ttc', 'enc/lasDelay', 'ipm4/sum', 'ipm4/sum', 
                    'tt/AMPL', 'epix_2/ROI_0_sum'],
            'Alias': ['ccm', 'ccm_E_setpoint', 'time_tool_correction', 'lxt_ttc', 
                      'encoder', 'ipm', 'ipm5', 'time_tool_ampl', 'epix']
        })
        
        self.default_filters = [
            {'Type': 'xray', 'Key': 'ipm', 'Value': 400.0},
            {'Type': 'simultaneous', 'Key': 'ipm', 'Value': 400.0},
            {'Type': 'simultaneous', 'Key': 'time_tool_ampl', 'Value': 0.001},
        ]
        
        # Initialize widgets
        self._create_widgets()
        
        # Create layout
        self.layout = self._create_layout()
        
        # Results storage
        self.xas = None
        self.visualization = None
        self.last_figure = None
        self.use_parallel.param.watch(self._toggle_num_cores_visibility, 'value')
    
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # ===== Experiment Parameters =====
        self.lcls_run = pn.widgets.IntInput(name='LCLS Run', value=22, start=1)
        self.hutch = pn.widgets.Select(name='Hutch', options=['xcs', 'mfx', 'cxi', 'mec'], value='xcs')
        self.experiment_id = pn.widgets.TextInput(name='Experiment ID', value='xcsx1008722')
        
        # ===== Time Binning =====
        self.mintime = pn.widgets.FloatInput(name='Min Time (ps)', value=-1.0, step=0.1)
        self.maxtime = pn.widgets.FloatInput(name='Max Time (ps)', value=1.0, step=0.1)
        self.numpoints = pn.widgets.IntInput(name='Number of Points', value=20, start=1, end=100)
        
        # ===== Analysis Options =====
        self.run_ranges = pn.widgets.TextInput(name='Run Parser Ranges', value='44 48')
        self.use_parallel = pn.widgets.Checkbox(name='Use Parallel Processing', value=False)
        self.num_cores = pn.widgets.IntInput(
            name='Number of Cores',
            value=4,
            start=1,
            end=16,
            # Set initial visibility based on the checkbox's starting value
            visible=self.use_parallel.value 
        )        
        self.scattering_mode = pn.widgets.Checkbox(name='Scattering Mode', value=True)
        self.verbose = pn.widgets.Checkbox(name='Verbose Output', value=True)
        
        # ===== Visualization =====
        self.vmin = pn.widgets.FloatInput(name='Color Scale Min', value=-1.25, step=0.05)
        self.vmax = pn.widgets.FloatInput(name='Color Scale Max', value=1.25, step=0.05)
        self.ylim_min = pn.widgets.FloatInput(name='Y-axis Min', value=-0.75, step=0.05)
        self.ylim_max = pn.widgets.FloatInput(name='Y-axis Max', value=0.75, step=0.05)
        
        # ===== Key Aliases Table =====

        self.key_aliases_df = self.default_key_aliases.copy()
        self.key_aliases_table = pn.widgets.Tabulator(
            self.key_aliases_df,
            editors={'Key': 'input', 'Alias': 'input'},
            show_index=False,
            width=400,
            max_height=600
        )
        self.add_key_button = pn.widgets.Button(name='➕ Add Row', button_type='success', width=100)
        self.add_key_button.on_click(self._add_key_row)
        self.remove_key_button = pn.widgets.Button(name='➖ Remove Selected', button_type='danger', width=150)
        self.remove_key_button.on_click(self._remove_key_row)
        self.reset_keys_button = pn.widgets.Button(name='🔄 Reset to Defaults', button_type='warning', width=150)
        self.reset_keys_button.on_click(self._reset_keys)
        
        # ===== Filters =====
        self.filters_df = pd.DataFrame(self.default_filters)
        self.filters_table = pn.widgets.Tabulator(self.filters_df, show_index=False, max_height=600,width=400,)
        self.new_filter_type = pn.widgets.Select(name='Filter Type', options=['xray', 'laser', 'simultaneous'])
        self.new_filter_key = pn.widgets.Select(name='Key to Filter', options=self.key_aliases_df['Alias'].tolist())
        self.new_filter_value = pn.widgets.FloatInput(name='Threshold Value', value=0.0, step=0.1)
        self.add_filter_button = pn.widgets.Button(name='➕ Add Filter', button_type='success', width=120)
        self.add_filter_button.on_click(self._add_filter)
        self.remove_filter_button = pn.widgets.Button(name='➖ Remove Selected', button_type='danger', width=150)
        self.remove_filter_button.on_click(self._remove_filter)
        
        # ===== Run Analysis =====
        self.run_button = pn.widgets.Button(name='▶️ Run Analysis', button_type='primary', width=200, height=50)
        self.run_button.on_click(self._run_analysis)
        
        # ===== Progress and Status =====
        self.progress_bar = pn.indicators.Progress(name='Progress', value=0, max=100, width=400, visible=False)
        self.status_text = pn.pane.Markdown("**Status:** Ready to run analysis")
        self.log_output = pn.widgets.TextAreaInput(name='Analysis Log', height=150, disabled=True)
        
        # ===== Results Display =====
        self.results_pane = pn.pane.Matplotlib(None, tight=True)
        self.metrics_pane = pn.pane.Markdown("")
        self.stats_pane = pn.pane.DataFrame(
            None,  # Start with no data
            max_height=250,
            sizing_mode='stretch_width'
        )
        # ===== Export =====
        self.export_config_button = pn.widgets.FileDownload(
            callback=self._export_config,
            filename='xas_config.json',
            button_type='default',
            label='💾 Download Configuration',
            width=250
        )    

    
    def _add_key_row(self, event):
        new_row = pd.DataFrame({'Key': [''], 'Alias': ['']})
        self.key_aliases_df = pd.concat([self.key_aliases_df, new_row], ignore_index=True)
        self.key_aliases_table.value = self.key_aliases_df
        self.new_filter_key.options = self.key_aliases_df['Alias'].tolist()
    
    def _remove_key_row(self, event):
        if len(self.key_aliases_table.selection) > 0:
            idx = self.key_aliases_table.selection[0]
            self.key_aliases_df = self.key_aliases_df.drop(idx).reset_index(drop=True)
            self.key_aliases_table.value = self.key_aliases_df
            self.new_filter_key.options = self.key_aliases_df['Alias'].tolist()
    
    def _reset_keys(self, event):
        self.key_aliases_df = self.default_key_aliases.copy()
        self.key_aliases_table.value = self.key_aliases_df
        self.new_filter_key.options = self.key_aliases_df['Alias'].tolist()
    
    def _add_filter(self, event):
        new_filter = pd.DataFrame({
            'Type': [self.new_filter_type.value],
            'Key': [self.new_filter_key.value],
            'Value': [self.new_filter_value.value]
        })
        self.filters_df = pd.concat([self.filters_df, new_filter], ignore_index=True)
        self.filters_table.value = self.filters_df
    
    def _remove_filter(self, event):
        if len(self.filters_table.selection) > 0:
            idx = self.filters_table.selection[0]
            self.filters_df = self.filters_df.drop(idx).reset_index(drop=True)
            self.filters_table.value = self.filters_df
    
    def _log(self, message):
        current = self.log_output.value or ""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_output.value = f"{current}{timestamp} {message}\n"
    
    def _run_analysis(self, event):
        try:
            # --- (Setup and Validation - Unchanged) ---
            self.progress_bar.visible = True
            self.progress_bar.value = 0
            self.log_output.value = ""
            self.results_pane.object = None
            self.stats_pane.object = None # Reset statistics pane
            
            self._log("Validating configuration...")
            current_keys_df = self.key_aliases_table.value
            current_filters_df = self.filters_table.value
            
            if len(current_keys_df) == 0:
                self.status_text.object = "**Status:** ❌ Error - No key aliases defined"
                self._log("ERROR: Please add at least one key-alias mapping")
                return
    
            # --- (Initialization - Unchanged) ---
            self._log("🔧 Initializing experiment...")
            self.status_text.object = "**Status:** Initializing experiment..."
            self.progress_bar.value = 10
            xas_experiment = XSpect_Analysis.spectroscopy_experiment(
                lcls_run=self.lcls_run.value,
                hutch=self.hutch.value,
                experiment_id=self.experiment_id.value
            )
            
            self._log("⚙️ Configuring XAS analysis...")
            self.status_text.object = "**Status:** Configuring analysis..."
            self.progress_bar.value = 20
            self.xas = XSpect_Controller.XASBatchAnalysis()
            keys = current_keys_df['Key'].tolist()
            names = current_keys_df['Alias'].tolist()
            self.xas.set_key_aliases(keys, names)
            self._log(f"Set {len(keys)} key aliases")
            for _, filt in current_filters_df.iterrows():
                self.xas.add_filter(filt['Type'], filt['Key'], float(filt['Value']))
            self.xas.scattering = self.scattering_mode.value
            self.xas.mintime = self.mintime.value
            self.xas.maxtime = self.maxtime.value
            self.xas.numpoints = self.numpoints.value
            self.xas.verbose = self.verbose.value
            
            self.progress_bar.value = 30
            
            # --- (Run Parser and Statistics - MODIFIED) ---
            self._log("📊 Parsing runs and aggregating statistics...")
            self.status_text.object = "**Status:** Aggregating statistics..."
            
            self.xas.run_parser([self.run_ranges.value])
            

            self.progress_bar.value = 40
            
            # --- (Primary Analysis Loop - Unchanged) ---
            self._log("🔬 Running primary analysis (this may take a while)...")
            self.status_text.object = "**Status:** Running analysis..."
            start_time = time.time()
            if self.use_parallel.value:
                self._log(f"Using {self.num_cores.value} cores for parallel processing")
                self.xas.primary_analysis_parallel_loop(self.num_cores.value, xas_experiment)
            else:
                self.xas.primary_analysis_loop(xas_experiment)
            analysis_time = time.time() - start_time
            
            # --- (Populate Log Box - MODIFIED) ---
            # 3. Get the detailed status log from the backend
            if hasattr(self.xas, 'status') and hasattr(self.xas, 'status_datetime'):
                # Clear the simple log and replace with the detailed one
                self.log_output.value = "" 
                detailed_log = []
                for dt, msg in zip(self.xas.status_datetime, self.xas.status):
                    detailed_log.append(f"{dt} {msg}")
                self._log("\n".join(detailed_log))
                self._log(f"✅ Analysis complete in {analysis_time:.2f}s") # Add final message
            else:
                self._log(f"✅ Analysis complete in {analysis_time:.2f}s")
    
            self.progress_bar.value = 70
            
            # --- (Visualization - Unchanged) ---
            self._log("📈 Generating visualization...")
            self.status_text.object = "**Status:** Generating visualization..."
            self.visualization = XSpect_Visualization.XASVisualization()
            self.visualization.combine_spectra(
                xas_analysis=self.xas,
                xas_laser_key='epix_simultaneous_laser_time_energy_binned',
                xas_key='epix_xray_not_laser_time_energy_binned',
                norm_laser_key='ipm_simultaneous_laser_time_energy_binned',
                norm_key='ipm_xray_not_laser_time_energy_binned',
                interpolate=True
            )
                        # 1. Call the statistics methods
            self.xas.aggregate_statistics()
            run_stats_dict = self.xas.run_statistics
            
            # 2. Convert dictionary to a nice Pandas DataFrame and display it
            if run_stats_dict:
                stats_df = pd.DataFrame.from_dict(run_stats_dict, orient='index')
                stats_df.index.name = 'Run'
                self.stats_pane.object = stats_df
                self._log("Run statistics aggregated successfully.")

            
            self.progress_bar.value = 85
            plt.close('all')
            self.visualization.plot_2d_difference_spectrum(
                self.xas, vmin=self.vmin.value, vmax=self.vmax.value
            )
            plt.ylim(self.ylim_min.value, self.ylim_max.value)
            fig = plt.gcf()
            plt.tight_layout()
            self.last_figure = fig
            self.results_pane.object = fig
            
            # --- (Finalization - Unchanged) ---
            self.progress_bar.value = 100
            self._log("✅ Visualization complete!")
            self.status_text.object = f"**Status:** ✅ Analysis complete! ({analysis_time:.2f}s)"
            self.metrics_pane.object = f"""
            ### Analysis Results
            | Metric | Value |
            |--- |--- |
            | **Analysis Time** | {analysis_time:.2f}s |
            | **Time Points** | {self.numpoints.value} |
            """
            self.progress_bar.visible = False
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self._log(f"❌ ERROR: {str(e)}\n\n{error_details}")
            self.status_text.object = f"**Status:** ❌ Error during analysis"
            self.progress_bar.visible = False
    
    def _export_config(self):
        config = {
            'experiment': {
                'lcls_run': self.lcls_run.value,
                'hutch': self.hutch.value,
                'experiment_id': self.experiment_id.value,
                'run_ranges': self.run_ranges.value
            },
            'key_aliases': self.key_aliases_table.value.to_dict('records'),
            'filters': self.filters_table.value.to_dict('records'),
            'time_binning': {
                'min_time': self.mintime.value,
                'max_time': self.maxtime.value,
                'num_points': self.numpoints.value
            },
            'analysis_options': {
                'use_parallel': self.use_parallel.value,
                'num_cores': self.num_cores.value,
                'scattering_mode': self.scattering_mode.value,
                'verbose': self.verbose.value
            },
            'visualization': {
                'vmin': self.vmin.value,
                'vmax': self.vmax.value,
                'ylim_min': self.ylim_min.value,
                'ylim_max': self.ylim_max.value
            }
        }
        
        sio = StringIO()
        json.dump(config, sio, indent=2)
        sio.seek(0)
        return sio
    
    def _create_layout(self):
        """Create the complete GUI layout"""
        
        header = pn.pane.Markdown("# 🔬 XSpect Spectroscopy Analysis GUI")
        
        # ===== Key Aliases Tab =====
        key_aliases_tab = pn.Column(
            pn.pane.Markdown("### Key-Alias Mapping"),
            self.key_aliases_table,
            pn.Row(self.add_key_button, self.remove_key_button, self.reset_keys_button)
        )
        
        # ===== Filters Tab =====
        filters_tab = pn.Column(
            pn.pane.Markdown("### Current Filters"),
            self.filters_table,
            self.remove_filter_button,
            pn.layout.Divider(),
            pn.pane.Markdown("### Add New Filter"),
            pn.Row(self.new_filter_type, self.new_filter_key, self.new_filter_value, self.add_filter_button)
        )
        
        # ===== Run Analysis Tab =====
        run_analysis_tab = pn.Column(
            pn.pane.Markdown("### Run Analysis"),
            pn.Row(self.run_button, self.export_config_button),
            pn.layout.Divider(),
            self.status_text,
            self.progress_bar,
            pn.layout.Divider(),
            pn.pane.Markdown("### Analysis Log"),
            self.log_output,
            pn.layout.Divider(),
            pn.pane.Markdown("### Results"),
            self.metrics_pane,
            self.results_pane,
            pn.layout.Divider(), # ADD a divider for separation
            
            # ADD these lines for the statistics display
            pn.pane.Markdown("### Run Statistics"),
            self.stats_pane
        )
        
        tabs = pn.Tabs(
            ('Key Aliases', key_aliases_tab),
            ('Filters', filters_tab),
            ('Run Analysis', run_analysis_tab),
            dynamic=True
        )
        
        sidebar = pn.Column(
            pn.pane.Markdown("## Parameters"),
            pn.pane.Markdown("### Experiment"),
            self.lcls_run, self.hutch, self.experiment_id,
            pn.layout.Divider(),
            pn.pane.Markdown("### Time Binning"),
            self.mintime, self.maxtime, self.numpoints,
            pn.layout.Divider(),
            pn.pane.Markdown("### Analysis Options"),
            self.run_ranges, self.use_parallel, self.num_cores, self.scattering_mode, self.verbose,
            pn.layout.Divider(),
            pn.pane.Markdown("### Visualization"),
            self.vmin, self.vmax, self.ylim_min, self.ylim_max,
            width=320
        )
        
        main_layout = pn.Column(
            header,
            pn.Row(
                sidebar,
                tabs
            )
        )
        main_layout_content = pn.Column(
        header,
        pn.Row(
            sidebar,
            tabs
        ),
        # 2. Give it a fixed width to stop it from stretching
        width=1200,  # <-- You can adjust this value!
        sizing_mode='fixed'
    )
        
        # 3. Center the fixed-width layout within the available space for a professional look
        centered_layout = pn.Row(main_layout_content, align='center')
        
        # 4. Return the new centered layout
        return centered_layout
    
        #return main_layout
    
    def view(self):
        return self.layout
    
    def servable(self):
        return self.layout.servable()
    def _toggle_num_cores_visibility(self, event):
        """
        This callback is triggered when the 'use_parallel' checkbox is changed.
        It shows or hides the 'num_cores' widget based on the new value.
        """
    # The 'event.new' attribute contains the new value of the checkbox (True or False)
        self.num_cores.visible = event.new

# ===== For Jupyter Notebook Usage =====
def create_gui():
    gui = XSpectGUI()
    return gui.view()

# ===== For Standalone App =====
if __name__ == '__main__':
    gui = XSpectGUI()
    gui.servable()
else:
    # When imported in Jupyter
    gui = XSpectGUI()
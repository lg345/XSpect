import os


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
                self.experiment_directory = experiment_directory
                return experiment_directory
        raise Exception("Unable to find experiment directory.")


class spectroscopy_experiment(experiment):
    """
    A class to represent a spectroscopy experiment.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_detector(self, detector_name, detector_dimensions):
        self.detector_name = detector_name
        self.detector_dimensions = detector_dimensions

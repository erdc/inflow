"""
Tools to generate lateral inflow for a routing model given land surface model
(LSM) files containing one or more runoff variables and a weight table that
assigns areas of intersection with LSM grid cells to catchments.
"""
from glob import glob
from datetime import datetime, timedelta
import multiprocessing
import warnings
import logging
import os
import re

import numpy as np
from netCDF4 import Dataset, num2date, date2num

from inflow import utils

from inflow.lsm_runoff_rules import apply_era_interim_t255_runoff_rule
from inflow.lsm_runoff_rules import apply_era_interim_t1279_runoff_rule

SECONDS_PER_HOUR = 3600
CURRENT_TIMESTR = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')

class InflowAccumulator:
    """
    Manager for extracting land surface model runoff from netCDF and
    aggregating it over catchments.
    """
    INPUT_RUNOFF_FILL_VALUE = 0.0
    M3_RIV_FILL_VALUE = 0.0

    def __init__(self,
                 output_filename,
                 steps_per_input_file,
                 weight_table_file,
                 runoff_variable_names,
                 meters_per_input_runoff_unit,
                 input_time_step_hours,
                 output_time_step_hours,
                 land_surface_model_description,
                 input_runoff_file=None,
                 input_runoff_directory=None,
                 start_datetime=None,
                 end_datetime=None,
                 file_datetime_format='%Y%m%d',
                 file_timestamp_re_pattern=r'\d{8}',
                 input_runoff_file_ext='nc',
                 nproc=1,
                 output_time_units='seconds since 1970-01-01 00:00:00',
                 invalid_value=-9999,
                 runoff_rule_name=None,
                 rivid_lat_lon_file=None,
                 ensemble_index=None,
                 strict_file_checking=True,
                 log_filename=f'inflow_{CURRENT_TIMESTR}.log',
                 min_logging_level='INFO'):
        """
        Create a new InflowAccumulator instance.

        Parameters
        ----------
        output_filename : str
            Name of output file.
        steps_per_input_file : int
            Number of time steps in input file.
        weight_table_file : str, optional
            Name of file containing the weight table.
        runoff_variable_names : list
            Names of variables to be accumulated.
        meters_per_input_runoff_unit : float
            Factor to convert input runoff units to meters
        land_surface_model_description : str
            Identifier for the land surface model to be included as metadata in
            the output file.
        input_runoff_file : str (optional)
            Path to input runoff netCDF file. If both `input_runoff_file` and
            `input_runoff_directory` are specified. `input_runoff_file` will
            be processed and `input_runoff_directory` will be ignored.
        input_runoff_directory : str
            Name of directory where input runoff netCDF files are located.
            For a single file, `input_runoff_file` may be specified, and
            `input_runoff_directory` may be left as None (default value).
        input_time_step_hours : int or array_like (optional)
            Time increment in hours for each entry in the input file. This may
            be an integer value for a uniform time step or an array for
            for variable time step size.
        output_time_step_hours : int
            Time increment in hours for each entry in the output file.
        start_datetime : datetime.datetime, optional
            Input files with timestamps before this date will be ignored.
        end_datetime : datetime.datetime, optional
            Input files with timestamps after this date will be ignored.
        file_datetime_format : str, optional
            Pattern used to convert timestamp in input filenames to datetime.
        file_timestamp_re_pattern : str, optional
            Regular expression pattern used to identify timestamp in input
            files.
        input_runoff_file_ext : str, optional
            Input runoff file extension (e.g. "nc").
        nproc : int, optional
            Number of processors to use for parallel processing.
        output_time_units : str, optional
            Description of measure and origin for output time variable.
        invalid_value : int, optional
            Value used to denote an invalid entry in the weight table.
        runoff_rule_name : str, optional
            Identifier for input runoff processing rule.
        rivid_lat_lon_file : str, optional
            Name of file containing lat/lon coordinates for each river id.
        ensemble_index : int, optional
            If the first dimension of the input runoff array corresponds to
            members of an ensemble, this provides the index of the desired
            member. Default is None (to be used if the first dimension
            corresponds to time or geospatial information).
        strict_file_checking : bool, optional
            If True, read information from each input file to verify
            consistency with user-specified parameters.
        log_filename : str, optional
            The name of a file to which log information is to be written.
            If set to None, log information will not be recorded.
        min_logging_level : str, optional
            Minimum logging severity level for which log information is to be
            recorded.
        """
        # Attributes from input arguments.
        self.output_filename = output_filename
        self.steps_per_input_file = steps_per_input_file
        self.weight_table_file = weight_table_file
        self.runoff_variable_names = runoff_variable_names
        self.meters_per_input_runoff_unit = meters_per_input_runoff_unit
        self.input_time_step_hours = input_time_step_hours
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.file_datetime_format = file_datetime_format
        self.file_timestamp_re_pattern = file_timestamp_re_pattern
        self.input_runoff_file_ext = input_runoff_file_ext
        self.nproc = nproc
        self.land_surface_model_description = land_surface_model_description
        self.input_runoff_file = input_runoff_file
        self.input_runoff_directory = input_runoff_directory
        self.output_time_step_hours = output_time_step_hours
        self.output_time_step_seconds = (
            self.output_time_step_hours * SECONDS_PER_HOUR)
        self.output_time_units = output_time_units
        self.invalid_value = invalid_value
        self.runoff_rule_name = runoff_rule_name
        self.rivid_lat_lon_file = rivid_lat_lon_file
        self.ensemble_index = ensemble_index
        self.strict_file_checking = strict_file_checking
        self.log_filename = log_filename
        self.min_logging_level = min_logging_level

        # Instantiate a "do-nothing" logger to prevent errors that will occur
        # if `self.logger` is not defined.
        self.logger = logging.getLogger('dummy')

        # Derived attributes (to be determined).
        self.input_file_list = None
        self.sample_file = None
        self.sample_time = None
        self.input_runoff_ndim = None
        self.input_runoff_variable_shape = None
        self.sample_steps_per_input_file = None
        self.sample_time_step_hours = None
        self.output_steps_per_input_file = None
        self.grouped_file_condition = None
        self.files_per_group = None
        self.output_steps_per_file_group = None
        self.integrate_within_file_condition = None
        self.runoff_rule = None
        self.grouped_input_file_list = None
        self.output_indices = []
        self.time = None
        self.rivid = None
        self.latitude = None
        self.longitude = None
        self.weight_rivid = None
        self.weight_area = None
        self.weight_id = None
        self.weight_lat_indices = None
        self.weight_lon_indices = None
        self.rivid_weight_indices = []
        self.lsm_lat_indices = None
        self.lsm_lon_indices = None
        self.lat_lon_weight_indices = None
        self.lsm_min_lat_index = None
        self.lsm_max_lat_index = None
        self.lsm_min_lon_index = None
        self.lsm_max_lon_index = None
        self.lsm_lat_slice = None
        self.lsm_lon_slice = None
        self.n_lsm_lat_slice = None
        self.n_lsm_lon_slice = None
        self.subset_indices = None
        self.job_list = []
        self.time_step_is_variable = False

        self.runoff_rule_dict = {
            None: None,
            'erai_t255': apply_era_interim_t255_runoff_rule,
            'erai_t1279': apply_era_interim_t1279_runoff_rule}

    def configure_logging(self):
        """
        Configure logging.
        """
        if self.min_logging_level is None:
            level = logging.NOTSET
        elif self.min_logging_level.upper() == 'CRITICAL':
            level = logging.CRITICAL
        elif self.min_logging_level.upper() == 'ERROR':
            level = logging.ERROR
        elif self.min_logging_level.upper() == 'WARNING':
            level = logging.WARNING
        elif self.min_logging_level.upper() == 'INFO':
            level = logging.INFO
        elif self.min_logging_level.upper() == 'DEBUG':
            level = logging.DEBUG
        elif self.min_logging_level.upper() == 'NOTSET':
            level = logging.NOTSET
        else:
            level = logging.NOTSET

        if self.log_filename is None:
            pass
        else:
            logging_format = '%(asctime)s:%(name)s[%(levelname)s]: %(message)s'
            logging.basicConfig(filename=self.log_filename, level=level,
                                format=logging_format)
            self.logger = logging.getLogger(__name__)

    def evaluate_input_timestep(self):
        """
        Determine if `input_time_step_hours` is constant or variable.
        """
        if not utils.isiterable(self.input_time_step_hours):
            self.time_step_is_variable = False
        else:
            self.input_time_step_hours = np.asarray(self.input_time_step_hours)
            if (self.input_time_step_hours \
                == self.input_time_step_hours[0]).all():
                self.input_time_step_hours = self.input_time_step_hours[0]
                self.time_step_is_variable = False
            else:
                self.time_step_is_variable = True

    def generate_input_runoff_file_list(self):
        """
        Generate a time-ordered array of files from which to extract runoff.
        """
        if self.input_runoff_file is not None:
            self.input_file_list = [self.input_runoff_file]
            self.input_runoff_directory = os.path.dirname(
                self.input_runoff_file)
            self.input_runoff_file_ext = (
                self.input_runoff_file.split('/')[-1].split(',')[-1])
        elif self.input_runoff_directory is not None:
            self.generate_input_runoff_file_list_from_directory()
        else:
            self.logger.error('Either `input_runoff_file` or ' +
                              '`input_runoff_directory` must be specified.')

    def generate_input_runoff_file_list_from_directory(self):
        """
        Generate a list of files located in `input_runoff_directory` and
        sort by date.
        """
        self.logger.info(
            'Locating input runoff files with extension %s in directory %s.',
            self.input_runoff_file_ext, self.input_runoff_directory)

        input_file_expr = os.path.join(
            self.input_runoff_directory, f'*.{self.input_runoff_file_ext}')
        input_file_list = glob(input_file_expr)

        assert input_file_list, (
            f'No files found in {self.input_runoff_directory} with file ' +
            f'extension "{self.input_runoff_file_ext}".')

        sample_file = input_file_list[0]

        match = re.search(self.file_timestamp_re_pattern, sample_file)

        assert match is not None, (
            f'Filename {sample_file} does not contain a timestamp ' +
            'matching file_timestamp_re_pattern ' +
            f'{self.file_timestamp_re_pattern}.')

        datestr = match.group()

        assert datetime.strptime(datestr, self.file_datetime_format), (
            f'Unable to conver {datestr} to a datetime object using ' +
            f'file_datetime_format {self.file_datetime_format}.')

        input_file_array = np.array(input_file_list)

        logging.info('Sorting input runoff files by date/time.')

        input_timestamp_list = [
            utils.parse_timestamp_from_filename(
                f, self.file_timestamp_re_pattern, self.file_datetime_format)
            for f in input_file_array]
        input_timestamp_array = np.array(input_timestamp_list)
        sorted_by_time = input_timestamp_array.argsort()
        input_file_array = input_file_array[sorted_by_time]
        input_timestamp_array = input_timestamp_array[sorted_by_time]

        in_time_bounds = np.ones_like(input_timestamp_array, dtype=bool)

        if self.start_datetime is not None:
            self.logger.info('Ignoring files before %s.', self.start_datetime)

            on_or_after_start = (
                input_timestamp_array >= self.start_datetime)
            in_time_bounds = np.logical_and(in_time_bounds, on_or_after_start)

        if self.end_datetime is not None:
            self.logger.info('Ignoring files after %s.', self.end_datetime)

            on_or_before_end = (
                input_timestamp_array <= self.end_datetime)
            in_time_bounds = np.logical_and(in_time_bounds, on_or_before_end)

        input_file_array = input_file_array[in_time_bounds]

        self.input_file_list = list(input_file_array)

    def parse_sample_input_file(self):
        """
        Read basic variable information from a sample input file.
        """
        try:
            self.sample_file = self.input_file_list[0]
        except IndexError:
            self.sample_file = None

        assert self.sample_file is not None, (
            f'No input files found in {self.input_runoff_directory}.')

        self.logger.info('Reading sample input file %s.', self.sample_file)

        try:
            sample_data = Dataset(self.sample_file)
        except:
            sample_data = None

        assert sample_data is not None, (
            f'Unable to read file {self.sample_file} as a netCDF dataset.')

        try:
            sample_file_1 = self.input_file_list[1]
        except:
            sample_file_1 = None

        try:
            sample_data_1 = Dataset(sample_file_1)
        except:
            sample_data_1 = None

        try:
            sample_time_variable = sample_data['time']
        except:
            sample_time_variable = None

        try:
            sample_time_variable_1 = sample_data_1['time']
        except:
            sample_time_variable_1 = None

        try:
            self.sample_time = num2date(sample_time_variable[:],
                                        sample_time_variable.units)
        except:
            self.sample_time = None

        try:
            sample_time_1 = num2date(sample_time_variable_1[:],
                                   sample_time_variable_1.units)
        except:
            sample_time_1 = None

        sample_runoff_name = self.runoff_variable_names[0]

        try:
            sample_runoff_variable = sample_data[sample_runoff_name]
        except:
            sample_runoff_variable = None

        assert sample_runoff_variable is not None, (
            f'Variable "{sample_runoff_name}" not found in file ' +
            f'{self.sample_file}.')

        try:
            self.input_runoff_ndim = sample_runoff_variable.ndim
        except:
            self.input_runoff_ndim = None

        assert self.input_runoff_ndim is not None, (
            f'Variable {sample_runoff_variable} is not an n-dimensional ' +
            'array. At least two dimensions expected.')

        try:
            self.input_runoff_variable_shape = sample_runoff_variable.shape
        except:
            self.input_runoff_variable_shape = None

        assert self.input_runoff_variable_shape is not None, (
            f'Variable {sample_runoff_name} does not have a shape ' +
            'attribute. This should be a tuple with at least two elements.')

        if self.sample_time is not None:
            try:
                self.sample_steps_per_input_file = len(self.sample_time)
            except TypeError:
                self.sample_steps_per_input_file = 1
        elif self.input_runoff_ndim == 3:
            if self.ensemble_index is None:
                # Assume first dimension of runoff array corresponds to time.
                self.sample_steps_per_input_file = (
                        self.input_runoff_variable_shape[0])
            else:
                self.sample_steps_per_input_file = 1
        else:
            self.sample_steps_per_input_file = 1

        if self.sample_steps_per_input_file > 1:
            sample_time_diff = (
                #self.sample_time[1] - self.sample_time[0]).total_seconds()
                np.diff(self.sample_time))
            sample_time_step_seconds = np.asarray(
                [d.total_seconds() for d in sample_time_diff])
            sample_time_step_seconds = np.insert(
                sample_time_step_seconds, 0, sample_time_step_seconds[0])

        elif sample_time_1 is not None:
            try:
                sample_time_step_seconds = (
                    sample_time_1[0] - self.sample_time[0]).total_seconds()
            except (TypeError, IndexError):
                sample_time_step_seconds = (
                    sample_time_1 - self.sample_time).total_seconds()
        else:
            sample_time_step_seconds = None

        if sample_time_step_seconds is None:
            self.sample_time_step_hours = None
        else:
            self.sample_time_step_hours = (
                sample_time_step_seconds // SECONDS_PER_HOUR)

        if utils.isiterable(self.sample_time_step_hours):
            if (self.sample_time_step_hours \
                == self.sample_time_step_hours[0]).all():
                self.sample_time_step_hours = self.sample_time_step_hours[0]

        sample_data.close()

    def verify_user_parameters(self):
        """
        Verify that user-specified parameters are consistent with input data.
        """
        self.logger.info('Verifying that data in sample input file %s are ' \
                    'consistent with user-specified parameters.',
                    self.sample_file)

        assert (self.sample_steps_per_input_file ==
                self.steps_per_input_file), (
                    f'steps_per_input_file = {self.steps_per_input_file} ' +
                    f'specified, but {self.sample_steps_per_input_file} ' +
                    'found in {self.sample_file}.')

        if self.sample_time_step_hours is None:
            warnings.warn(
                'Unable to determine input time step from ' +
                f'{self.sample_file}. Using user-specified time step of ' +
                f'{self.input_time_step_hours} hour(s).')
        else:
            if not np.array_equal(
                    self.input_time_step_hours, self.sample_time_step_hours):
                warnings.warn(
                    'input_time_step_hours of ' +
                    f'{self.input_time_step_hours} hour(s) specified, ' +
                    f'but time step of {self.sample_time_step_hours} ' +
                    'hour(s) found for files in ' +
                    f'{self.input_runoff_directory}.')

        if self.time_step_is_variable:
            hours_per_input_file = np.sum(self.input_time_step_hours)
        else:
            hours_per_input_file = (
                self.input_time_step_hours * self.steps_per_input_file)

        output_time_step_divisible_by_input_file_hours = (
            self.output_time_step_hours % hours_per_input_file == 0)

        input_file_hours_divisible_by_output_time_step = (
            hours_per_input_file % self.output_time_step_hours == 0)

        assert (output_time_step_divisible_by_input_file_hours or
                input_file_hours_divisible_by_output_time_step), (
                    'output_time_step_hours must be an integer multiple' +
                    'of hours_per_input_file or hours_per_input_file ' +
                    'must be an integer multiple of ' +
                    'output_time_step_hours.')

        runoff_rule_keys = list(self.runoff_rule_dict.keys())
        assert self.runoff_rule_name in runoff_rule_keys, (
            f'Runoff rule {self.runoff_rule_name} not recognized. ' +
            f'Recognized runoff rules are {",".join(runoff_rule_keys)}.')

    def determine_output_steps_per_input_file(self):
        """
        Determine the number of output time steps per input file.
        """
        if self.time_step_is_variable:
            total_time = np.sum(self.input_time_step_hours)
            self.output_steps_per_input_file = (
                total_time / self.output_time_step_hours)
        else:
            if self.input_time_step_hours == self.output_time_step_hours:
                self.output_steps_per_input_file = self.steps_per_input_file
            else:
                output_steps_per_input_step = (
                    self.input_time_step_hours / self.output_time_step_hours)
                self.output_steps_per_input_file = (
                    self.steps_per_input_file * output_steps_per_input_step)

    def determine_file_integration_type(self):
        """
        Determine if input file values should be integrated within files
        and/or across multiple files.
        """
        if self.output_steps_per_input_file < 1:
            self.grouped_file_condition = True
        elif self.output_steps_per_input_file >= 1:
            self.grouped_file_condition = False

        if self.grouped_file_condition:
            self.files_per_group = (
                self.output_time_step_hours // self.input_time_step_hours)
            self.logger.info('Grouping files with %s files per group.',
                        self.files_per_group)
        else:
            self.files_per_group = 1

        self.output_steps_per_file_group = int(
            self.output_steps_per_input_file * self.files_per_group)

        if self.time_step_is_variable:
            self.integrate_within_file_condition = False
        else:
            self.integrate_within_file_condition = (
                self.output_steps_per_file_group < self.steps_per_input_file)

        if self.integrate_within_file_condition:
            self.logger.info('Summing accumulated runoff to produce output on a ' \
                        '%s-hour time step', self.output_time_step_hours)

    def determine_runoff_rule(self):
        """
        Assign function associated with `runoff_rule_name` to `runoff_rule` if
        `runoff_rule_name` is specified.
        """
        self.runoff_rule = self.runoff_rule_dict[self.runoff_rule_name]
        if self.runoff_rule is not None:
            self.logger.info('Applying function %s.', str(self.runoff_rule))

    def validate_input_files(self):
        """
        Verify that the specified input variables are present and that
        their dimensions are consistent in each input file.
        """
        for f in self.input_file_list:
            d = Dataset(f)

            for key in self.runoff_variable_names:
                file_runoff_shape = d[key].shape
                assert key in d.variables.keys(), (
                    f'Variable {key} not found in {f}.')

                assert (file_runoff_shape ==
                        self.input_runoff_variable_shape), (
                            f'Variable {key} in {f} does not have the ' +
                            f'same shape as variable {key} in ' +
                            f'{self.sample_file}.')

            try:
                file_time = d['time'][:]
            except:
                file_time = None

            if file_time is not None:
                try:
                    steps_per_file = len(file_time)
                except TypeError:
                    steps_per_file = 1
            elif len(file_runoff_shape) == 3:
                if self.ensemble_index is None:
                    steps_per_file = file_runoff_shape[0]
                else:
                    steps_per_file = 1
            else:
                steps_per_file = 1

            assert steps_per_file == self.steps_per_input_file, (
                f'File {f} has a different number of timesteps than ' +
                f'file {self.sample_file}.')

    def group_input_runoff_file_list(self):
        """
        In the case where values are to be integrated across n > 1 files,
        break the input file list into sublists of length n.
        """
        if self.grouped_file_condition:
            self.logger.info('Grouping input files in groups of %s to aggregate ' \
                        'runoff.', self.files_per_group)

            self.grouped_input_file_list = []
            ntrunc = len(self.input_file_list) % self.files_per_group

            if ntrunc == 0:
                stop = None
            else:
                stop = -ntrunc
                warnings.warn(
                    'Input files will be processed in groups ' +
                    f'of {self.files_per_group} to allow an ' +
                    'output time step of ' +
                    f'{self.output_time_step_hours}. This will ' +
                    f'result in {ntrunc} files being omitted.')

            input_file_list = self.input_file_list[:stop]
            nfiles = len(input_file_list)
            for grouped_idx in range(0, nfiles, self.files_per_group):
                self.grouped_input_file_list.append(self.input_file_list[
                    grouped_idx:grouped_idx + self.files_per_group])
        else:
            self.grouped_input_file_list = self.input_file_list

    def determine_output_indices(self):
        """
        Create a list of start and end output file indices for each input
        file.
        """
        self.logger.info('Determining ouput runoff array indices ' \
                    'corresponding to each input runoff increment.')

        self.output_indices = []

        start_idx = 0

        increment = int(self.files_per_group *
                        self.output_steps_per_input_file)

        for f in self.grouped_input_file_list:
            end_idx = (start_idx + increment)
            self.output_indices.append((start_idx, end_idx))
            start_idx += increment

    def generate_output_time_variable(self):
        """
        Construct the time variable for the output file.
        """
        n_time_step = int(len(self.input_file_list) *
                               self.output_steps_per_input_file)

        if self.start_datetime is None:
            try:
                self.start_datetime = utils.parse_timestamp_from_filename(
                    self.sample_file,
                    re_search_pattern=self.file_timestamp_re_pattern,
                    datetime_pattern=self.file_datetime_format)
            except:
                self.start_datetime = self.sample_time[0]

        start_seconds = date2num(self.start_datetime,
                                 self.output_time_units)

        elapsed_seconds = (
            np.arange(n_time_step) * self.output_time_step_seconds)

        total_elapsed_seconds = int(elapsed_seconds[-1])

        final_datetime = (
            self.start_datetime + timedelta(seconds=total_elapsed_seconds))

        self.logger.info('Constructing the output time variable from %s to %s ' \
                    'with a %s-second time step', self.start_datetime,
                    final_datetime, self.output_time_step_seconds)

        self.time = start_seconds + elapsed_seconds

    def initialize_inflow_nc(self):
        """
        Write variables, dimensions, and attributes to output file. This
        method populates all output data except the 'm3_riv' variable, which
        will be written to the file in parallel.
        """
        self.logger.info('Initializing dimensions and variables in ouput file %s.',
                    self.output_filename)

        data_out_nc = Dataset(self.output_filename, 'w')

        # create dimensions
        data_out_nc.createDimension('time', len(self.time))
        data_out_nc.createDimension('rivid', len(self.rivid))
        data_out_nc.createDimension('nv', 2)

        # create variables
        # m3_riv
        m3_riv_var = data_out_nc.createVariable('m3_riv', 'f4',
                                                ('time', 'rivid'),
                                                fill_value=0)
        m3_riv_var.long_name = 'accumulated external water volume ' \
                               'inflow upstream of each river reach'
        m3_riv_var.units = 'm3'
        m3_riv_var.coordinates = 'lon lat'
        m3_riv_var.grid_mapping = 'crs'
        m3_riv_var.cell_methods = "time: sum"

        # rivid
        rivid_var = data_out_nc.createVariable('rivid', 'i4', ('rivid',))
        rivid_var.long_name = 'unique identifier for each river reach'
        rivid_var.units = '1'
        rivid_var.cf_role = 'timeseries_id'
        rivid_var[:] = self.rivid

        # time
        time_var = data_out_nc.createVariable('time', 'i4', ('time',))
        time_var.long_name = 'time'
        time_var.standard_name = 'time'
        time_var.units = self.output_time_units
        time_var.axis = 'T'
        time_var.calendar = 'gregorian'
        time_var.bounds = 'time_bnds'
        time_var[:] = self.time

        # time_bnds
        time_bnds_var = data_out_nc.createVariable('time_bnds', 'i4',
                                                   ('time', 'nv',))
        for time_index, time_element in enumerate(self.time): #time_array):
            time_bnds_var[time_index, 0] = time_element
            time_bnds_var[time_index, 1] = (
                time_element + self.output_time_step_seconds)

        # longitude
        if self.longitude is None:
            self.logger.warning('No longitude values specified. Not writing ' \
                           'longitude variable to %s.', self.output_filename)
        else:
            lon_var = data_out_nc.createVariable('lon', 'f8', ('rivid',),
                                                 fill_value=-9999.0)
            lon_var.long_name = \
                'longitude of a point related to each river reach'
            lon_var.standard_name = 'longitude'
            lon_var.units = 'degrees_east'
            lon_var.axis = 'X'

        # latitude
        if self.latitude is None:
            self.logger.warning('No latitude values specified. Not writing ' \
                           'latitude variable to %s.', self.output_filename)
        else:
            lat_var = data_out_nc.createVariable('lat', 'f8', ('rivid',),
                                                 fill_value=-9999.0)
            lat_var.long_name = \
                'latitude of a point related to each river reach'
            lat_var.standard_name = 'latitude'
            lat_var.units = 'degrees_north'
            lat_var.axis = 'Y'

        if np.logical_and(self.latitude is None, self.longitude is None):
            self.logger.warning('No geospatial information specified. Not ' \
                           'writing coordinate reference system variable ' \
                           'to %s.', self.output_filename)
        else:
            crs_var = data_out_nc.createVariable('crs', 'i4')
            crs_var.grid_mapping_name = 'latitude_longitude'
            crs_var.epsg_code = 'EPSG:4326'  # WGS 84
            crs_var.semi_major_axis = 6378137.0
            crs_var.inverse_flattening = 298.257223563

        # add global attributes
        # data_out_nc.Conventions = 'CF-1.6'
        data_out_nc.title = ('RAPID Inflow from ' +
                             f'{self.land_surface_model_description}')

        data_out_nc.history = f'date_created: {datetime.utcnow()}'

        # data_out_nc.featureType = 'timeSeries'
        #data_out_nc.institution = modeling_institution

        # write lat lon data
        if self.latitude is not None:
            lat_var[:] = self.latitude
        if self.longitude is not None:
            lon_var[:] = self.longitude

        # close file
        data_out_nc.close()

    def read_weight_table(self):
        """
        Read the weight table file.
        """
        self.logger.info('Reading weight table from %s.', self.weight_table_file)

        weight_table = np.genfromtxt(self.weight_table_file, delimiter=',',
                                     skip_header=1)

        self.weight_rivid = weight_table[:,0].astype(int)
        self.weight_area = weight_table[:,1]
        self.weight_lat_indices = weight_table[:,3].astype(int)
        self.weight_lon_indices = weight_table[:,2].astype(int)

        # Include try except clause for compatibility with older weight-table
        # files that do not contain a unique id field.
        try:
            self.weight_id = weight_table[:,7].astype(int)
        except:
            self.weight_id = None

        # Check if the weight table contains any invalid values. If it does,
        # assign placeholder values to prevent downstream invalid value
        # errors.
        valid = self.weight_lat_indices != self.invalid_value

        dummy_lat_index = self.weight_lat_indices[valid][0]
        dummy_lon_index = self.weight_lon_indices[valid][0]

        self.weight_lat_indices[~valid] = dummy_lat_index
        self.weight_lon_indices[~valid] = dummy_lon_index

        self.rivid = utils.unique_ordered(self.weight_rivid)

    def read_rivid_lat_lon(self):
        """
        Read the latitude and longitude coordinates corresponding to
        `rivid` from `rivid_lat_lon_file` (if the file is
        specified) and assign values to `latitude` and `longitude`.
        """
        if self.rivid_lat_lon_file is None:
            self.logger.warning('No lat/long file specified.')

            data = None
        else:
            self.logger.info('Reading latitude and longitude from %s.',
                        self.rivid_lat_lon_file)

            data = np.genfromtxt(self.rivid_lat_lon_file, delimiter=',',
                                 skip_header=1, usecols=[0,1,2])

        if data is None:
            rivid_lat_lon_dict = None
        else:
            rivid_lat_lon_dict = {}

            rivid = data[:,0]
            lat = data[:,1]
            lon = data[:,2]

            for uid, c1, c2 in zip(rivid, lat, lon):
                rivid_lat_lon_dict[uid] = (c1, c2)

        if rivid_lat_lon_dict is not None:
            self.latitude = np.zeros(len(self.rivid))
            self.longitude = np.zeros(len(self.rivid))
            for idx, uid in enumerate(self.rivid):
                self.latitude[idx] = rivid_lat_lon_dict[uid][0]
                self.longitude[idx] = rivid_lat_lon_dict[uid][1]

    def find_rivid_weight_indices(self):
        """
        Given the ordered array of unique identifiers, `self.rivid`,
        create a list whose elements are the indices in the weight table
        corresponding to those identifiers. The nth element in the list
        will contain the indices corresponding to the nth element in
        `self.rivid`.
        """
        self.logger.info('Identifying weight-table indices for each rivid.')

        self.rivid_weight_indices = []
        for rivid in self.rivid:
            rivid_weight_idx = np.where(self.weight_rivid == rivid)[0]
            self.rivid_weight_indices.append(rivid_weight_idx)

    def find_lat_lon_weight_indices(self):
        """
        Create an array of unique input lat/lon indices
        `lat_lon_indices`. Then create an array,
        `lat_lon_weight_indices` with the same size as the number of
        rows in the weight table. Finally, populate the nth element of
        `lat_lon_weight_indices` with the index in
        `lat_lon_indices` corresponding to the nth weight-table entry.
        """
        self.logger.info('Identifying weight-table indices for each input ' \
                    'runoff grid location.')

        weight_lat_lon_indices = np.column_stack(
            [self.weight_lat_indices, self.weight_lon_indices])

        lat_lon_indices = np.unique(weight_lat_lon_indices, axis=0)
        self.lsm_lat_indices = lat_lon_indices[:,0]
        self.lsm_lon_indices = lat_lon_indices[:,1]

        self.lat_lon_weight_indices = np.zeros(
            len(weight_lat_lon_indices),dtype=int)

        for idx, lat_lon_idx in enumerate(lat_lon_indices):
            lat_lon_weight_idx = (
                (weight_lat_lon_indices == lat_lon_idx).all(axis=1))
            self.lat_lon_weight_indices[lat_lon_weight_idx] = idx

    def find_lat_lon_input_indices(self):
        """
        Identify the largest and smallest latitude and longitude indices to
        be extracted from the input files and determine array slices that
        comprise all of the indices that lie within these bounds.
        """
        self.logger.info(
            'Determining minimal subset of the input runoff grid that ' \
            'includes all weight-table spatial locations.')

        self.lsm_min_lat_index = self.weight_lat_indices.min()
        self.lsm_max_lat_index = self.weight_lat_indices.max()
        self.lsm_min_lon_index = self.weight_lon_indices.min()
        self.lsm_max_lon_index = self.weight_lon_indices.max()

        self.lsm_lat_slice = slice(self.lsm_min_lat_index,
                                   self.lsm_max_lat_index + 1)
        self.lsm_lon_slice = slice(self.lsm_min_lon_index,
                                   self.lsm_max_lon_index + 1)

        self.n_lsm_lat_slice = (
            self.lsm_lat_slice.stop - self.lsm_lat_slice.start)
        self.n_lsm_lon_slice = (
            self.lsm_lon_slice.stop - self.lsm_lon_slice.start)

    def find_subset_indices(self):
        """
        Determine a new set of indices that correspond to only those spatial
        locations in the output file that are represented in the weight
        table. This array is structured to conform to the shape of the
        "flattened" input array. i.e. it provides the indices of the
        relevant spatial locations after the dimensions of the input runoff
        array have been changed from (time, lat, lon) to (time, lat/lon).
        """
        self.logger.info('Determining subset runoff grid indices.')

        self.subset_indices = (
            (self.lsm_lat_indices -
             self.lsm_min_lat_index)*self.n_lsm_lon_slice +
            (self.lsm_lon_indices - self.lsm_min_lon_index))

    def write_multiprocessing_job_list(self):
        """
        Write a list of dictionaries, each of which contains information
        required to process a single input file.
        """
        self.logger.info('Writing job list for parallel processing.')

        self.job_list = []

        mp_lock = multiprocessing.Manager().Lock()

        for idx, item in enumerate(self.grouped_input_file_list):
            args = {}
            args['input_file_list'] = item
            args['output_indices'] = self.output_indices[idx]
            args['mp_lock'] = mp_lock
            self.job_list.append(args)

    def adjust_inflow_for_variable_input_time_step(self, inflow):
        """
        Upscale and downscale accumulated runoff in the case that input
        runoff data has a variable time-step length.
        """
        spatial_dimension_size = int(inflow.shape[1])
        unique_time_increment = np.unique(self.input_time_step_hours)
        output_inflow_shape = (
            self.output_steps_per_input_file, spatial_dimension_size)
        output_inflow = np.zeros(output_inflow_shape)

        output_index = 0
        for increment in unique_time_increment:
            indices_increment = (self.input_time_step_hours == increment)
            inflow_increment = inflow[indices_increment]
            output_stride = increment / self.output_time_step_hours
            n_increment = np.sum(indices_increment)
            n_out = output_stride * n_increment

            if n_out.is_integer():
                n_out = int(n_out)
            else:
                self.logger.error('Number of output time steps `n_out` must ' +
                                  'be an integer value.')
            if increment < self.output_time_step_hours:
                inflow_increment = utils.sum_over_time_increment(
                    inflow_increment, n_output_steps=n_out)
            elif increment == self.output_time_step_hours:
                pass
            else:
                weight = 1 / output_stride
                inflow_increment = weight * np.repeat(
                    inflow_increment, output_stride, axis=0)

            self.logger.info(f'Writing to indices {output_index} to ' +
                             f'{output_index + n_out}.')
            self.logger.info(
                f'Using input time increment of {increment} hours.')
            self.logger.info(f'Writing {output_stride} entries for each ' +
                             'input time step.')

            output_inflow[output_index:output_index + n_out] = inflow_increment
            output_index += n_out

        return output_inflow

    def read_write_inflow(self, args):
        """
        Extract runoff timeseries data from one netCDF file and write
        accumulated runoff [m^3] to another.

        Parameters
        ----------
        args : dict
            File/process-specific parameters
        """
        input_file_list = args['input_file_list']
        if not isinstance(input_file_list, list):
            input_file_list = [input_file_list]

        start_idx, end_idx = args['output_indices']

        input_file_str = '\n  '.join(input_file_list)

        self.logger.info('Reading the following input file(s):\n  %s',
                    input_file_str)
        self.logger.info('Writing to output indices %s to %s',
                    start_idx, end_idx-1)

        mp_lock = args['mp_lock']

        cumulative_inflow = np.zeros(
            [self.output_steps_per_file_group, len(self.rivid)])

        for input_filename in input_file_list:
            data_in = Dataset(input_filename)

            input_runoff = np.zeros([self.steps_per_input_file,
                                     self.n_lsm_lat_slice,
                                     self.n_lsm_lon_slice])

            # Sum values over all specified runoff variables for the region
            # indicated by `lat_slice` and `lon_slice`. Dimensions of
            # `input_runoff` are (time x lat x lon), where time, lat, and lon
            # refer to the dimensions of the subset extracted from `data_in`.
            for runoff_key in self.runoff_variable_names:
                if self.input_runoff_ndim == 3:
                    if self.ensemble_index is None:
                        runoff_increment = data_in[runoff_key][
                            :, self.lsm_lat_slice, self.lsm_lon_slice]
                    else:
                        runoff_increment = data_in[runoff_key][
                            self.ensemble_index, self.lsm_lat_slice,
                            self.lsm_lon_slice]
                elif self.input_runoff_ndim == 2:
                    runoff_increment = data_in[runoff_key][
                            self.lsm_lat_slice, self.lsm_lon_slice]

                input_runoff += runoff_increment.filled(
                        fill_value=self.INPUT_RUNOFF_FILL_VALUE)

            data_in.close()

            # Reshape the input runoff array so that the first dimension
            # corresponds to time and the second dimension corresponds to
            # geospatial coordinates. This reduces the number of dimensions
            # from three (e.g time, lat, lon) to two (e.g. time, latlon).
            # Dimensions of `input_runoff` are (time x latlon), where time is
            # the dimension of the `time` variable in `data_in` and latlon is
            # the product of the lat and lon dimensions in `data_in`.
            input_runoff = input_runoff.reshape(
                self.steps_per_input_file,
                self.n_lsm_lon_slice * self.n_lsm_lat_slice)

            # Extract only values that correspond to unique locations
            # represented in the weight table. `subset_indices` provides the
            # indices in the spatial dimension of `input_runoff` that
            # correspond to these unique locations. The new dimensions of
            # `input_runoff` are (time x latlon), where latlon now refers to
            # the number of unique latlon coordinate pairs appearing in the
            # weight table.
            input_runoff  = input_runoff[:, self.subset_indices]

            if self.runoff_rule is not None:
                input_runoff = self.runoff_rule(input_runoff)

            # Convert the runoff from its native units to meters. The
            # dimensions of `input_runoff_meters` are (time x latlon), where
            # latlon refers to the number of unique latlon coordinate pairs
            # appearing in the weight table.
            input_runoff_meters = (input_runoff *
                                   self.meters_per_input_runoff_unit)

            # Redistribute runoff values at unique spatial coordinates to all
            # weight table locations (indexed by latitude, longitude, and
            # catchment id). The dimensions of `weight_runoff_meters` are
            # (time x nweight), where nweight is the number of entries in the
            # weight table.
            weight_runoff_meters = input_runoff_meters[
                :, self.lat_lon_weight_indices]

            # Convert runoff in [m^2] to [m^3] by multiplying input runoff by
            # areas provided by the weight table. The dimensions of
            # `weighted_runoff_m3` are (time x nweight), where nweight is the
            # number of entries in the weight table.
            weighted_runoff_m3 = weight_runoff_meters * self.weight_area

            # `accumulated_runoff_m3` will hold the the cumulative runoff
            # volumes for each catchment. The dimensions of
            # `accumulated_runoff_m3` are (time x nrivid), where nrivid is
            # number of unique catchment identifiers that appear in the weight
            # table.
            accumulated_runoff_m3 = np.zeros([self.steps_per_input_file,
                                              len(self.rivid)])

            # For each catchment ID, sum runoff [m^3] over all regions within
            # the associated catchment and record the result in the
            # corresponding column of `accumulated_runoff_m3`.
            for rivid_idx, rivid_weight_idx in enumerate(
                    self.rivid_weight_indices):
                summed_by_rivid = np.sum(
                    weighted_runoff_m3[:, rivid_weight_idx], axis=1)
                accumulated_runoff_m3[:,rivid_idx] = summed_by_rivid

            if self.integrate_within_file_condition:
                if isinstance(self.output_steps_per_input_file, int):
                    output_steps_per_input_file = (
                        self.output_steps_per_input_file)
                elif self.output_steps_per_input_file.is_integer():
                    output_steps_per_input_file = int(
                        self.output_steps_per_input_file)
                else:
                    raise ValueError(
                        'output_steps_per_input_file must have an ' +
                        'integer value when used as an argument for ' +
                        'sum_over_time_increment. Found value ' +
                        'output_steps_per_input_file = ' +
                        f'{self.output_steps_per_input_file}.')

                accumulated_runoff_m3 = utils.sum_over_time_increment(
                    accumulated_runoff_m3, output_steps_per_input_file)

            elif self.time_step_is_variable:
                accumulated_runoff_m3 = (
                    self.adjust_inflow_for_variable_input_time_step(
                        accumulated_runoff_m3))

            cumulative_inflow += accumulated_runoff_m3

            # Replace invalid values with 0.0. 0.0 is masked (default
            # "_FillValue") in the output "m3_riv" variable.
            cumulative_inflow = np.where((cumulative_inflow < 0.0),
                    self.M3_RIV_FILL_VALUE, cumulative_inflow)
            cumulative_inflow = np.where(
                np.isnan(cumulative_inflow), self.M3_RIV_FILL_VALUE,
                cumulative_inflow)

        # Write the accumulated runoff [m^3] to the output file at the
        # appropriate indices along the time dimension. Use a multiprocessing
        # lock to prevent more than one process writing to the file at a time.
        mp_lock.acquire()

        data_out = Dataset(self.output_filename, "a")

        try:
            data_out['m3_riv'][start_idx:end_idx, :] = cumulative_inflow
        except:
            warnings.warn(
                f'Unable to write to "m3_riv" variable from indices {start_idx} ' +
                f'to {end_idx} in file {self.output_filename}.')

        data_out.close()
        
        mp_lock.release()

    def log_input_arguments(self):
        """
        Write a log entry displaying arguments used to initialize the class
        instance.
        """
        input_keys = [
            'output_filename',
            'input_runoff_directory',
            'steps_per_input_file',
            'weight_table_file',
            'runoff_variable_names',
            'meters_per_input_runoff_unit',
            'input_time_step_hours',
            'start_datetime',
            'end_datetime',
            'file_datetime_format',
            'file_timestamp_re_pattern',
            'input_runoff_file_ext',
            'nproc',
            'land_surface_model_description',
            'output_time_step_hours',
            'output_time_units',
            'invalid_value',
            'runoff_rule_name',
            'rivid_lat_lon_file',
            'ensemble_index',
            'strict_file_checking',
            'log_filename',
            'min_logging_level']

        input_str = ''
        for k in input_keys:
            input_str += f'  {k}: {self.__dict__[k]}\n'

        self.logger.info(
            'Generating inflow file with the following parameters:\n%s',
            input_str)

    def generate_inflow_file(self):
        """
        The main routine for the InflowAccumulator class.
        """
        self.configure_logging()

        self.log_input_arguments()

        self.generate_input_runoff_file_list()

        self.parse_sample_input_file()

        self.evaluate_input_timestep()

        self.verify_user_parameters()

        self.determine_runoff_rule()

        self.determine_output_steps_per_input_file()

        self.determine_file_integration_type()

        if self.strict_file_checking:
            self.logger.info('Validating input files.')
            self.validate_input_files()
        else:
            self.logger.info('Not validating input files.')

        self.read_weight_table()

        self.find_rivid_weight_indices()

        self.find_lat_lon_weight_indices()

        self.find_lat_lon_input_indices()

        self.find_subset_indices()

        self.generate_output_time_variable()

        self.group_input_runoff_file_list()

        self.determine_output_indices()

        self.read_rivid_lat_lon()

        self.initialize_inflow_nc()

        self.write_multiprocessing_job_list()

        self.logger.info('Running read_write_inflow() with %s processor(s).',
                    self.nproc)

        if self.nproc == 1:
            # Process input files serially. This is useful for debugging.
            for job in self.job_list:
                self.read_write_inflow(job)
        else:
            with multiprocessing.Pool(self.nproc) as pool:
                pool.map(self.read_write_inflow, self.job_list)

        self.logger.info('Done.')

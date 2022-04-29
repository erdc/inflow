"""
Tools to generate lateral inflow for a routing model given land surface model 
(LSM) files containing one or more runoff variables and a weight table that
assigns areas of intersection with LSM grid cells to catchments. 
"""
from glob import glob
import numpy as np
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
from functools import partial
import multiprocessing
import sys
import os
import re

SECONDS_PER_HOUR = 3600

def parse_time_from_nc(filename):
    """
    Extract the time variable from a netCDF file.

    Parameters
    ----------
    filename : str
        Name of file.

    Returns
    -------
    time : ndarray
        1D array of integer values.
    units : str
        Description of measure and origin of time values in `time`.
    """
    d = Dataset(filename)
    time_variable = d['time']
    units = time_variable.units
    time = time_variable[:]
    d.close()

    return time, units

def parse_timestamp_from_filename(filename, re_search_pattern=r'\d{8}',
                                  datetime_pattern='%Y%m%d'):
    """
    Determine date and time of file from its name.
    
    Parameters
    ----------
    filename : str
        Name of file.
    re_search_pattern : str
        Regular expression pattern used to identify date.
    datetime_pattern : str
        Rule to convert a string to a datetime object.

    Returns
    -------
    dt : datetime.datetime
        Date and time as a datetime object.
    """
    match = re.search(re_search_pattern, filename)
    datestr = match.group()
    dt = datetime.strptime(datestr, datetime_pattern)

    return dt

def convert_time(in_datenum_array, input_units, output_units):
    """
    Convert numerical values of date/time between measure/origin systems.

    Parameters
    ----------
    in_datenum_array : ndarray
        1D array of integer values representing time.
    input_units : str
        Description of measure and origin for `in_datenum_array`.
    output_units :
        Description of measure and origin for new time system.

    Returns
    -------
    out_datenum_array : ndarray
        1D array of integer values representing time.
    """
    datetime_array = num2date(in_datenum_array, input_units)
    out_datenum_array = date2num(datetime_array, output_units)

    return out_datenum_array

def sum_over_time_increment(data, old_timestep_hours,
                            new_timestep_hours, steps_per_file):
    """
    Sum values over specified interval in the time dimension.

    Parameters
    ----------
    data : ndarray
        Array with first dimension corresponding to a time variable.
    old_timestep_hours : int
        Time increment for values in `data`.
    new_timestep_hours : int
        Time increment over which to sum `data`.
    steps_per_file: int
        Number of timesteps represented in `data`.

    Returns
    -------
    summed_data : ndarray
        `data` summed over `new_timestep_hours`.
    """
    file_time_hours = steps_per_file * old_timestep_hours
    
    new_time_dim = int(file_time_hours / new_timestep_hours)

    # We add a new dimension, tmp_dim, to sum over.
    tmp_dim = int(new_timestep_hours)
    data = data.reshape(new_time_dim, tmp_dim, -1)
    summed_data = data.sum(axis=1)

    return summed_data
    
class InflowAccumulator:
    """
    Manager for extracting land surface model runoff from netCDF and
    aggregating it over catchments.
    """
    def __init__(self,
                 output_filename,
                 input_runoff_file_directory,
                 steps_per_input_file,
                 weight_table_file,
                 runoff_variable_names,
                 meters_per_input_runoff_unit,
                 output_time_step_hours,
                 land_surface_model_description,
                 input_time_step_hours=None,
                 start_datetime=None,
                 end_datetime=None,
                 file_datetime_format='%Y%m%d',
                 file_timestamp_re_pattern=r'\d{8}',
                 input_runoff_file_ext='nc',
                 nproc=1,
                 output_time_units='seconds since 1970-01-01 00:00:00',
                 invalid_value=-9999,
                 convert_one_hour_to_three=False):
        """
        Create a new InflowAccumulator instance.

        Parameters
        ----------
        output_filename : str
            Name of output file.
        input_runoff_file_directory : str
            Name of directory where input runoff netCDF files are located.
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
        input_time_step_hours : int, optional
            Time increment in hours for each entry in the input file.  
        output_time_step_hours : int, optional
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
        convert_one_hour_to_three : bool
            Convert runoff input on a one-hourly timestep to output on a
            three-hourly timestep.
        """
        self.output_filename = output_filename
        self.input_runoff_file_directory = input_runoff_file_directory
        self.steps_per_input_file = steps_per_input_file
        self.weight_table_file = weight_table_file
        self.runoff_variable_names = runoff_variable_names
        self.meters_per_input_runoff_unit = meters_per_input_runoff_unit
        self.input_time_step_hours = input_time_step_hours
        self.output_time_step_hours = output_time_step_hours
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.file_datetime_format = file_datetime_format
        self.file_timestamp_re_pattern = file_timestamp_re_pattern
        self.input_runoff_file_ext = input_runoff_file_ext
        self.nproc = nproc
        self.land_surface_model_description = land_surface_model_description
        self.output_time_units = output_time_units
        self.invalid_value = invalid_value
        self.input_file_array = None
        self.input_time_variable = None
        self.rivid = None
        self.convert_one_hour_to_three = convert_one_hour_to_three

        self.time = None
        self.weight_lat_indices = None
        self.weight_lon_indices = None
        self.lat_slice = None
        self.lon_slice = None
        self.n_lat_slice = None
        self.n_lon_slice = None
        
    def generate_input_runoff_file_array(self):
        """
        Generate a time-ordered array of files from which to extract runoff.
        """
        input_file_expr = os.path.join(
            self.input_runoff_file_directory, f'*.{self.input_runoff_file_ext}')
        input_file_list = glob(input_file_expr)
        input_file_array = np.array(input_file_list)
        input_timestamp_list = [
            parse_timestamp_from_filename(
                f, self.file_timestamp_re_pattern,
                self.file_datetime_format)
            for f in input_file_array]
        input_timestamp_array = np.array(input_timestamp_list)
        sorted_by_time = input_timestamp_array.argsort()
        input_file_array = input_file_array[sorted_by_time]
        input_timestamp_array = input_timestamp_array[sorted_by_time]
        
        in_time_bounds = np.ones_like(input_timestamp_array, dtype=bool)

        if self.start_datetime is not None:
            on_or_after_start = (
                input_timestamp_array >= self.start_datetime)
            in_time_bounds = np.logical_and(in_time_bounds, on_or_after_start)

        if self.end_datetime is not None:
            on_or_before_end = (
                input_timestamp_array <= self.end_datetime)
            in_time_bounds = np.logical_and(in_time_bounds, on_or_before_end)

        self.input_file_array = input_file_array[in_time_bounds]
        
    def determine_output_indices(self):
        """
        Create a list of start and end output file indices for each input
        file.
        """
        self.output_indices = []

        start_idx = 0
        for f in self.input_file_array:
            end_idx = start_idx + self.steps_per_input_file
            self.output_indices.append((start_idx, end_idx))
            start_idx += self.steps_per_input_file
        
    def concatenate_time_variable(self):
        """
        Extract time variable from all input files, convert to output units,
        and combine in a single array.
        """
        # TODO: we will have to modify the time variable if the output
        # time step is different than the input time step.
        n_time_step = len(self.input_file_array) * self.steps_per_input_file
    
        time = np.zeros(n_time_step)

        for f, idx in zip(self.input_file_array, self.output_indices):
            start_idx = idx[0]
            end_idx = idx[1]
            file_time, units = parse_time_from_nc(f)
            converted_time = convert_time(file_time, units,
                                          self.output_time_units)
            time[start_idx:end_idx] = converted_time[:]

        self.time = time
        
    def initialize_inflow_nc(self):
        """
        Write variables, dimensions, and attributes to output file. This
        method populates all output data except the 'm3_riv' variable, which
        will be written to the file in parallel.
        """
        output_time_step_seconds = (
            self.output_time_step_hours * SECONDS_PER_HOUR)
                
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

        # initial_time_seconds = \
        #     (start_datetime_utc.replace(tzinfo=utc) -
        #      datetime(1970, 1, 1, tzinfo=utc)).total_seconds()
        # final_time_seconds = \
        #     initial_time_seconds + number_of_timesteps\
        #     * simulation_time_step_seconds
        # time_array = np.arange(initial_time_seconds, final_time_seconds,
        #                        simulation_time_step_seconds)
        time_var[:] = self.time #time_array

        # time_bnds
        time_bnds_var = data_out_nc.createVariable('time_bnds', 'i4',
                                                   ('time', 'nv',))
        for time_index, time_element in enumerate(self.time): #time_array):
            time_bnds_var[time_index, 0] = time_element
            time_bnds_var[time_index, 1] = \
                time_element + output_time_step_seconds

        # TODO: for larger files, it may make sense to omit the latitude
        # and longitude variables or to store with lower precision.
        
        # longitude
        lon_var = data_out_nc.createVariable('lon', 'f8', ('rivid',),
                                             fill_value=-9999.0)
        lon_var.long_name = \
            'longitude of a point related to each river reach'
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var.axis = 'X'

        # latitude
        lat_var = data_out_nc.createVariable('lat', 'f8', ('rivid',),
                                             fill_value=-9999.0)
        lat_var.long_name = \
            'latitude of a point related to each river reach'
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var.axis = 'Y'

        crs_var = data_out_nc.createVariable('crs', 'i4')
        crs_var.grid_mapping_name = 'latitude_longitude'
        crs_var.epsg_code = 'EPSG:4326'  # WGS 84
        crs_var.semi_major_axis = 6378137.0
        crs_var.inverse_flattening = 298.257223563

        # add global attributes
        # data_out_nc.Conventions = 'CF-1.6'
        data_out_nc.title = 'RAPID Inflow from {0}'.format(
            self.land_surface_model_description)
        data_out_nc.history = 'date_created: {0}'.format(
            datetime.utcnow())
        # data_out_nc.featureType = 'timeSeries'
        #data_out_nc.institution = modeling_institution

        # write lat lon data
        #self._write_lat_lon(data_out_nc, in_rivid_lat_lon_z_file)

        # close file
        data_out_nc.close()  
        
    def read_weight_table(self):
        """
        Read the weight table file.
        """
        weight_table = np.genfromtxt(self.weight_table_file, delimiter=',',
                                     skip_header=1)

        self.weight_rivid = weight_table[:,0].astype(int)
        self.weight_area = weight_table[:,1]
        self.weight_lat_indices = weight_table[:,3].astype(int)
        self.weight_lon_indices = weight_table[:,2].astype(int)
        self.weight_id = weight_table[:,7].astype(int)

        # Check if the weight table contains any invalid values. If it does,
        # assign placeholder values to prevent downstream invalid value
        # errors. 
        valid = self.weight_lat_indices != self.invalid_value

        dummy_lat_index = self.weight_lat_indices[valid][0]
        dummy_lon_index = self.weight_lon_indices[valid][0]

        self.weight_lat_indices[~valid] = dummy_lat_index
        self.weight_lon_indices[~valid] = dummy_lon_index
        
        self.rivid = np.unique(self.weight_rivid)
        
    def find_rivid_weight_indices(self):
        """
        Given the ordered array of unique identifiers, `self.rivid`, 
        create a list whose elements are the indices in the weight table 
        corresponding to those identifiers. The nth element in the list
        will contain the indices corresponding to the nth element in 
        `self.rivid`.
        """
        self.rivid_weight_indices = []
        for rivid in self.rivid:
            rivid_weight_idx = np.where(self.weight_rivid == rivid)[0]
            self.rivid_weight_indices.append(rivid_weight_idx)

    def find_lat_lon_weight_indices(self):
        """
        Create an array of unique input lat/lon indices 
        `self.lat_lon_indices`. Then create an array, 
        `self.lat_lon_weight_indices` with the same size as the number of 
        rows in the weight table. Finally, populate the nth element of 
        `self.lat_lon_weight_indices` with the index in 
        `self.lat_lon_indices` corresponding to the nth weight-table entry.
        """
        self.weight_lat_lon_indices = np.column_stack(
            [self.weight_lat_indices, self.weight_lon_indices])

        self.lat_lon_indices = np.unique(self.weight_lat_lon_indices, axis=0)
        self.lat_indices = self.lat_lon_indices[:,0]
        self.lon_indices = self.lat_lon_indices[:,1]

        self.lat_lon_weight_indices = np.zeros(
            len(self.weight_lat_lon_indices),dtype=int)
        
        for idx, lat_lon_idx in enumerate(self.lat_lon_indices):
            lat_lon_weight_idx = (
                (self.weight_lat_lon_indices == lat_lon_idx).all(axis=1))
            self.lat_lon_weight_indices[lat_lon_weight_idx] = idx

    def find_lat_lon_input_indices(self):
        """
        Identify the largest and smallest latitude and longitude indices to 
        be extracted from the input files and determine array slices that 
        comprise all of the indices that lie within these bounds.
        """
        self.min_lat_index = self.weight_lat_indices.min()
        self.max_lat_index = self.weight_lat_indices.max()
        self.min_lon_index = self.weight_lon_indices.min()
        self.max_lon_index = self.weight_lon_indices.max()

        self.lat_slice = slice(self.min_lat_index, self.max_lat_index + 1)
        self.lon_slice = slice(self.min_lon_index, self.max_lon_index + 1)

        self.n_lat_slice = self.lat_slice.stop - self.lat_slice.start
        self.n_lon_slice = self.lon_slice.stop - self.lon_slice.start
        
    def find_subset_indices(self):
        """
        Determine a new set of indices that correspond to only those spatial
        locations in the output file that are represented in the weight 
        table. This array is structured to conform to the shape of the 
        "flattened" input array. i.e. it provides the indices of the 
        relevant spatial locations after the dimensions of the input runoff 
        array have been changed from (time, lat, lon) to (time, lat/lon).
        """
        self.subset_indices = (
            (self.lat_indices - self.min_lat_index)*self.n_lon_slice +
            (self.lon_indices - self.min_lon_index))
        
    def write_multiprocessing_job_list(self):
        """
        Write a list of dictionaries, each of which contains information 
        required to process a single input file.
        """
        self.job_list = []

        mp_lock = multiprocessing.Manager().Lock()
        
        for idx, f in enumerate(self.input_file_array):
            args = {}
            args['input_filename'] = f
            args['output_indices'] = self.output_indices[idx]
            args['mp_lock'] = mp_lock
            self.job_list.append(args)

    def read_write_inflow(self, args):
        """
        Extract timeseries data from one netCDF file and write it to another
        at the appropriately indexed locations.

        Parameters
        ----------
        args : dict
            File/process-specific parameters
        """
        input_filename = args['input_filename']
        start_idx, end_idx = args['output_indices']
        mp_lock = args['mp_lock']
        
        data_in = Dataset(input_filename)

        steps = np.arange(self.steps_per_input_file)

        input_runoff = np.zeros([self.steps_per_input_file,
                                 self.n_lat_slice,
                                 self.n_lon_slice])

        # Sum values over all specified runoff variables for the region
        # indicated by `lat_slice` and `lon_slice`.
        for runoff_key in self.runoff_variable_names:
            input_runoff += data_in[runoff_key][:,self.lat_slice,
                                                self.lon_slice]

        data_in.close()

        # Reshape the input runoff array so that the first dimension
        # corresponds to time and the second dimension corresponds to
        # geospatial coordinates. This reduces the number of dimensions from
        # three (e.g time, lat, lon) to two (e.g. time, latlon).
        input_runoff = input_runoff.reshape(
            self.steps_per_input_file,self.n_lon_slice*self.n_lat_slice)
               
        # Extract only values that correspond to unique locations represented
        # in the weight table. `subset_indices` provides the indices in the
        # spatial dimension of `input_runoff` that correspond to these unique
        # locations.
        input_runoff  = input_runoff[:,self.subset_indices]
        
        # Convert the runoff from its native units to meters.
        input_runoff_meters = input_runoff * self.meters_per_input_runoff_unit

        # Redistribute runoff values at unique spatial coordinates to all
        # weight table locations (indexed by latitude, longitude, and
        # catchment id).
        input_runoff_meters = input_runoff_meters[
            :,self.lat_lon_weight_indices]

        # Convert runoff in [m^2] to [m^3] by multiplying input runoff by areas
        # provided by the weight table.
        weighted_runoff_m3 = input_runoff_meters * self.weight_area

        accumulated_runoff_m3 = np.zeros([self.steps_per_input_file,
                                          len(self.rivid)])

        # For each catchment ID, sum runoff [m^3] over each region within the
        # associated catchment.
        for idx, rivid_weight_idx in enumerate(self.rivid_weight_indices):
            summed_by_rivid = np.sum(weighted_runoff_m3[:,rivid_weight_idx])
            accumulated_runoff_m3[:,idx] = summed_by_rivid
            # print('idx, accumulated_runoff_m3')
            # print(idx, accumulated_runoff_m3)

        if self.convert_one_hour_to_three:
            accumulated_runoff_m3 = sum_over_time_increment(
                accumulated_runoff_m3, 1, 3, self.steps_per_input_file)

        # Write the accumulated runoff [m^3] to the output file at the
        # appropriate indices along the time dimension. Use a multiprocessing
        # lock to prevent more than one process writing to the file at a time.

        # MPG: `mp_lock` is now generated as one of the items in the job list
        # (see above). It is no longer an instance attribute.
        #self.mp_lock.acquire()
        
        mp_lock.acquire()
        
        data_out = Dataset(self.output_filename, "a")
        data_out['m3_riv'][start_idx:end_idx,:] = accumulated_runoff_m3
        data_out.close()

        # MPG: See comment above regarding `mp_lock`.
        #self.mp_lock.release()
        mp_lock.release()
        
    def generate_inflow_file(self):
        """
        The main routine for the InflowAccumulator class.
        """
        self.read_weight_table()

        self.find_rivid_weight_indices()

        self.find_lat_lon_weight_indices()

        self.find_lat_lon_input_indices()
        
        self.find_subset_indices()
        
        self.generate_input_runoff_file_array()

        self.determine_output_indices()

        self.concatenate_time_variable()
        
        self.initialize_inflow_nc()
        
        self.write_multiprocessing_job_list()
        
        self.mp_lock = multiprocessing.Manager().Lock()

        # DEBUG:
        # self.read_write_inflow(self.job_list[0])
        
        pool = multiprocessing.Pool(self.nproc)
        
        pool.map(self.read_write_inflow, self.job_list)

if __name__=='__main__':
    output_filename = 'inflow_check.nc'
    input_runoff_file_directory = 'tests/data/lsm_grids/gldas2'
    weight_table_file = 'tests/data/weight_table/weight_gldas2.csv'
    runoff_variable_names = ['Qs_acc', 'Qsb_acc']
    M3_PER_KG = 0.001
    land_surface_model_description = 'GLDAS2'
    file_datetime_format = '%Y%m%d.%H'
    file_timestamp_re_pattern = r'\d{8}.\d{2}'
    input_runoff_file_ext = 'nc4'
    start_datetime = datetime(2010, 12, 31)
    end_datetime = datetime(2010, 12, 31, 3)
    output_time_step_hours = 3
    steps_per_input_file = 1
    convert_one_hour_to_three = False
    nproc = 2
    
    inflow_accumulator = InflowAccumulator(
        output_filename,
        input_runoff_file_directory,
        steps_per_input_file,
        weight_table_file,
        runoff_variable_names,
        meters_per_input_runoff_unit=M3_PER_KG,
        land_surface_model_description=land_surface_model_description,
        input_time_step_hours=None,
        output_time_step_hours=output_time_step_hours,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        file_datetime_format=file_datetime_format,
        file_timestamp_re_pattern=file_timestamp_re_pattern,
        input_runoff_file_ext=input_runoff_file_ext,
        nproc=nproc,
        output_time_units='seconds since 1970-01-01 00:00:00',
        convert_one_hour_to_three=convert_one_hour_to_three)

    inflow_accumulator.generate_inflow_file()

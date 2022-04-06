"""
Tools to generate lateral inflow for a routing model given land surface model 
(LSM) files containing one or more runoff variables and a weight table that
assigns areas of intersection with LSM grid cells to catchments. 
"""
from glob import glob
import numpy as np
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import sys
import os
import re

SECONDS_PER_HOUR = 3600
M3_PER_KG = 0.001

def parse_time_from_nc(filename):
    d = Dataset(filename)
    time_variable = d['time']
    units = time_variable.units
    time = time_variable[:]

    return time, units

def parse_timestamp_from_filename(filename, re_search_pattern=r'\d{8}',
                                  datetime_pattern='%Y%m%d'):
    
    match = re.search(re_search_pattern, filename)
    datestr = match.group()
    dt = datetime.strptime(datestr, datetime_pattern)

    return dt

def convert_time(in_datenum_array, input_units, output_units):
    datetime_array = num2date(in_datenum_array, input_units)
    out_datenum_array = date2num(datetime_array, output_units)

    return out_datenum_array

def sum_over_time_increment(data, old_timestep_hours,
                            new_timestep_hours, steps_per_file):
    """
    Sum over old_timestep_hours-hourly timesteps so that inflow data
    time dimension reflects new_timestep_hours-hourly timestep.
    """
    file_time_hours = steps_per_file * old_timestep_hours
    
    new_time_dim = int(file_time_hours / new_timestep_hours)

    # We add a new dimension, tmp_dim, to sum over.
    tmp_dim = int(new_timestep_hours)
    data = data.reshape(new_time_dim, tmp_dim, -1)
    data = data.sum(axis=1)

    return inflow_data
    
def read_write_inflow(args):
    input_filename = args['input_filename']
    output_filename = args['output_filename']
    steps_per_file = args['steps_per_file']
    nrivid = args['nrivid']
    nweight = args['nweight']
    weight_area = args['weight_area']
    weight_rivid = args['weight_rivid']
    unique_rivid = args['unique_rivid']
    lat_indices = args['lat_indices']
    lon_indices = args['lon_indices']
    runoff_variables = args['runoff_variables']
    meters_per_input_unit = args['meters_per_input_unit']
    convert_one_hour_to_three = args['convert_one_hour_to_three']
    rivid_to_weight_index_dict = args['rivid_to_weight_index_dict']
    
    data_in = Dataset(input_filename)

    steps = np.arange(steps_per_file)
    
    input_runoff = np.zeros([steps_per_file, nrivid])
    for runoff_key in runoff_variables:
        for step in steps:
            runoff_step = data_in[runoff_key][step][lat_indices, lon_indices]
            input_runoff[step] += runoff_step

    input_runoff_meters = input_runoff * meters_per_input_unit
    
    weighted_runoff_meters = np.zeros([steps_per_file, nweight])
    for rivid_idx, weight_idx in rivid_to_weight_index_dict.items():
        weighted_runoff_meters[:,weight_idx] = input_runoff_meters[:,rivid_idx]

    weighted_runoff_m3 = weighted_runoff_meters * weight_area

    accumulated_runoff_m3 = np.zeros([steps_per_file, nrivid])
    
    for rivid_idx, weight_idx in rivid_to_weight_index_dict.items():
        summed_by_rivid = np.sum(weighted_runoff_m3[:,weight_idx], axis=1)
        accumulated_runoff_m3[:,rivid_idx] = summed_by_rivid

    if convert_one_hour_to_three:
        sum_over_time_increment(weight_runoff, 1, 3, steps_per_file)

    mp.lock.acquire()
    data_out
class InflowAccumulator:
    
    def __init__(self,
                 output_filename,
                 input_file_directory,
                 steps_per_input_file,
                 weight_table_file,
                 input_time_step_hours=None,
                 output_time_step_hours=None,
                 start_datetime=None,
                 end_datetime=None,
                 file_datetime_format='%Y%m%d',
                 file_timestamp_re_pattern=r'\d{8}',
                 input_file_ext='nc',
                 nproc=1,
                 land_surface_model_description=None,
                 input_time_units=None,
                 output_time_units='seconds since 1970-01-01 00:00:00',
                 invalid_value=-9999,
                 convert_one_hour_to_three=False):

        self.output_filename = output_filename
        self.input_file_directory = input_file_directory
        self.steps_per_input_file = steps_per_input_file
        self.weight_table_file = weight_table_file
        self.input_time_step_hours = input_time_step_hours
        self.output_time_step_hours = output_time_step_hours
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.file_datetime_format = file_datetime_format
        self.file_timestamp_re_pattern = file_timestamp_re_pattern
        self.input_file_ext = input_file_ext
        self.nproc = nproc
        self.land_surface_model_description = land_surface_model_description
        self.input_time_units = input_time_units
        self.output_time_units = output_time_units
        self.invalid_value = invalid_value
        self.input_file_array = None
        self.input_time_variable = None
        self.rivid = None
        self.convert_one_hour_to_three = convert_one_hour_to_three
        
        self.time = None
        self.weight_lat_indices = None
        self.weight_lon_indices = None
        
    def generate_input_file_array(self):

        input_file_expr = os.path.join(
            self.input_file_directory, f'*.{self.input_file_ext}')
        input_file_list = glob(input_file_expr)
        input_file_array = np.array(input_file_list)
        input_timestamp_list = [parse_timestamp_from_filename(
            f, file_timestamp_re_pattern, file_datetime_format) for f in
                          input_file_list]
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

    def concatenate_time_variable(self):

        nfiles = len(self.input_file_array)
        n_time_step = nfiles * self.steps_per_input_file
    
        time = np.zeros(n_time_step)

        start_idx = 0
        for f in self.input_file_array:
            end_idx = start_idx + self.steps_per_input_file
            file_time, units = parse_time_from_nc(f)
            converted_time = convert_time(file_time, units,
                                          self.output_time_units) 
            time[start_idx:end_idx] = converted_time[:]
            start_idx += self.steps_per_input_file

        self.time = time

    def initialize_inflow_nc(self):

        output_time_step_seconds = (
            self.output_time_step_hours * SECONDS_PER_HOUR)
                
        data_out_nc = Dataset(self.output_filename, 'w')

        # create dimensions
        data_out_nc.createDimension('time', self.n_time_step)
        data_out_nc.createDimension('rivid', self.nrivid)
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
        weight_table = np.genfromtxt(self.weight_table_file, delimiter=',',
                                     skip_header=1)

        self.weight_rivid = weight_table[:,0]
        self.weight_area = weight_table[:,1]
        self.weight_lat_indices = weight_table[:,3].astype(int)
        self.weight_lon_indices = weight_table[:,2].astype(int)
        self.weight_id = weight_table[:,7].astype(int)
        #idx = np.argsort(weight_id)
                        
        valid = self.weight_lat_indices != self.invalid_value

        dummy_lat_index = self.weight_lat_indices[valid][0]
        dummy_lon_index = self.weight_lon_indices[valid][0]

        self.weight_lat_indices[~valid] = dummy_lat_index
        self.weight_lon_indices[~valid] = dummy_lon_index

    def find_weight_indices(self):
        self.rivid_to_weight_index_dict = {}
        for rivid_idx, rivid in enumerate(self.rivid):
            weight_idx = np.where(self.weight_rivid == rivid)[0]
            self.rivid_to_weight_index_dict[rivid_idx] = weight_idx
            
    def generate_inflow_file(self):

        self.read_weight_table()

        self.nweight = len(self.weight_id)
        
        # np.unique returns unique entries sorted in lexicographic order.
        self.rivid, self.unique_weight_indices = np.unique(
            self.weight_rivid, return_index=True)

        # print(self.rivid)
        # print(self.unique_weight_indices)
        # sys.exit(0)
        self.lat_indices = self.weight_lat_indices[self.unique_weight_indices]
        self.lon_indices = self.weight_lon_indices[self.unique_weight_indices]
        self.area = self.weight_area[self.unique_weight_indices]
        
        self.nrivid = len(self.rivid)
        
        self.generate_input_file_array()

        self.concatenate_time_variable()

        self.n_time_step = len(self.time)

        self.find_weight_indices()
        
        self.initialize_inflow_nc()
        
        args = {}
        args['output_filename'] = self.output_filename
        args['steps_per_file'] = self.steps_per_input_file
        args['nrivid'] = self.nrivid
        args['nweight'] = self.nweight
        args['weight_area'] = self.weight_area
        args['weight_rivid'] = self.weight_rivid
        args['unique_rivid'] = self.rivid
        args['lat_indices'] = self.lat_indices
        args['lon_indices'] = self.lon_indices
        args['runoff_variables'] = ['Qs_acc', 'Qsb_acc']
        args['meters_per_input_unit'] = M3_PER_KG
        args['convert_one_hour_to_three'] = self.convert_one_hour_to_three
        args['rivid_to_weight_index_dict'] = self.rivid_to_weight_index_dict

        for f in self.input_file_array:
            args['input_filename'] = 'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0000.020.nc4'
            read_write_inflow(args)

if __name__=='__main__':
    output_filename = 'inflow_check.nc'
    input_file_directory = 'tests/data/lsm_grids/gldas2'
    #rivid_file = 'tests/data/connectivity/rapid_connect_saguache.csv'
    weight_table_file = 'tests/data/weight_table/weight_gldas2.csv'
    file_datetime_format = '%Y%m%d.%H'
    file_timestamp_re_pattern = r'\d{8}.\d{2}'
    input_file_ext = 'nc4'
    start_datetime = datetime(2010, 12, 31)
    end_datetime = datetime(2010, 12, 31, 3)
    output_time_step_hours = 3
    steps_per_input_file = 1
    convert_one_hour_to_three = False
    
    inflow_accumulator = InflowAccumulator(
        output_filename=output_filename,
        input_file_directory=input_file_directory,
        steps_per_input_file=steps_per_input_file,
        weight_table_file=weight_table_file,
        input_time_step_hours=None,
        output_time_step_hours=output_time_step_hours,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        file_datetime_format=file_datetime_format,
        file_timestamp_re_pattern=file_timestamp_re_pattern,
        input_file_ext=input_file_ext,
        nproc=1,
        land_surface_model_description=None,
        input_time_units=None,
        output_time_units='seconds since 1970-01-01 00:00:00',
        convert_one_hour_to_three=convert_one_hour_to_three)

    inflow_accumulator.generate_inflow_file()
                         

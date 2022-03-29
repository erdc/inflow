"""
Tools to generate lateral inflow for a routing model given land surface model 
(LSM) files containing one or more runoff variables and a weight table that
assigns areas of intersection with LSM grid cells to catchments. 
"""
from glob import glob
import numpy as np
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import os
import re

SECONDS_PER_HOUR = 3600

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

def read_write_inflow(args):
    input_filename = args[0]
    steps_per_file = args[1]
    n_rivid = args[2]
    lat_indices = args[3]
    lon_indices = args[4]
    runoff_variables = args[5]

    total_runoff = np.zeros([steps_per_file, n_rivid])
    
    d = Dataset(input_filename)
    for runoff_var in runoff_variables:
        print(runoff)
        for idx in range(steps_per_file):
            print('idx', idx)
            print('inflow', inflow[idx].shape)
            print('runoff', d[runoff].shape)
            print('runoff[idx]', d[runoff][idx].shape)
            print('runoff[idx][lat_indices, lon_indices]',
              d[runoff_var][idx][lat_indices, lon_indices].shape)
            
            runoff[idx] += d[runoff_var][idx][lat_indices, lon_indices]
    
    print(inflow)
    
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
                 invalid_value=-9999):

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

        self.time = None
        #self.input_indices = None
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

    def initialize_lateral_inflow_nc(self):

        output_time_step_seconds = (
            self.output_time_step_hours * SECONDS_PER_HOUR)
                
        data_out_nc = Dataset(self.output_filename, 'w')

        # create dimensions
        data_out_nc.createDimension('time', self.n_time_step)
        data_out_nc.createDimension('rivid', self.n_rivid)
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

    # def read_write_inflow(args):
    #     input_file = args[0]
    #     weight_table_file = args[1]
    #     output_file = args[2]
    #     mp_lock = args[3]
    #     steps_per_file = args[4]
    #     convert_one_hour_to_three = args[5]        
        
    def read_weight_table(self):
        weight_table = np.genfromtxt(self.weight_table_file, delimiter=',',
                                     skip_header=1)

        rivid = weight_table[:,0]
        weight_area = weight_table[:,1]
        weight_lat_indices = weight_table[:,3].astype(int)
        weight_lon_indices = weight_table[:,2].astype(int)
        
        valid = weight_lat_indices != self.invalid_value

        dummy_lat_index = weight_lat_indices[valid][0]
        dummy_lon_index = weight_lon_indices[valid][0]

        weight_lat_indices[~valid] = dummy_lat_index
        weight_lon_indices[~valid] = dummy_lon_index

        self.rivid = np.unique(rivid)
        self.weight_area = weight_area
        self.weight_lat_indices = weight_lat_indices
        self.weight_lon_indices = weight_lon_indices
        
    def generate_inflow_file(self):

        self.read_weight_table()

        self.n_rivid = len(self.rivid)
                           
        self.generate_input_file_array()

        self.concatenate_time_variable()

        self.n_time_step = len(self.time)
            
        self.initialize_lateral_inflow_nc()

        args = (
            'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
            self.n_time_step,
            self.n_rivid,
            self.weight_lat_indices,
            self.weight_lon_indices,
            ['Qs_acc', 'Qsb_acc'])
            
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
        output_time_units='seconds since 1970-01-01 00:00:00')

    inflow_accumulator.generate_inflow_file()
                         

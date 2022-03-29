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
                 
def parse_time_from_input_nc(filename):
    d = Dataset(filename)
    time_variable = d['time']
    units = time_variable.units
    time = time_variable[:]

    return time, units

def convert_time(time, input_units,
                 output_units='seconds since 1970-01-01 00:00:00'):
    in_datetime_array = num2date(time, input_units)
    out_datenum_array = date2num(in_datetime_array, output_units)

    return out_datenum_array, output_units

def concatenate_input_time_variable(flist, steps_per_file):
    nfiles = len(flist)
    n_time_step = nfiles * steps_per_file
    
    time = np.zeros(n_time_step)

    start_idx = 0
    for f in flist:
        end_idx = start_idx + steps_per_file
        file_time, units = parse_time_from_input_nc(f)
        time[start_idx:end_idx] = file_time[:]
        start_idx += steps_per_file

    return time, units
        
def initialize_lateral_inflow_nc(filename, rivid_list, time, time_units,
                                 simulation_time_step_seconds,
                                 land_surface_model_description):
    number_of_timesteps = len(time)
    
    data_out_nc = Dataset(filename, 'w')

    # create dimensions
    data_out_nc.createDimension('time', number_of_timesteps)
    data_out_nc.createDimension('rivid', len(rivid_list))
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
    rivid_var[:] = rivid_list

    # time
    time_var = data_out_nc.createVariable('time', 'i4', ('time',))
    time_var.long_name = 'time'
    time_var.standard_name = 'time'
    time_var.units = time_units
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
    time_var[:] = time #time_array

    # time_bnds
    time_bnds_var = data_out_nc.createVariable('time_bnds', 'i4',
                                               ('time', 'nv',))
    for time_index, time_element in enumerate(time): #time_array):
        time_bnds_var[time_index, 0] = time_element
        time_bnds_var[time_index, 1] = \
            time_element + simulation_time_step_seconds

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
        land_surface_model_description)
    data_out_nc.history = 'date_created: {0}'.format(
        datetime.utcnow())
    # data_out_nc.featureType = 'timeSeries'
    #data_out_nc.institution = modeling_institution

    # write lat lon data
    #self._write_lat_lon(data_out_nc, in_rivid_lat_lon_z_file)

    # close file
    data_out_nc.close()
            
def execute(self, nc_file_list, index_list, in_weight_table, out_nc, grid_type,
            mp_lock, steps_per_file=1, convert_one_hour_to_three=False):

    demo_file_list = nc_file_list[0]
    if not isinstance(nc_file_list[0], list):
        demo_file_list = [demo_file_list]

    self.data_validation(demo_file_list[0])
    self.read_in_weight_table(in_weight_table)

    conversion_factor = self.get_conversion_factor(demo_file_list[0],
                                                   len(demo_file_list))

    # get indices of subset of data
    lon_ind_all = [int(i) for i in self.dict_list[self.header_wt[2]]]
    lat_ind_all = [int(j) for j in self.dict_list[self.header_wt[3]]]

    # Obtain a subset of  runoff data based on the indices in the
    # weight table
    min_lon_ind_all = min(lon_ind_all)
    max_lon_ind_all = max(lon_ind_all)
    min_lat_ind_all = min(lat_ind_all)
    max_lat_ind_all = max(lat_ind_all)
    lon_slice = slice(min_lon_ind_all, max_lon_ind_all + 1)
    lat_slice = slice(min_lat_ind_all, max_lat_ind_all + 1)
    index_new = []

    # combine inflow data
    for nc_file_array_index, nc_file_array in enumerate(nc_file_list):

        index = index_list[nc_file_array_index]

        if not isinstance(nc_file_array, list):
            nc_file_array = [nc_file_array]

        data_subset_all = None
        for nc_file in nc_file_array:
            # Validate the netcdf dataset
            self.data_validation(nc_file)

            # Read the netcdf dataset
            data_in_nc = Dataset(nc_file)

            # Calculate water inflows
            runoff_dimension_size = \
                len(data_in_nc.variables[self.runoff_vars[0]].dimensions)
            if runoff_dimension_size == 2:
                # obtain subset of surface and subsurface runoff
                data_subset_runoff = \
                    data_in_nc.variables[self.runoff_vars[0]][
                        lat_slice, lon_slice]
                for var_name in self.runoff_vars[1:]:
                    data_subset_runoff += \
                        data_in_nc.variables[var_name][
                            lat_slice, lon_slice]

                # get runoff dims
                len_time_subset = 1
                len_lat_subset = data_subset_runoff.shape[0]
                len_lon_subset = data_subset_runoff.shape[1]

                # reshape the runoff
                data_subset_runoff = data_subset_runoff.reshape(
                    len_lat_subset * len_lon_subset)

            elif runoff_dimension_size == 3:
                # obtain subset of surface and subsurface runoff
                data_subset_runoff = \
                    data_in_nc.variables[self.runoff_vars[0]][
                        :, lat_slice, lon_slice]
                for var_name in self.runoff_vars[1:]:
                    data_subset_runoff += \
                        data_in_nc.variables[var_name][
                            :, lat_slice, lon_slice]

                # get runoff dims
                len_time_subset = data_subset_runoff.shape[0]
                len_lat_subset = data_subset_runoff.shape[1]
                len_lon_subset = data_subset_runoff.shape[2]
                # reshape the runoff
                data_subset_runoff = \
                    data_subset_runoff.reshape(
                        len_time_subset,
                        (len_lat_subset * len_lon_subset))

            data_in_nc.close()

            if not index_new:
                # compute new indices based on the data_subset_surface
                for r in range(0, self.count):
                    ind_lat_orig = lat_ind_all[r]
                    ind_lon_orig = lon_ind_all[r]
                    index_new.append(
                        (ind_lat_orig - min_lat_ind_all) * len_lon_subset
                        + (ind_lon_orig - min_lon_ind_all))

            # obtain a new subset of data
            if runoff_dimension_size == 2:
                data_subset_new = data_subset_runoff[index_new]
            elif runoff_dimension_size == 3:
                data_subset_new = data_subset_runoff[:, index_new]

            # FILTER DATA
            try:
                # set masked values to zero
                data_subset_new = data_subset_new.filled(fill_value=0)
            except AttributeError:
                pass
            # set negative values to zero
            data_subset_new[data_subset_new < 0] = 0

            # combine data
            if data_subset_all is None:
                data_subset_all = data_subset_new
            else:
                data_subset_all = np.add(data_subset_all, data_subset_new)

        if runoff_dimension_size == 3 and len_time_subset > 1:
            inflow_data = np.zeros((len_time_subset, self.size_stream_id))
        else:
            inflow_data = np.zeros(self.size_stream_id)

        pointer = 0
        for stream_index in xrange(self.size_stream_id):
            npoints = int(self.dict_list[self.header_wt[4]][pointer])
            # Check if all npoints points correspond to the same streamID
            if len(set(self.dict_list[self.header_wt[0]][
                       pointer: (pointer + npoints)])) != 1:
                print("ROW INDEX {0}".format(pointer))
                print("COMID {0}".format(
                    self.dict_list[self.header_wt[0]][pointer]))
                raise Exception(self.error_messages[2])

            area_sqm_npoints = \
                np.array([float(k) for k in
                          self.dict_list[self.header_wt[1]][
                          pointer: (pointer + npoints)]])

            # assume data is incremental
            if runoff_dimension_size == 3:
                data_goal = data_subset_all[:, pointer:(pointer + npoints)]
            else:
                data_goal = data_subset_all[pointer:(pointer + npoints)]

            if grid_type == 't255':
                # A) ERA Interim Low Res (T255) - data is cumulative
                # from time 3/6/9/12
                # (time zero not included, so assumed to be zero)
                ro_first_half = \
                    np.concatenate([data_goal[0:1, ],
                                    np.subtract(data_goal[1:4, ],
                                                data_goal[0:3, ])])
                # from time 15/18/21/24
                # (time restarts at time 12, assumed to be zero)
                ro_second_half = \
                    np.concatenate([data_goal[4:5, ],
                                    np.subtract(data_goal[5:, ],
                                                data_goal[4:7, ])])
                ro_stream = \
                    np.multiply(
                        np.concatenate([ro_first_half, ro_second_half]),
                        area_sqm_npoints)

            elif grid_type == 't1279':
                # A) ERA Interim Low Res (T1279) - data is cumulative
                # from time 6/12
                # 0 1 2 3 4
                # (time zero not included, so assumed to be zero)
                ro_first_half = \
                    np.concatenate([data_goal[0:1, ],
                                    np.subtract(data_goal[1:2, ],
                                                data_goal[0:1, ])])
                # from time 15/18/21/24
                # (time restarts at time 12, assumed to be zero)
                ro_second_half = \
                    np.concatenate([data_goal[2:3, ],
                                    np.subtract(data_goal[3:, ],
                                                data_goal[2:3, ])])
                ro_stream = \
                    np.multiply(
                        np.concatenate([ro_first_half, ro_second_half]),
                        area_sqm_npoints)

            else:
                ro_stream = data_goal * area_sqm_npoints * \
                            conversion_factor

            # filter nan
            ro_stream[np.isnan(ro_stream)] = 0

            if ro_stream.any():
                if runoff_dimension_size == 3 and len_time_subset > 1:
                    inflow_data[:, stream_index] = ro_stream.sum(axis=1)
                else:
                    inflow_data[stream_index] = ro_stream.sum()

            pointer += npoints

        if convert_one_hour_to_three:
           inflow_data = self.sum_inflow_over_time_increment(
               inflow_data, 1, 3, steps_per_file)
           len_time_subset /= 3

        start_idx = int(index*len_time_subset)
        end_idx = int((index+1)*len_time_subset)   
        # only one process is allowed to write at a time to netcdf file
        mp_lock.acquire()
        data_out_nc = Dataset(out_nc, "a")
        if runoff_dimension_size == 3 and len_time_subset > 1:
            data_out_nc.variables['m3_riv'][
                start_idx:end_idx, :] = inflow_data
        else:
            data_out_nc.variables['m3_riv'][index] = inflow_data
        data_out_nc.close()
        mp_lock.release()

def generate_inflows_from_runoff(args):
    """
    Multiprocessing function.
    """
    runoff_file_list = args[0]
    file_index_list = args[1]
    
def parse_timestamp_from_filename(filename, re_search_pattern=r'\d{8}',
                                  datetime_pattern='%Y%m%d'):
    
    match = re.search(re_search_pattern, filename)
    datestr = match.group()
    dt = datetime.strptime(datestr, datetime_pattern)

    return dt

def generate_input_file_array(input_file_directory, input_file_ext,
                              file_timestamp_re_pattern=r'\d{8}',
                              file_datetime_format='%Y%m%d',
                              simulation_start_datetime=None,
                              simulation_end_datetime=None):

    input_file_expr = os.path.join(
        input_file_directory, f'*.{input_file_ext}')
    lsm_file_list = glob(input_file_expr)
    lsm_file_array = np.array(lsm_file_list)
    lsm_timestamp_list = [parse_timestamp_from_filename(
        f, file_timestamp_re_pattern, file_datetime_format) for f in
                      lsm_file_list]
    lsm_timestamp_array = np.array(lsm_timestamp_list)
    sorted_by_time = lsm_timestamp_array.argsort()
    lsm_file_array = lsm_file_array[sorted_by_time]
    lsm_timestamp_array = lsm_timestamp_array[sorted_by_time]

    in_time_bounds = np.ones_like(lsm_timestamp_array, dtype=bool)
    
    if simulation_start_datetime is not None:
        on_or_after_start = lsm_timestamp_array >= simulation_start_datetime
        in_time_bounds = np.logical_and(in_time_bounds, on_or_after_start)

    if simulation_end_datetime is not None:
        on_or_before_end = lsm_timestamp_array <= simulation_end_datetime
        in_time_bounds = np.logical_and(in_time_bounds, on_or_before_end)

    lsm_file_array = lsm_file_array[in_time_bounds]

    return lsm_file_array

def generate_inflow_file(filename,
                         input_file_directory,
                         steps_per_file,
                         simulation_time_step_hours,
                         rivid_file,
                         weight_table_file=None,
                         input_time_step_hours=None,
                         output_time_step_hours=None,
                         simulation_start_datetime=None,
                         simulation_end_datetime=None,
                         file_datetime_format='%Y%m%d',
                         file_timestamp_re_pattern=r'\d{8}',
                         input_file_ext='nc',
                         num_processors=1,
                         land_surface_model_description=None):

    rivid_array = np.genfromtxt(rivid_file, delimiter=',', usecols=0)
    
    lsm_file_array = generate_inflow_file_array(
        input_file_directory, input_file_ext,
        file_timestamp_re_pattern=r'\d{8}',
        file_datetime_format='%Y%m%d',
        simulation_start_datetime=None,
        simulation_end_datetime=None)

    input_time, input_time_units = concatenate_input_time_variable(
        lsm_file_array, steps_per_file)

    time, time_units = convert_time(input_time, input_time_units)

    seconds_per_hour = 3600
    
    simulation_time_step_seconds = (
        simulation_time_step_hours * seconds_per_hour)
    
    initialize_lateral_inflow_nc(filename, rivid_array, time, time_units,
                                 simulation_time_step_seconds,
                                 land_surface_model_description)
    
    #return (lsm_file_array, lsm_timestamp_array)

if __name__=='__main__':
    filename = 'inflow_check.nc'
    input_file_directory = 'tests/data/lsm_grids/gldas2'
    rivid_file = 'tests/data/connectivity/rapid_connect_saguache.csv'
    file_datetime_format = '%Y%m%d.%H'
    file_timestamp_re_pattern = r'\d{8}.\d{2}'
    simulation_start_datetime = datetime(2010, 12, 31)
    simulation_end_datetime = datetime(2010, 12, 31, 3)
    simulation_time_step_hours = 24
    steps_per_file = 1
    print(os.listdir(input_file_directory))
    generate_inflow_file(filename,
                         input_file_directory,
                         steps_per_file,
                         simulation_time_step_hours,
                         rivid_file,
                         input_file_ext='nc4',
                         simulation_start_datetime=(
                             simulation_start_datetime),
                         simulation_end_datetime=(simulation_end_datetime),
                         file_datetime_format=(file_datetime_format),
                         file_timestamp_re_pattern=(
                             file_timestamp_re_pattern))
    

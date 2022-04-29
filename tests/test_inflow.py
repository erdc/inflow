"""
Test module for inflow.py.
"""
import numpy as np
from numpy import array, array_equal
import multiprocessing
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import os

from inflow import inflow
from inflow.inflow import InflowAccumulator

SECONDS_PER_HOUR = 3600
M3_PER_KG = 0.001

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TEST_DIR, 'data')
OUTPUT_DIR = os.path.join(TEST_DIR, 'output')
MAIN_DIR = os.path.join(TEST_DIR, os.pardir, 'inflow')

# Standalone functions

def test_parse_time_from_nc():
    """
    Verify that parse_time_from_nc correctly extracts time variable and units 
    from a netCDF file.
    """
    filename = os.path.join(DATA_DIR, 'lsm_grids', 'gldas2',
                            'GLDAS_NOAH025_3H.A20101231.0000.020.nc4')
    time, units = inflow.parse_time_from_nc(filename)
    
    expected_time = [33134220.]
    expected_units = ' minutes since 1948-01-01 03:00:00'
    
    assert time == expected_time
    assert units == expected_units

def test_parse_timestamp_from_filename():
    """
    Verify that parse_timestamp_from_filename correctly parses a timestamp from
    a filename and converts it to a datetime.datetime object.
    """
    filename = 'GLDAS_NOAH025_3H.A20101231.0000.020.nc4'
    datetime_pattern = '%Y%m%d.%H'
    re_search_pattern = r'\d{8}.\d{2}'

    result = inflow.parse_timestamp_from_filename(
        filename, re_search_pattern=re_search_pattern,
        datetime_pattern=datetime_pattern)
    
    expected = datetime(2010,12,31,0)

    assert result == expected
    
def test_convert_time():
    input_units = ' minutes since 1948-01-01 03:00:00'
    output_units = 'seconds since 1970-01-01 00:00:00'

    in_datenum_array = array([33134220.])
    out_datenum_array = inflow.convert_time(in_datenum_array, input_units,
                                            output_units)

    expected = array([1293753600])

    assert out_datenum_array == expected

def test_sum_over_time_increment():
    pass

# InflowAccumulator methods

def generate_default_inflow_accumulator_arguments():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_gldas2_check.nc')
    input_runoff_file_directory = os.path.join(DATA_DIR, 'lsm_grids', 'gldas2')
    steps_per_input_file = 1
    weight_table_file = os.path.join(DATA_DIR, 'weight_table',
                                     'weight_gldas2.csv')
    runoff_variable_names = ['Qs_acc', 'Qsb_acc']
    meters_per_input_runoff_unit = M3_PER_KG
    output_time_step_hours = 3
    land_surface_model_description = 'GLDAS2'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, output_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y%m%d.%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}.\d{2}'
    kwargs['input_runoff_file_ext'] = 'nc4'
    kwargs['start_datetime'] = datetime(2010, 12, 31)
    kwargs['end_datetime'] = datetime(2010, 12, 31, 3)
    kwargs['convert_one_hour_to_three'] = False
    kwargs['nproc'] = 2

    return (args, kwargs)

def test_initialize_inflow_accumulator():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)
    
def test_generate_input_runoff_file_array():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.generate_input_runoff_file_array()
    
    input_file_array = inflow_accumulator.input_file_array
    
    expected = array(
        [os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
         for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                   'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']])

    assert array_equal(input_file_array, expected)

def test_determine_output_indices():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    input_file_array = array(
        [os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
         for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                   'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']])

    inflow_accumulator.input_file_array = input_file_array
    
    inflow_accumulator.determine_output_indices()

    output_indices = inflow_accumulator.output_indices

    expected = [(0, 1), (1, 2)]

    assert output_indices == expected

def test_concatenate_time_variable():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.input_file_array = array(
        [os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
         for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                   'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']])

    inflow_accumulator.output_indices = [(0, 1), (1, 2)]

    inflow_accumulator.concatenate_time_variable()

    time = inflow_accumulator.time

    expected = array([1.2937536e+09, 1.2937644e+09])
    
    assert array_equal(time, expected)

def test_initialize_inflow_nc():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.input_file_array = array(
        [os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
         for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                   'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']])
    
    inflow_accumulator.output_indices = [(0, 1), (1, 2)]

    time = array([1.2937536e+09, 1.2937644e+09])

    inflow_accumulator.time = time

    output_filename = os.path.join(OUTPUT_DIR, 'gldas2_m3_init.nc')

    inflow_accumulator.output_filename = output_filename

    inflow_accumulator.n_time_step = 2

    rivid = array([17880258, 17880268, 17880282])

    inflow_accumulator.rivid = rivid
    
    inflow_accumulator.initialize_inflow_nc()

    d = Dataset(output_filename)

    keys = d.variables.keys()
    
    assert 'time' in keys
    assert 'rivid' in keys
    assert 'm3_riv' in keys
    
    assert d['time'].dimensions == ('time',)
    assert d['rivid'].dimensions == ('rivid',)
    assert d['m3_riv'].dimensions == ('time', 'rivid')
    
    assert d.dimensions['time'].size == len(time)
    assert d.dimensions['rivid'].size == len(rivid)

def test_read_weight_table():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.read_weight_table()

    weight_lat_indices = array([393, 393, 392, 392, 392, 393, 392, 393, 392,
                                393, 392, 393, 392, 393])
    
    weight_lon_indices = array([294, 294, 294, 294, 294, 293, 294, 294, 294,
                                294, 294, 294, 294, 294])


    weight_rivid = array([17880258, 17880268, 17880282, 17880284, 17880298,
                          17880830, 17880830, 17880830, 17880832, 17880832,
                          17880834, 17880834, 17880836, 17880836])

    weight_area = array([0., 1790099.9953, 2027699.982, 1539000.024,
                         602999.9944, 6862561.1979, 966715.0097,
                         17414823.8584, 912730.4112,  7220569.5496,
                         357239.3258, 2158260.6815, 563973.8007,
                         3354626.2261])

    weight_id = array([1788025899999999, 1788026803930294, 1788028203920294,
                       1788028403920294, 1788029803920294, 1788083003930293,
                       1788083003920294, 1788083003930294, 1788083203920294,
                       1788083203930294, 1788083403920294, 1788083403930294,
                       1788083603920294, 1788083603930294])

    assert array_equal(
        inflow_accumulator.weight_lat_indices, weight_lat_indices)
    
    assert array_equal(
        inflow_accumulator.weight_lon_indices, weight_lon_indices)
    
    assert array_equal(
        inflow_accumulator.weight_rivid, weight_rivid)
    
    assert array_equal(
        inflow_accumulator.weight_area, weight_area)
    
    assert array_equal(
        inflow_accumulator.weight_id, weight_id)

def test_find_rivid_weight_indices():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.weight_rivid = array(
        [17880258, 17880268, 17880282, 17880284, 17880298, 17880830,
         17880830, 17880830, 17880832, 17880832, 17880834, 17880834,
         17880836, 17880836])

    inflow_accumulator.rivid = array(
        [17880258, 17880268, 17880282, 17880284, 17880298, 17880830,
         17880832, 17880834, 17880836])

    inflow_accumulator.find_rivid_weight_indices()

    expected = [array([0]), array([1]), array([2]), array([3]), array([4]),
                array([5, 6, 7]), array([8, 9]), array([10, 11]),
                array([12, 13])]
    
    assert (array_equal(a, b) for a, b in zip(
        inflow_accumulator.rivid_weight_indices, expected))

def test_find_lat_lon_weight_indices():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.weight_lat_indices = array(
        [393, 393, 392, 392, 392, 393, 392, 393, 392, 393, 392, 393, 392,
         393])
    
    inflow_accumulator.weight_lon_indices = array(
        [294, 294, 294, 294, 294, 293, 294, 294, 294, 294, 294, 294, 294,
         294])

    inflow_accumulator.find_lat_lon_weight_indices()

    expected = array([2, 2, 0, 0, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2])

    assert array_equal(inflow_accumulator.lat_lon_weight_indices, expected)

def test_find_lat_lon_input_indices():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.weight_lat_indices = array(
        [393, 393, 392, 392, 392, 393, 392, 393, 392, 393, 392, 393, 392,
         393])
    
    inflow_accumulator.weight_lon_indices = array(
        [294, 294, 294, 294, 294, 293, 294, 294, 294, 294, 294, 294, 294,
         294])

    inflow_accumulator.find_lat_lon_input_indices()

    assert inflow_accumulator.lat_slice == slice(392, 394, None)

    assert inflow_accumulator.lon_slice == slice(293, 295, None)

def test_find_subset_indices():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)
    
    inflow_accumulator.lat_indices = array([392, 393, 393])
    inflow_accumulator.lon_indices = array([294, 293, 294])

    inflow_accumulator.min_lat_index = 392
    inflow_accumulator.min_lon_index = 293

    inflow_accumulator.n_lat_slice = 2
    inflow_accumulator.n_lon_slice = 2
    
    inflow_accumulator.find_subset_indices()

    expected = array([1, 2, 3])
    
    assert array_equal(inflow_accumulator.subset_indices, expected)

def test_write_multiprocessing_job_list():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.output_indices = array([(0, 1), (1, 2)])

    inflow_accumulator.input_file_array = array([
        'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0000.020.nc4', 'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0300.020.nc4'])

    inflow_accumulator.write_multiprocessing_job_list()

    expected = [
        {'input_filename': 'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0000.020.nc4', 'output_indices': (0, 1)},
        {'input_filename': 'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0300.020.nc4', 'output_indices': (1, 2)}]
    
    assert (a == b for a, b in zip(inflow_accumulator.job_list, expected))

def test_read_write_inflow():
    args, kwargs = generate_default_inflow_accumulator_arguments()

    # Change output filename to be test-specific.
    output_filename = os.path.join(
        OUTPUT_DIR, 'test_read_write_inflow_gldas2.nc')
    args[0] = output_filename
    
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.rivid = array([17880258, 17880268, 17880282])
    inflow_accumulator.lat_slice = slice(392, 394, None)
    inflow_accumulator.lon_slice = slice(293, 295, None)
    inflow_accumulator.n_lat_slice = 2
    inflow_accumulator.n_lon_slice = 2
    inflow_accumulator.subset_indices = array([1, 2, 3])
    inflow_accumulator.lat_lon_weight_indices =  array(
        [2, 2, 0, 0, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2])
    inflow_accumulator.weight_area = array(
        [0., 1790099.9953, 2027699.982, 1539000.024, 602999.9944,
         6862561.1979, 966715.0097, 17414823.8584, 912730.4112, 7220569.5496,
         357239.3258, 2158260.6815, 563973.8007, 3354626.2261])
    inflow_accumulator.rivid_weight_indices = [
        array([0]), array([1]), array([2])]
    inflow_accumulator.time = [1.2937536e+09]
    inflow_accumulator.n_time_step = 1

    # This following routine is tested by test_initialize_inflow_nc().
    inflow_accumulator.initialize_inflow_nc()

    args = {}
    args['input_filename'] = ('tests/data/lsm_grids/gldas2/' +
                              'GLDAS_NOAH025_3H.A20101231.0000.020.nc4')
    
    args['output_indices'] = (0, 1)
    args['mp_lock'] = multiprocessing.Manager().Lock()
    
    inflow_accumulator.read_write_inflow(args)

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data
    expected = array([[0.0, 0.0, 0.14193899929523468]])

    assert array_equal(result, expected)
    
def test_generate_inflow_file_gldas2():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_gldas2_check.nc')
    input_runoff_file_directory = os.path.join(DATA_DIR, 'lsm_grids', 'gldas2')
    steps_per_input_file = 1
    weight_table_file = os.path.join(DATA_DIR, 'weight_table',
                                     'weight_gldas2.csv')
    runoff_variable_names = ['Qs_acc', 'Qsb_acc']
    meters_per_input_runoff_unit = M3_PER_KG
    output_time_step_hours = 3
    land_surface_model_description = 'GLDAS2'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, output_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y%m%d.%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}.\d{2}'
    kwargs['input_runoff_file_ext'] = 'nc4'
    kwargs['start_datetime'] = datetime(2010, 12, 31)
    kwargs['end_datetime'] = datetime(2010, 12, 31, 3)
    kwargs['nproc'] = 1

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data

    expected = array(
        [[0.0, 0.0, 0.14193899929523468, 0.10773000121116638,
          0.04221000149846077, 0.27354687452316284, 0.06389112770557404,
          0.02500675432384014, 0.03947816789150238],
         [0.0, 0.0, 0.14193899929523468, 0.10773000121116638,
          0.04221000149846077, 0.27354687452316284, 0.06389112770557404,
          0.02500675432384014, 0.03947816789150238]])

    assert array_equal(result, expected)

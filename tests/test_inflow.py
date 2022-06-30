"""
Test module for inflow.py.
"""
import numpy as np
from numpy import array, array_equal
from numpy.testing import assert_allclose, assert_array_equal
import multiprocessing
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import pytest
import os

from inflow import inflow
from inflow.inflow import InflowAccumulator

SECONDS_PER_HOUR = 3600
M3_PER_KG = 0.001
M_PER_MM = 0.001
BENCHMARK_TEST_RELATIVE_TOLERANCE = 1e-06

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TEST_DIR, 'data')
OUTPUT_DIR = os.path.join(TEST_DIR, 'output')
MAIN_DIR = os.path.join(TEST_DIR, os.pardir, 'inflow')
RAPIDPY_BENCHMARK_DIR = os.path.join(TEST_DIR, 'rapidpy_benchmark')

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

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(
        DATA_DIR, 'lsm_grids', 'gldas2')
    steps_per_input_file = 1
    weight_table_file = os.path.join(DATA_DIR, 'weight_table',
                                     'weight_gldas2.csv')
    runoff_variable_names = ['Qs_acc', 'Qsb_acc']
    meters_per_input_runoff_unit = M3_PER_KG
    input_time_step_hours = 3
    land_surface_model_description = 'GLDAS2'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y%m%d.%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}.\d{2}'
    kwargs['input_runoff_file_ext'] = 'nc4'
    kwargs['start_datetime'] = datetime(2010, 12, 31)
    kwargs['end_datetime'] = datetime(2010, 12, 31, 3)
    kwargs['nproc'] = 1

    return (args, kwargs)

def test_initialize_inflow_accumulator():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)
    
def test_generate_input_runoff_file_array():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.generate_input_runoff_file_list()
    
    input_file_list = inflow_accumulator.input_file_list
    
    expected = array(
        [os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
         for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                   'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']])

    assert array_equal(input_file_list, expected)

def test_determine_output_indices():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    input_file_list = [os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
                       for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                                 'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']]

    inflow_accumulator.grouped_input_file_list = input_file_list

    inflow_accumulator.files_per_group = 1

    inflow_accumulator.output_steps_per_input_file = 1
    
    inflow_accumulator.determine_output_indices()

    output_indices = inflow_accumulator.output_indices

    expected = [(0, 1), (1, 2)]

    assert output_indices == expected

def test_generate_output_time_variable():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.input_file_list = [
        os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
        for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                  'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']]

    inflow_accumulator.output_steps_per_input_file = 1
    
    inflow_accumulator.generate_output_time_variable()

    time = inflow_accumulator.time

    expected = array([1293753600, 1293764400])

    assert array_equal(time, expected)

def test_initialize_inflow_nc():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.input_file_list = [
        os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
        for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                  'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']]

    time = array([1293753600, 1293764400])

    inflow_accumulator.time = time

    output_filename = os.path.join(OUTPUT_DIR, 'gldas2_m3_init.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

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

def test_read_rivid_lat_lon():
    """
    Verify that the lat/lon coordinates associated with `rivid` are consistent 
    with the coordinates provided in `rivid_lat_lon_file`.
    """
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.rivid = array(
        [17880258, 17880268, 17880282, 17880284, 17880298, 17880830,
         17880832, 17880834, 17880836])
    
    inflow_accumulator.rivid_lat_lon_file = os.path.join(
        DATA_DIR, 'rivid_lat_lon', 'rivid_lat_lon_z_saguache_co.csv')

    inflow_accumulator.read_rivid_lat_lon()

    expected_lat = np.array([38.2801335633626, 38.26393714825848,
                             38.233653357746064, 38.23166800178719,
                             38.21769269378019, 38.27495900388607,
                             38.262818826634984, 38.260799322005575,
                             38.25139225972157])

    expected_lon = np.array([-106.45474032038857, -106.44347678466951,
                             -106.46758637698422, -106.47629365649524,
                             -106.47094451961267, -106.49661969412226,
                             -106.46992625081815, -106.46167313145538,
                             -106.45604669355333])

    assert array_equal(inflow_accumulator.latitude, expected_lat)
    assert array_equal(inflow_accumulator.longitude, expected_lon)

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

    assert inflow_accumulator.lsm_lat_slice == slice(392, 394, None)

    assert inflow_accumulator.lsm_lon_slice == slice(293, 295, None)

def test_find_subset_indices():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)
    
    inflow_accumulator.lsm_lat_indices = array([392, 393, 393])
    inflow_accumulator.lsm_lon_indices = array([294, 293, 294])

    inflow_accumulator.lsm_min_lat_index = 392
    inflow_accumulator.lsm_min_lon_index = 293

    inflow_accumulator.n_lsm_lat_slice = 2
    inflow_accumulator.n_lsm_lon_slice = 2
    
    inflow_accumulator.find_subset_indices()

    expected = array([1, 2, 3])
    
    assert array_equal(inflow_accumulator.subset_indices, expected)

def test_write_multiprocessing_job_list():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.output_indices = array([(0, 1), (1, 2)])

    inflow_accumulator.grouped_input_file_list = [
        'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
        'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0300.020.nc4']

    inflow_accumulator.write_multiprocessing_job_list()

    expected = [
        {'input_file_list': 'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0000.020.nc4', 'output_indices': (0, 1)},
        {'input_file_list': 'tests/data/lsm_grids/gldas2/GLDAS_NOAH025_3H.A20101231.0300.020.nc4', 'output_indices': (1, 2)}]
    
    assert (a == b for a, b in zip(inflow_accumulator.job_list, expected))

def test_read_write_inflow():
    args, kwargs = generate_default_inflow_accumulator_arguments()

    # Change output filename to be test-specific.
    output_filename = os.path.join(
        OUTPUT_DIR, 'test_read_write_inflow_gldas2.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    args[0] = output_filename
    
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.rivid = array([17880258, 17880268, 17880282])
    inflow_accumulator.lsm_lat_slice = slice(392, 394, None)
    inflow_accumulator.lsm_lon_slice = slice(293, 295, None)
    inflow_accumulator.n_lsm_lat_slice = 2
    inflow_accumulator.n_lsm_lon_slice = 2
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
    inflow_accumulator.input_runoff_ndim = 3
    inflow_accumulator.n_time_step = 1
    inflow_accumulator.output_steps_per_file_group = 1
    inflow_accumulator.integrate_within_file_condition = False
    inflow_accumulator.runoff_rule = None

    # This following routine is tested by test_initialize_inflow_nc().
    inflow_accumulator.initialize_inflow_nc()

    args = {}
    args['input_file_list'] = ('tests/data/lsm_grids/gldas2/' +
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

    if os.path.exists(output_filename):
        os.remove(output_filename)

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
    kwargs['nproc'] = 2

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

# RAPIDpy benchmark tests

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_erai_t511_3h():
    output_filename = os.path.join(OUTPUT_DIR,
                                   'inflow_erai_t511_3h_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'erai_t511_3h')
    steps_per_input_file = 8
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_erai_t511.csv')
    
    runoff_variable_names = ['RO']
    meters_per_input_runoff_unit = 1
    input_time_step_hours = 3
    land_surface_model_description = 'ERAI_T511_3H'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y%m%d'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}'
    kwargs['nproc'] = 1

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data

    expected = array(
        [[0.0, 0.09419456869363785, 0.10669701546430588,
         0.08098176121711731, 0.031729694455862045, 0.9746001958847046,
         0.3536497950553894, 0.1323649138212204, 0.2060927450656891],
        [0.0, 0.09419456869363785, 0.10669701546430588,
         0.08098176121711731, 0.031729694455862045, 0.9746001958847046,
         0.3536497950553894, 0.1323649138212204, 0.2060927450656891],
        [0.0, 0.09419456869363785, 0.10669701546430588,
         0.08098176121711731, 0.031729694455862045, 0.9841606616973877,
         0.35565850138664246, 0.1323649138212204, 0.20609553158283234],
        [0.0, 0.09419456869363785, 0.10669701546430588,
         0.08098176121711731, 0.031729694455862045, 0.9841606616973877,
         0.35565850138664246, 0.1323649138212204, 0.20609553158283234],
        [0.0, 0.10086321085691452, 0.1142507866024971, 0.0867149829864502,
         0.03397604450583458, 1.041698694229126, 0.37828779220581055,
         0.14173588156700134, 0.22068282961845398],
        [0.0, 44.61029815673828, 50.53142547607422, 38.352745056152344,
         15.027099609375, 607.6677856445312, 198.18397521972656,
         62.68767547607422, 97.64747619628906],
        [0.0, 0.10086321085691452, 0.1142507866024971, 0.0867149829864502,
         0.03397604450583458, 0.9843358397483826, 0.36623555421829224,
         0.14173588156700134, 0.22066614031791687],
        [0.0, 0.10086321085691452, 0.1142507866024971, 0.0867149829864502,
         0.03397604450583458, 1.0225777626037598, 0.37427037954330444,
         0.14173588156700134, 0.22067727148532867],
        [0.0, 0.09367357939481735, 0.10610687732696533,
         0.08053385466337204, 0.0315541997551918, 0.9715988636016846,
         0.35219573974609375, 0.13163281977176666, 0.20495355129241943],
        [0.0, 0.09367357939481735, 0.10610687732696533,
         0.08053385466337204, 0.0315541997551918, 0.9715988636016846,
         0.35219573974609375, 0.13163281977176666, 0.20495355129241943],
        [0.0, 0.09752888977527618, 0.11047390103340149,
         0.08384837210178375, 0.03285286948084831, 1.0033692121505737,
         0.36496442556381226, 0.13705040514469147, 0.2133864015340805],
        [0.0, 0.09648691862821579, 0.1092936247587204,
         0.08295255899429321, 0.03250187635421753, 0.9782455563545227,
         0.35803890228271484, 0.1355861872434616, 0.2111024409532547],
        [0.0, 0.09419456869363785, 0.10669701546430588,
         0.08098176121711731, 0.031729694455862045, 0.9841606616973877,
         0.35565850138664246, 0.1323649138212204, 0.20609553158283234],
        [0.0, 26.08543586730957, 29.54775619506836, 22.426393508911133,
         8.786949157714844, 318.6654357910156, 108.18317413330078,
         36.656005859375, 57.08774185180664],
        [0.0, 0.0944029688835144, 0.10693307220935822,
         0.08116092532873154, 0.03179989382624626, 0.9662402868270874,
         0.3522227108478546, 0.13265776634216309, 0.2065456509590149],
        [0.0, 0.09211061894893646, 0.1043364629149437,
         0.07919012755155563, 0.03102771006524563, 0.9721553325653076,
         0.34984228014945984, 0.12943649291992188, 0.20153872668743134]])
    
    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_nldas2():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_nldas2_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'nldas2')
    steps_per_input_file = 1
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_nldas.csv')
    runoff_variable_names = ['SSRUNsfc_110_SFC_ave2h', 'BGRUNsfc_110_SFC_ave2h']
    meters_per_input_runoff_unit = M3_PER_KG
    input_time_step_hours = 1
    land_surface_model_description = 'NLDAS2'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['output_time_step_hours'] = 3
    kwargs['start_datetime'] = datetime(2003,1,21,21)
    kwargs['file_datetime_format'] = '%Y%m%d.%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}\.\d{2}'
    kwargs['nproc'] = 1

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data
    
    expected = array([[0.0, 24.703380584716797, 12.16619873046875,
                       9.233997344970703, 3.617999315261841,
                       344.34783935546875, 104.7877197265625,
                       31.66785430908203, 49.23480224609375]])

    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)
    
@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_era20cm():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_era_20cm_t159_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'era_20cm_t159')
    steps_per_input_file = 8
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_era_20cm_t159.csv')
    runoff_variable_names = ['ro']
    meters_per_input_runoff_unit = 1
    input_time_step_hours = 3
    land_surface_model_description = 'ERA_20CM_T159'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y%m%d'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}'
    kwargs['nproc'] = 1

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data

    expected = array([[0.0, 2.3576064109802246, 2.6705315113067627,
                       2.0269014835357666, 0.794166088104248,
                       33.24711227416992, 10.711759567260742,
                       3.312976360321045, 5.160894393920898],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.3576064109802246, 2.6705315113067627,
                       2.0269014835357666, 0.794166088104248,
                       33.24711227416992, 10.711759567260742,
                       3.312976360321045, 5.160894393920898],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.3576064109802246, 2.6705315113067627,
                       2.0269014835357666, 0.794166088104248,
                       33.24711227416992, 10.711759567260742,
                       3.312976360321045, 5.160894393920898],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.3576064109802246, 2.6705315113067627,
                       2.0269014835357666, 0.794166088104248,
                       33.24711227416992, 10.711759567260742,
                       3.312976360321045, 5.160894393920898],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625],
                      [0.0, 2.4482836723327637, 2.7732443809509277,
                       2.1048593521118164, 0.8247109055519104,
                       34.52584457397461, 11.123749732971191,
                       3.4403984546661377, 5.3593902587890625]])

    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_jules():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_jules_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'jules')
    steps_per_input_file = 1
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_jules.csv')
    runoff_variable_names = ['Qs_inst', 'Qsb_inst']
    meters_per_input_runoff_unit = M3_PER_KG * SECONDS_PER_HOUR
    input_time_step_hours = 1
    land_surface_model_description = 'JULES'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['output_time_step_hours'] = 3
    kwargs['file_datetime_format'] = '%Y%m%d_%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}\_\d{2}'
    kwargs['nproc'] = 1

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data

    expected = array(
        [[567.2524, 1333.3466, 819.36694, 389.24612, 734.6993,
          386.6492, 557.1014, 932.4421, 26.200369, 341.65445,
          1834.395, 62.19203, 352.77277, 384.8503, 1138.589,
          674.9053, 478.6787, 739.4272, 3043.0105, 727.45166,
          447.07193, 777.92694, 1329.6213, 1204.9293, 431.5345],
         [663.1168, 1615.9993, 837.7314, 420.313, 891.6125,
          452.95334, 631.63666, 1082.7152, 29.233103, 349.62073,
          2073.8196, 68.62213, 381.25674, 390.36002, 1213.2894,
          745.4743, 484.16626, 764.2803, 2951.5667, 733.0229,
          463.45758, 827.03186, 1285.5298, 1205.2982, 412.09442],
         [1533.7926, 4029.0576, 1177.0028, 733.2773, 2634.9016,
          1324.6854, 1488.8208, 2899.0767, 71.06555, 551.11255,
          4087.5203, 162.90779, 901.2595, 609.6083, 2503.032,
          1932.7179,  802.68304, 1594.1141, 3486.9854, 1200.677,
          1074.6743, 2171.5093, 2124.154, 2645.4622,  838.5452],
         [775.8643, 2045.799, 567.0287, 362.8517, 1340.3578,
          673.16266, 751.8168, 1470.9219, 35.898342, 268.98993,
          2045.5653, 82.12956, 453.43222, 296.56866, 1250.91,
          976.29553, 393.45337, 795.0479, 1825.1614, 642.7834,
          538.6402, 1095.3221, 1085.5365, 1320.2661, 416.30664],
         [1584.5144, 3627.6807, 2722.3706, 1255.0293, 1451.8827,
          722.3964, 1361.8049, 1987.8574, 58.183926, 974.3252,
          5858.7866, 137.63277, 727.896, 1042.6696, 2722.3499,
          1299.1763, 1162.8489, 1504.7202, 8102.988, 1586.684,
          795.42346, 1287.2291, 2468.9915, 1981.7826, 635.3534],
         [1474.6241, 3369.2576, 2460.1555, 1133.7378, 1519.7162,
          784.6322, 1341.7104, 2048.854, 59.649895, 926.74884,
          5270.99, 141.85724, 777.2639, 1014.79474, 2741.926,
          1422.8892, 1182.3446, 1634.5261, 7802.7427, 1634.1256,
          917.6845, 1526.302, 2782.424, 2386.8276, 809.6976],
         [693.25757, 1747.2635, 763.84064, 403.498, 1013.0527,
          505.63132, 655.0133, 1174.2416, 30.44782, 324.02194,
          2060.419, 70.86837, 391.2334, 359.74286, 1200.5563,
          787.9429, 446.87408, 748.77844, 2522.6782, 652.81006,
          466.09406, 866.4598, 1157.9978, 1186.3085, 394.35907],
         [500.34415, 1304.2186, 455.29852, 263.50278, 767.73425,
          374.6933, 459.49377, 853.7997, 21.155867, 193.03069,
          1423.3131, 48.525066, 261.92407, 209.43002, 777.4773,
          546.2827, 256.07864, 463.7026, 1379.1094, 384.2922,
          295.60648, 580.8533, 625.6821, 712.07074, 218.7111]])

    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_era_t511_24h():
    output_filename = os.path.join(OUTPUT_DIR,
                                   'inflow_erai_t511_24h_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'erai_t511_24h')
    steps_per_input_file = 1
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_erai_t511.csv')
    runoff_variable_names = ['RO']
    meters_per_input_runoff_unit = 1
    input_time_step_hours = 24
    land_surface_model_description = 'ERAI_T511_24H'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y%m%d'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}'
    kwargs['nproc'] = 1

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data

    expected = array(
        [[0.0, 1.8838914632797241, 2.1339402198791504, 1.6196352243423462,
          0.6345939040184021, 20.830469131469727, 7.354215145111084,
          2.6472983360290527, 4.122244358062744],
         [0.0, 1.8838914632797241, 2.1339402198791504, 1.6196352243423462,
          0.6345939040184021, 20.60797119140625, 7.307466983795166,
          2.6472983360290527, 4.1221795082092285]])

    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_wrf():
    output_filename = os.path.join(OUTPUT_DIR,
                                   'inflow_wrf_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'wrf')
    steps_per_input_file = 1
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_wrf.csv')
    runoff_variable_names = ['SFROFF', 'UDROFF']
    meters_per_input_runoff_unit = M_PER_MM
    input_time_step_hours = 1
    land_surface_model_description = 'WRF'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y%m%d%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{10}'
    kwargs['nproc'] = 1

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)

    result = data_out['m3_riv'][:].data

    expected = array(
        [[10.884502, 63.19402, 5.5821285, 173.81693, 24.541079, 60.41146,
          92.68378, 41.732574],
         [10.907328, 63.32505, 5.5938344, 174.17603, 24.592543, 60.538143,
          92.878136, 41.820087],
         [10.929441, 63.452057, 5.6051755, 174.52443, 24.642403, 60.66088,
          93.06644, 41.904877],
         [10.950879, 63.575165, 5.6161695, 174.86218, 24.690737, 60.779865,
          93.248985, 41.98707],
         [10.971579, 63.694008, 5.6267853, 175.18839, 24.737408, 60.89475,
          93.42524, 42.066433],
         [10.991621, 63.80905, 5.637064, 175.50418, 24.782597, 61.00599,
          93.59591, 42.143276],
         [11.01099, 63.92017, 5.6469975, 175.80913, 24.826267, 61.11349,
          93.76083, 42.21754],
         [11.029728, 64.027695, 5.656607, 176.1045, 24.868515, 61.217495,
          93.920395, 42.289387],
         [11.047858, 64.131775, 5.6659055, 176.39066, 24.909393, 61.318123,
          94.07478, 42.3589],
         [11.065461, 64.23282, 5.6749334, 176.66847, 24.949083, 61.41582,
          94.22467, 42.42639],
         [11.082502, 64.330605, 5.683673, 176.93742, 24.987505, 61.510403,
          94.36978, 42.49173],
         [11.098947, 64.42493, 5.6921062, 177.19684, 25.024582, 61.601673,
          94.50981, 42.55478],
         [11.11491, 64.515274, 5.700293, 177.44142, 25.060574, 61.690273,
          94.64574, 42.615982],
         [11.129852, 64.59666, 5.7079563, 177.65137, 25.094265, 61.773205,
          94.77297, 42.673275],
         [11.14304, 64.66441, 5.71472, 177.8127, 25.123999, 61.8464,
          94.88527, 42.72384],
         [11.153705, 64.71513, 5.720189, 177.91933, 25.148045, 61.905594,
          94.97609, 42.76473],
         [11.161491, 64.74722, 5.7241826, 177.96844, 25.165602, 61.94881,
          95.04239, 42.794582],
         [11.166056, 64.759415, 5.7265234, 177.95877, 25.175892, 61.974144,
          95.08125, 42.812084],
         [11.167226, 64.75194, 5.7271233, 177.89496, 25.17853, 61.980637,
          95.09122, 42.816566],
         [11.165115, 64.727196, 5.726041, 177.7891, 25.17377, 61.968925,
          95.07325, 42.80848],
         [11.160505, 64.68999, 5.7236767, 177.65628, 25.163378, 61.943336,
          95.03399, 42.790802],
         [11.153763, 64.64164, 5.7202187, 177.4963, 25.148174, 61.905914,
          94.97658, 42.76495],
         [11.145116, 64.58409, 5.715784, 177.31705, 25.12868, 61.85792,
          94.90295, 42.731796]])

    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_cmip5():
    output_filename = os.path.join(OUTPUT_DIR,
                                   'inflow_cmip5_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'cmip5')
    steps_per_input_file = 3
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_cmip5.csv')
    runoff_variable_names = ['total runoff']
    meters_per_input_runoff_unit = M_PER_MM
    input_time_step_hours = 24
    land_surface_model_description = 'cmip5_ccsm4_rcp60_r1i1p1'
    
    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['file_datetime_format'] = '%Y'
    kwargs['file_timestamp_re_pattern'] = r'\d{4}'
    kwargs['nproc'] = 1
    
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)
    
    result = data_out['m3_riv'][:].data

    expected = array(
        [[2.3669102e+02, 2.0495879e+02, 1.2458789e+03, 7.8238092e+02,
          4.6817994e+00, 6.2111877e+02, 1.0404017e+03],
         [8.0262001e+01, 6.9501595e+01, 4.2247797e+02, 2.6530563e+02,
          1.5875998e+00, 2.1062158e+02, 3.5280057e+02],
         [9.0090008e+00, 7.8011994e+00, 4.7420998e+01, 2.9779203e+01,
          1.7819998e-01, 2.3641199e+01, 3.9600067e+01]])
    
    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_era5():
    output_filename = os.path.join(OUTPUT_DIR,
                                   'inflow_era5_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'era5')
    steps_per_input_file = 24
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_era5.csv')
    runoff_variable_names = ['ro']
    meters_per_input_runoff_unit = 1
    input_time_step_hours = 1
    land_surface_model_description = 'ERA5'
    
    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['output_time_step_hours'] = 3
    kwargs['file_datetime_format'] = '%Y%m%d'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}'
    kwargs['nproc'] = 1
    
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)
    
    result = data_out['m3_riv'][:].data

    expected = array(
        [[34.5418, 30.594246, 21.74515, 37.7375, 128.0618, 52.37494],
         [34.5418, 30.594246, 21.74515, 37.7375, 128.0618, 52.37494],
         [36.635246, 31.98321, 21.74515, 37.7375, 129.51561, 52.37494],
         [37.681965, 32.677692, 21.74515, 37.7375, 130.24251, 52.37494],
         [36.635246, 31.98321, 21.74515, 37.7375, 129.51561, 52.37494],
         [36.635246, 31.98321, 21.74515, 37.7375, 129.51561, 52.37494],
         [37.681965, 32.677692, 21.74515, 37.7375, 130.24251, 52.37494],
         [37.681965, 32.677692, 21.74515, 37.7375, 130.24251, 52.37494]])

    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_lis():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_lis_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'lis')
    steps_per_input_file = 1
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_lis.csv')
    runoff_variable_names = ['Qs_inst', 'Qsb_inst']
    meters_per_input_runoff_unit = M3_PER_KG * SECONDS_PER_HOUR
    input_time_step_hours = 1
    land_surface_model_description = 'LIS'
    
    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['output_time_step_hours'] = 3
    kwargs['file_datetime_format'] = '%Y%m%d%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{10}'
    kwargs['nproc'] = 1
    
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)
    
    result = data_out['m3_riv'][:].data
    
    expected = array(
        [[ 945.5628  , 3211.4817  , 1383.4126  ,  926.36346 , 1070.4043  ,
          946.64044 ,  726.6858  , 2381.6956  ,   66.47308 ,  758.75574 ,
         4328.0205  ,   88.032425,  518.34717 , 1082.6536  , 2256.727   ,
         1267.9154  , 1388.7334  , 2017.4576  , 4231.574   , 1370.4656  ,
          718.87524 , 1636.4852  , 2537.6174  , 1905.2468  ,  336.53613 ],
        [ 953.27344 , 3232.131   , 1384.4031  ,  926.8229  , 1087.0782  ,
          952.69086 ,  733.68384 , 2393.081   ,   66.648575,  758.1122  ,
         4337.3203  ,   88.74647 ,  521.6534  , 1080.7017  , 2262.2405  ,
         1275.09    , 1386.3109  , 2017.0533  , 4226.0117  , 1368.3821  ,
          721.3354  , 1641.7554  , 2537.4758  , 1909.9053  ,  338.57443 ],
        [1059.913   , 3526.0505  , 1520.8453  , 1004.8099  , 1197.7274  ,
         1015.3904  ,  817.6489  , 2559.1545  ,   70.993866,  812.3189  ,
         4729.0596  ,   97.39914 ,  567.62854 , 1144.0055  , 2431.362   ,
         1368.7968  , 1458.7915  , 2119.1653  , 4523.7397  , 1435.2616  ,
          768.01587 , 1731.3744  , 2674.5535  , 2014.343   ,  363.32184 ],
        [ 999.41833 , 3330.2432  , 1476.912   ,  967.9539  , 1117.4521  ,
          966.3641  ,  772.43787 , 2438.4941  ,   68.1366  ,  789.1424  ,
         4523.101   ,   92.77318 ,  543.0794  , 1114.7448  , 2345.8345  ,
         1309.4086  , 1423.6398  , 2060.2297  , 4456.02    , 1404.7291  ,
          743.5193  , 1672.547   , 2590.6777  , 1965.6437  ,  356.88217 ],
        [ 934.25275 , 3160.1382  , 1366.5446  ,  911.3642  , 1058.0148  ,
          931.3004  ,  719.79004 , 2342.0432  ,   65.363785,  746.07117 ,
         4258.1167  ,   87.19576 ,  512.2903  , 1063.2008  , 2224.8184  ,
         1250.1874  , 1363.9293  , 1982.685   , 4172.448   , 1346.885   ,
          709.2305  , 1611.7954  , 2487.1594  , 1881.3212  ,  334.3533  ],
        [ 931.14294 , 3147.1458  , 1362.3784  ,  907.6995  , 1054.6149  ,
          927.74963 ,  717.72363 , 2332.357   ,   65.09151 ,  742.9558  ,
         4240.749   ,   86.95881 ,  510.5798  , 1058.4174  , 2216.848   ,
         1245.5339  , 1357.817   , 1974.1113  , 4157.8306  , 1341.0609  ,
          706.6907  , 1605.5321  , 2475.0833  , 1875.2444  ,  333.74512 ],
        [ 928.1732  , 3134.6216  , 1358.3087  ,  904.118   , 1050.4464  ,
          923.9091  ,  715.77405 , 2322.921   ,   64.82568 ,  739.9064  ,
         4223.7554  ,   86.72855 ,  508.96417 , 1053.7289  , 2209.0168  ,
         1241.061   , 1351.8184  , 1965.7008  , 4143.4976  , 1335.339   ,
          704.232   , 1599.431   , 2463.3992  , 1869.2788  ,  333.1282  ],
        [ 925.5732  , 3122.8599  , 1354.3364  ,  900.62866 , 1047.53    ,
          920.36206 ,  714.183   , 2313.7456  ,   64.567   ,  736.92303 ,
         4207.5444  ,   86.534294,  507.53552 , 1049.135   , 2201.4722  ,
         1236.8646  , 1345.9342  , 1957.4546  , 4129.4536  , 1329.7195  ,
          701.91895 , 1593.5602  , 2452.0886  , 1863.4773  ,  332.57147 ]])
    
    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_lis():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_lis_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'lis')
    steps_per_input_file = 1
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_lis.csv')
    runoff_variable_names = ['Qs_inst', 'Qsb_inst']
    meters_per_input_runoff_unit = M3_PER_KG * SECONDS_PER_HOUR
    input_time_step_hours = 1
    land_surface_model_description = 'LIS'
    
    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['output_time_step_hours'] = 3
    kwargs['file_datetime_format'] = '%Y%m%d%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{10}'
    kwargs['nproc'] = 1
    
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()

    data_out = Dataset(output_filename)
    
    result = data_out['m3_riv'][:].data
    
    expected = array(
        [[ 945.5628  , 3211.4817  , 1383.4126  ,  926.36346 , 1070.4043  ,
          946.64044 ,  726.6858  , 2381.6956  ,   66.47308 ,  758.75574 ,
         4328.0205  ,   88.032425,  518.34717 , 1082.6536  , 2256.727   ,
         1267.9154  , 1388.7334  , 2017.4576  , 4231.574   , 1370.4656  ,
          718.87524 , 1636.4852  , 2537.6174  , 1905.2468  ,  336.53613 ],
        [ 953.27344 , 3232.131   , 1384.4031  ,  926.8229  , 1087.0782  ,
          952.69086 ,  733.68384 , 2393.081   ,   66.648575,  758.1122  ,
         4337.3203  ,   88.74647 ,  521.6534  , 1080.7017  , 2262.2405  ,
         1275.09    , 1386.3109  , 2017.0533  , 4226.0117  , 1368.3821  ,
          721.3354  , 1641.7554  , 2537.4758  , 1909.9053  ,  338.57443 ],
        [1059.913   , 3526.0505  , 1520.8453  , 1004.8099  , 1197.7274  ,
         1015.3904  ,  817.6489  , 2559.1545  ,   70.993866,  812.3189  ,
         4729.0596  ,   97.39914 ,  567.62854 , 1144.0055  , 2431.362   ,
         1368.7968  , 1458.7915  , 2119.1653  , 4523.7397  , 1435.2616  ,
          768.01587 , 1731.3744  , 2674.5535  , 2014.343   ,  363.32184 ],
        [ 999.41833 , 3330.2432  , 1476.912   ,  967.9539  , 1117.4521  ,
          966.3641  ,  772.43787 , 2438.4941  ,   68.1366  ,  789.1424  ,
         4523.101   ,   92.77318 ,  543.0794  , 1114.7448  , 2345.8345  ,
         1309.4086  , 1423.6398  , 2060.2297  , 4456.02    , 1404.7291  ,
          743.5193  , 1672.547   , 2590.6777  , 1965.6437  ,  356.88217 ],
        [ 934.25275 , 3160.1382  , 1366.5446  ,  911.3642  , 1058.0148  ,
          931.3004  ,  719.79004 , 2342.0432  ,   65.363785,  746.07117 ,
         4258.1167  ,   87.19576 ,  512.2903  , 1063.2008  , 2224.8184  ,
         1250.1874  , 1363.9293  , 1982.685   , 4172.448   , 1346.885   ,
          709.2305  , 1611.7954  , 2487.1594  , 1881.3212  ,  334.3533  ],
        [ 931.14294 , 3147.1458  , 1362.3784  ,  907.6995  , 1054.6149  ,
          927.74963 ,  717.72363 , 2332.357   ,   65.09151 ,  742.9558  ,
         4240.749   ,   86.95881 ,  510.5798  , 1058.4174  , 2216.848   ,
         1245.5339  , 1357.817   , 1974.1113  , 4157.8306  , 1341.0609  ,
          706.6907  , 1605.5321  , 2475.0833  , 1875.2444  ,  333.74512 ],
        [ 928.1732  , 3134.6216  , 1358.3087  ,  904.118   , 1050.4464  ,
          923.9091  ,  715.77405 , 2322.921   ,   64.82568 ,  739.9064  ,
         4223.7554  ,   86.72855 ,  508.96417 , 1053.7289  , 2209.0168  ,
         1241.061   , 1351.8184  , 1965.7008  , 4143.4976  , 1335.339   ,
          704.232   , 1599.431   , 2463.3992  , 1869.2788  ,  333.1282  ],
        [ 925.5732  , 3122.8599  , 1354.3364  ,  900.62866 , 1047.53    ,
          920.36206 ,  714.183   , 2313.7456  ,   64.567   ,  736.92303 ,
         4207.5444  ,   86.534294,  507.53552 , 1049.135   , 2201.4722  ,
         1236.8646  , 1345.9342  , 1957.4546  , 4129.4536  , 1329.7195  ,
          701.91895 , 1593.5602  , 2452.0886  , 1863.4773  ,  332.57147 ]])
    
    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_erai_t255():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_erai_t255_3h_check.nc')

    if os.path.exists(output_filename):
        os.remove(output_filename)

    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'erai_t255_3h')
    steps_per_input_file = 8
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_era_t255.csv')
    runoff_variable_names = ['ro']
    meters_per_input_runoff_unit = 1
    input_time_step_hours = 3
    land_surface_model_description = 'ERAI_T255'
    
    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, input_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['output_time_step_hours'] = 3
    kwargs['file_datetime_format'] = '%Y%m%d'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}'
    kwargs['nproc'] = 1
    kwargs['runoff_rule_name'] = 'erai_t255'

    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    output_filename = inflow_accumulator.output_filename

    inflow_accumulator.generate_inflow_file()
    
    data_out = Dataset(output_filename)
    
    result = data_out['m3_riv'][:].data
    
    expected = array(
        [[0.0, 18.37784767150879, 3.4695231914520264, 2.633326768875122,
         1.0317713022232056, 250.89498901367188, 75.69084930419922,
         22.768783569335938, 35.40485763549805],
        [0.0, 15.314873695373535, 3.4695231914520264, 2.633326768875122,
         1.0317713022232056, 209.35484313964844, 63.336002349853516,
         19.075862884521484, 29.66488265991211],
        [0.0, 15.314871788024902, 0.0, 0.0, 0.0, 207.70071411132812,
         61.774253845214844, 18.464601516723633, 28.69988250732422],
        [0.0, 18.377849578857422, 3.4695234298706055, 2.633326768875122,
         1.0317713022232056, 250.89500427246094, 75.69085693359375,
         22.76878547668457, 35.40486145019531],
        [0.0, 18.37784767150879, 3.4695231914520264, 2.633326768875122,
         1.0317713022232056, 250.89498901367188, 75.69084930419922,
         22.768783569335938, 35.40485763549805],
        [0.0, 15.314873695373535, 3.4695231914520264, 2.633326768875122,
         1.0317713022232056, 209.35484313964844, 63.336002349853516,
         19.075862884521484, 29.66488265991211],
        [0.0, 15.314871788024902, 0.0, 0.0, 0.0, 207.70071411132812,
         61.774253845214844, 18.464601516723633, 28.69988250732422],
        [0.0, 18.377849578857422, 3.4695234298706055, 2.633326768875122,
         1.0317713022232056, 250.89500427246094, 75.69085693359375,
         22.76878547668457, 35.40486145019531],
        [0.0, 16.982892990112305, 2.4046289920806885, 1.8250846862792969,
         0.7150916457176208, 231.46885681152344, 69.58480072021484,
         20.899322509765625, 32.494544982910156],
        [0.0, 14.860030174255371, 2.4046289920806885, 1.8250846862792969,
         0.7150916457176208, 202.6785430908203, 61.02199935913086,
         18.339862823486328, 28.516326904296875],
        [0.0, 16.98288917541504, 0.0, 0.0, 0.0, 230.32240295410156,
         68.50238800048828, 20.475671768188477, 31.825725555419922],
        [0.0, 16.982894897460938, 2.4046285152435303, 1.8250843286514282,
         0.7150914669036865, 231.46890258789062, 69.5848159790039,
         20.89932632446289, 32.49454879760742],
        [0.0, 16.982892990112305, 2.4046289920806885, 1.8250846862792969,
         0.7150916457176208, 231.46885681152344, 69.58480072021484,
         20.899322509765625, 32.494544982910156],
        [0.0, 14.860030174255371, 2.4046289920806885, 1.8250846862792969,
         0.7150916457176208, 202.6785430908203, 61.02199935913086,
         18.339862823486328, 28.516326904296875],
        [0.0, 16.98288917541504, 0.0, 0.0, 0.0, 230.32240295410156,
         68.50238800048828, 20.475671768188477, 31.825725555419922],
        [0.0, 14.860033988952637, 2.4046285152435303, 1.8250843286514282,
         0.7150914669036865, 202.6785888671875, 61.022010803222656,
         18.339866638183594, 28.516332626342773]])
    
    assert_allclose(result, expected, rtol=BENCHMARK_TEST_RELATIVE_TOLERANCE)

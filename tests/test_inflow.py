"""
Test module for inflow.py.
"""
import numpy as np
from numpy import array, array_equal
from numpy.testing import assert_allclose
import multiprocessing
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import pytest
import os

from inflow import inflow
from inflow.inflow import InflowAccumulator

SECONDS_PER_HOUR = 3600
M3_PER_KG = 0.001
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
    input_runoff_file_directory = os.path.join(
        DATA_DIR, 'lsm_grids', 'gldas2')
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
    
    inflow_accumulator.determine_output_indices()

    output_indices = inflow_accumulator.output_indices

    expected = [(0, 1), (1, 2)]

    assert output_indices == expected

def test_generate_output_time_variable():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.grouped_input_file_list = [
        os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
        for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                  'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']]

    inflow_accumulator.output_indices = [(0, 1), (1, 2)]

    inflow_accumulator.len_input_time_variable = 1
    
    inflow_accumulator.generate_output_time_variable()

    time = inflow_accumulator.time

    expected = array([1.2937536e+09, 1.2937644e+09])
    
    assert array_equal(time, expected)

def test_initialize_inflow_nc():
    args, kwargs = generate_default_inflow_accumulator_arguments()
    inflow_accumulator = InflowAccumulator(*args, **kwargs)

    inflow_accumulator.input_file_list = [
        os.path.join(DATA_DIR, 'lsm_grids', 'gldas2', f)
        for f in ['GLDAS_NOAH025_3H.A20101231.0000.020.nc4',
                  'GLDAS_NOAH025_3H.A20101231.0300.020.nc4']]
    
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
    inflow_accumulator.input_runoff_ndim = 3
    inflow_accumulator.n_time_step = 1

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

# RAPIDpy benchmark tests

@pytest.mark.skipif(not os.path.exists(RAPIDPY_BENCHMARK_DIR),
                    reason='Only run if RAPIDpy benchmark data is available.')
def test_generate_inflow_file_erai_t511():
    output_filename = os.path.join(OUTPUT_DIR, 'inflow_erai3_t511_check.nc')
    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'erai3_t511')
    steps_per_input_file = 8
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_era_t511.csv')
    runoff_variable_names = ['RO']
    meters_per_input_runoff_unit = 1
    output_time_step_hours = 3
    land_surface_model_description = 'ERAI3'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, output_time_step_hours,
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
    input_runoff_file_directory = os.path.join(RAPIDPY_BENCHMARK_DIR,
                                               'lsm_grids', 'nldas2')
    steps_per_input_file = 1
    weight_table_file = os.path.join(RAPIDPY_BENCHMARK_DIR, 'weight_table',
                                     'weight_nldas.csv')
    runoff_variable_names = ['SSRUNsfc_110_SFC_ave2h', 'BGRUNsfc_110_SFC_ave2h']
    meters_per_input_runoff_unit = 0.001
    output_time_step_hours = 1
    land_surface_model_description = 'NLDAS2'

    args = [output_filename, input_runoff_file_directory,
            steps_per_input_file, weight_table_file, runoff_variable_names,
            meters_per_input_runoff_unit, output_time_step_hours,
            land_surface_model_description]

    kwargs = {}
    kwargs['input_time_step_hours'] = 1
    kwargs['start_datetime'] = datetime(2003,1,21,21)
    kwargs['file_datetime_format'] = '%Y%m%d.%H'
    kwargs['file_timestamp_re_pattern'] = r'\d{8}\.\d{2}'
    kwargs['nproc'] = 1
    kwargs['convert_one_hour_to_three'] = True

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

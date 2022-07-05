"""
Test module for inflow.py.
"""
from datetime import datetime
import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import yaml

from inflow import utils

RELATIVE_TOLERANCE = 1e-06

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TEST_DIR, 'data')
OUTPUT_DIR = os.path.join(TEST_DIR, 'output')

def test_read_yaml():
    """
    Verify that read_yaml successfully reads a YAML file into a dictionary.
    """
    yml = os.path.join(DATA_DIR, 'yaml', 'inflow_gldas2.yml')

    params = utils.read_yaml(yml)

    expected = {
        'output_filename': 'm3_gldas2_3hr_201012310000_to_201012310300.nc',
        'input_runoff_file_directory': '../tests/data/lsm_grids/gldas2',
        'steps_per_input_file': 1,
        'runoff_variable_names': ['Qs_acc', 'Qsb_acc'],
        'meters_per_input_runoff_unit': 0.001}

    for k, v in expected.items():
        assert params[k] == v

def test_write_yaml():
    """
    Verify that write_yaml successfully writes a dictionary to a YAML file.
    """
    fname = os.path.join(OUTPUT_DIR, 'inflow_check.yml')

    params = {
        'output_filename': 'm3_gldas2_3hr_201012310000_to_201012310300.nc',
        'input_runoff_file_directory': '../tests/data/lsm_grids/gldas2',
        'steps_per_input_file': 1,
        'runoff_variable_names': ['Qs_acc', 'Qsb_acc'],
        'meters_per_input_runoff_unit': 0.001}

    utils.write_yaml(params, fname)

    with open(fname, 'r', encoding='UTF-8') as f:
        yml_dict = yaml.load(f, Loader=yaml.SafeLoader)

    for k, v in yml_dict.items():
        assert params[k] == v

def test_parse_time_from_nc():
    """
    Verify that parse_time_from_nc correctly extracts time variable and units
    from a netCDF file.
    """
    filename = os.path.join(DATA_DIR, 'lsm_grids', 'gldas2',
                            'GLDAS_NOAH025_3H.A20101231.0000.020.nc4')
    time, units = utils.parse_time_from_nc(filename)

    expected_time = [33134220.]
    expected_units = ' minutes since 1948-01-01 03:00:00'

    assert time == expected_time
    assert units == expected_units

def test_unique_ordered():
    """
    Verify that unique_ordered returns the unique elements of an array while
    preserving the order in which each element first appears.
    """
    values = [2, 2, 4, 4, 1, 3, 3, 3, 4, 4]

    unique = utils.unique_ordered(values)

    expected = np.array([2, 4, 1, 3])

    assert_array_equal(unique, expected)

def test_parse_timestamp_from_filename():
    """
    Verify that parse_timestamp_from_filename correctly parses a timestamp from
    a filename and converts it to a datetime.datetime object.
    """
    filename = 'GLDAS_NOAH025_3H.A20101231.0000.020.nc4'
    datetime_pattern = '%Y%m%d.%H'
    re_search_pattern = r'\d{8}.\d{2}'

    result = utils.parse_timestamp_from_filename(
        filename, re_search_pattern=re_search_pattern,
        datetime_pattern=datetime_pattern)

    expected = datetime(2010,12,31,0)

    assert result == expected

def test_convert_time():
    """
    Verify that `utils.convert_time()` correctly converts an integer
    timestamp from one set of units to another.
    """
    input_units = 'minutes since 1948-01-01 03:00:00'
    output_units = 'seconds since 1970-01-01 00:00:00'

    in_datenum_array = np.array([33134220.])
    out_datenum_array = utils.convert_time(in_datenum_array, input_units,
                                           output_units)

    expected = np.array([1293753600])

    assert out_datenum_array == expected

def test_sum_over_time_increment():
    """
    Verify that sum_over_time_increment sums `data` to reduce it to
    `n_output_steps` along the first dimesion of the array.
    """
    data = np.array(
        [[11.51393425, 10.19808169, 7.24838322, 12.57916589, 42.68726397,
          17.45831253],
         [11.51393425, 10.19808169, 7.24838322, 12.57916589, 42.68726397,
          17.45831253],
         [11.51393425, 10.19808169, 7.24838322, 12.57916589, 42.68726397,
          17.45831253],
         [11.51393425, 10.19808169, 7.24838322, 12.57916589, 42.68726397,
          17.45831253],
         [11.51393425, 10.19808169, 7.24838322, 12.57916589, 42.68726397,
          17.45831253],
         [11.51393425, 10.19808169, 7.24838322, 12.57916589, 42.68726397,
          17.45831253]])

    n_output_steps = 2

    summed = utils.sum_over_time_increment(data, n_output_steps)

    expected =  np.array(
        [[34.54180275, 30.59424507, 21.74514966, 37.73749767, 128.06179191,
          52.37493759],
         [34.54180275, 30.59424507, 21.74514966, 37.73749767, 128.06179191,
          52.37493759]])

    assert_allclose(summed, expected, rtol=RELATIVE_TOLERANCE)

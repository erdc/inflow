"""
Utility functions for the inflow repository.
"""
from datetime import datetime
import re

import numpy as np
from netCDF4 import Dataset, num2date, date2num
import yaml

def read_yaml(fname):
    """
    Read YAML file key-value pairs into a dictionary.

    Parameters
    ----------
    fname : str
        Name of YAML file.

    Returns
    -------
    param_dict : dict
        Key-value pairs parsed from `fname`.
    """
    with open(fname, 'r', encoding='UTF-8') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return param_dict

def write_yaml(param_dict, fname):
    """
    Write dictionary key-value pairs to a YAML file.

    Parameters
    ----------
    param_dict : dict
        Key-value pairs to be written to `fname`.

    Returns
    -------
    fname : str
        Name of YAML file.
    """
    with open(fname, 'w', encoding='UTF-8') as f:
        yaml.dump(param_dict, f, default_flow_style=False)

def unique_ordered(arr):
    """
    Identify unique values in an array and return them in their original order.

    Parameters
    ----------
    arr : ndarray
        1D array.

    Returns
    -------
    uvals : ndarray
        1D array containing the unique entries in `arr` in the order they
        appear in `arr`.
    """
    uvals = []
    processed = set()
    for x in arr:
        if x not in processed:
            uvals.append(x)
            processed.add(x)

    uvals = np.array(uvals)

    return uvals

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
    f = filename.split('/')[-1]
    match = re.search(re_search_pattern, f)
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

def sum_over_time_increment(data, n_output_steps):
    """
    Sum `data` over the first dimension to produce values at `n_output_steps`
    equally-spaced time intervals.

    Parameters
    ----------
    data : ndarray
        Array with first dimension corresponding to a time variable.
    n_output_steps : int
        The size of the first dimension of the output array after summing `data`
        along that axis.

    Returns
    -------
    summed_data : ndarray
        `data` summed over `new_timestep_hours`.
    """
    new_time_dim = n_output_steps

    n_input_steps = data.shape[0]

    if not n_input_steps % n_output_steps == 0:
        raise ValueError(
            'n_input_steps must be an integer_multiple ' +
            'of n_output_steps.' +
            f'n_input_steps = {n_input_steps} and ' +
            'n_output_steps = {n_output_steps}.')

    # We add a new dimension, tmp_dim, to sum over.
    tmp_dim = n_input_steps // n_output_steps

    data = data.reshape(new_time_dim, tmp_dim, -1)

    summed_data = data.sum(axis=1)

    return summed_data

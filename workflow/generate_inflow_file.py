#! /usr/bin/env python

from inflow import utils
from inflow import inflow
from datetime import datetime
import yaml
import sys

if __name__=='__main__':
    yaml_file = sys.argv[1]
    params = utils.read_yaml(yaml_file)

    start_datetime = datetime.strptime(
        params['start_timestamp'], '%Y%m%d')
    end_datetime = datetime.strptime(
        params['end_timestamp'], '%Y%m%d')

    args = [params['output_filename'],
            params['input_runoff_file_directory'],
            params['steps_per_input_file'],
            params['weight_table_file'],
            params['runoff_variable_names'],
            params['meters_per_input_runoff_unit'],
            params['output_time_step_hours'],
            params['land_surface_model_description']]

    default_kwargs = {
        'input_time_step_hours': None,
        'start_datetime': None,
        'end_datetime': None,
        'file_datetime_format': '%Y%m%d',
        'file_timestamp_re_pattern': r'\d{8}',
        'input_runoff_file_ext': 'nc',
        'nproc': None,
        'output_time_units': 'seconds since 1970-01-01 00:00:00',
        'invalid_value': -9999,
        'convert_one_hour_to_three': False}

    default_keys = default_kwargs.keys()

    kwargs = {k: v for k, v in params.items() if k in default_keys}

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    inflow_accumulator = inflow.InflowAccumulator(*args, **kwargs)
        
    inflow_accumulator.generate_inflow_file()

"""
Convenience script to generate a lateral inflow file using the
InflowAccumulator class.
"""
import sys
from datetime import datetime, timedelta

from inflow import utils
from inflow import inflow

if __name__=='__main__':
    yaml_file = sys.argv[1]
    params = utils.read_yaml(yaml_file)

    start_timestamp = params['start_timestamp']
    end_timestamp = params['end_timestamp']

    if start_timestamp is None:
        start_datetime = None
    elif len(start_timestamp) == 13:
        # Assume YYYYMMDD_hhmm format.
        start_datetime = datetime.strptime(
            params['start_timestamp'], '%Y%m%d_%H%M')
    elif len(start_timestamp) == 8:
        # Assume YYYYMMDDformat.
        start_datetime = datetime.strptime(
            params['start_timestamp'], '%Y%m%d')
    else:
        print(f'start_timestamp {start_timestamp} not recognized.')
        print('start_timestamp format should be YYYYMMDD or YYMMDD_hhmm.')
        sys.exit()

    if end_timestamp is None:
        end_datetime = None
    elif len(end_timestamp) == 13:
        # Assume YYYYMMDD_hhmm format.
        end_datetime = datetime.strptime(
            params['end_timestamp'], '%Y%m%d_%H%M')
    elif len(end_timestamp) == 8:
        # Assume YYYYMMDDformat.
        end_datetime = datetime.strptime(
            params['end_timestamp'], '%Y%m%d')
        # Increment end_datetime by 23 hours and 59 minutes so inflow will
        # be returned for the full day but not for the following day.
        end_datetime += timedelta(hours=23, minutes=59)
    else:
        print(f'end_timestamp {end_timestamp} not recognized.')
        print('end_timestamp format should be YYYYMMDD or YYMMDD_hhmm.')
        sys.exit()

    args = [params['output_filename'],
            params['input_runoff_file_directory'],
            params['steps_per_input_file'],
            params['weight_table_file'],
            params['runoff_variable_names'],
            params['meters_per_input_runoff_unit'],
            params['input_time_step_hours'],
            params['land_surface_model_description']]

    default_kwargs = {
        'output_time_step_hours': None,
        'start_datetime': None,
        'end_datetime': None,
        'file_datetime_format': '%Y%m%d',
        'file_timestamp_re_pattern': r'\d{8}',
        'input_runoff_file_ext': 'nc',
        'nproc': None,
        'output_time_units': 'seconds since 1970-01-01 00:00:00',
        'invalid_value': -9999,
        'runoff_rule_name': None,
        'rivid_lat_lon_file': None}

    default_keys = default_kwargs.keys()

    kwargs = {k: v for k, v in params.items() if k in default_keys}

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    kwargs['start_datetime'] = start_datetime

    kwargs['end_datetime'] = end_datetime

    inflow_accumulator = inflow.InflowAccumulator(*args, **kwargs)

    inflow_accumulator.generate_inflow_file()

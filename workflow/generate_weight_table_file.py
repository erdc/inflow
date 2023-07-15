"""
Convenience script for generating a weight table file using functionality in
weight.py.
"""
import sys
from inflow import utils
from inflow import weight

if __name__=='__main__':
    yaml_file = sys.argv[1]
    params = utils.read_yaml(yaml_file)

    args = [params['lsm_file'],
            params['catchment_file'],
            params['connectivity_file'],
            params['weight_table_file']]

    default_kwargs = {
        'lsm_lat_variable': 'lat',
        'lsm_lon_variable': 'lon',
        'geographic_auth_code': 4326,
        'catchment_area_field_name': None,
        'catchment_id_field_name': 'FEATUREID',
        'lsm_longitude_shift': 0,
        'lsm_land_fraction_var': None,
        'clip_to_catchment_shapefile_extent': True}

    default_keys = default_kwargs.keys()

    kwargs = {k: v for k, v in params.items() if k in default_keys}

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    weight.generate_weight_table(*args, **kwargs)

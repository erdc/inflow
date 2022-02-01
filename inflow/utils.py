#! /usr/bin/env python

import yaml

def read_yaml(fname):
    with open(fname, 'r') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return param_dict

def write_yaml(param_dict, fname):
    with open(fname, 'w') as f:
        yaml.dump(param_dict, f, default_flow_style=False)


if __name__=='__main__':
    fname = 'weight_gldas2.yml'
    params = read_yaml(fname)
    for k, v in params.items():
        print('{}: {}'.format(k, v))

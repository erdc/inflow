# inflow

Description
-----------
This repository contains tools to generate lateral inflow for use in a
streamflow routing model given land surface model runoff data and stream
network/catchment information. It is based on the RAPIDpy codebase written by
Alan Snow.

Download
--------
HTTPS:
$ git clone https://github.com/mgeheran/inflow.git

SSH:
$ git clone git@github.com:mgeheran/inflow.git

External dependencies
---------------------
- gdal
- netCDF4
- pyproj
- pytest-cov
- python
- pyyaml
- rtree
- scipy
- shapely

The file inflow_environment.yml contains instructions for installing the
necessary conda environment. If conda is installed, the following commands will
create and activate this environment (named "inflow" by default).

$ conda env create -f inflow_environment.yml
$ conda activate inflow

Installation
------------
Execute one of the following commands in the top-level directory of the
repository, i.e. where the file setup.py is located.

Install package for operational use:
$ pip install .

Install package for development:
$ pip install -e .

Testing
-------
Unit tests:
$ pytest

Unit tests with coverage report:
$ pytest --cov --cov-report=html
(creates directory with HTML files reporting test coverage)

Additionally, benchmark test data is available in the rapidpy_benchmark
repository (https://github.com/mgeheran/rapidpy_benchmark). If the top-level
directory of the rapidpy_benchmark directory is located in the tests directory
of the current repository, the above pytest commands above will run the
benchmark tests automatically. The benchmark tests should be run after any
modification to the codebase, but they are not necessary for operational use.

Examples
--------
The workflow directory contains two convenience scripts for generating a weight
table and a lateral inflow file (assuming a weight table has already been
generated). Both of these scripts may be run from the command line with the name
of an appropriately configured yaml file as an argument. The directory
tests/data/yaml contains an example file for each script. These two
(independent) examples may be run from the workflow directory with the
following commands.

$ python generate_inflow_file.py ../tests/data/yaml/inflow_gldas2.yml

$ python generate_weight_table_file.py ../tests/data/yaml/weight_lis.yml

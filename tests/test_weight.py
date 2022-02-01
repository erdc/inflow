from inflow.weight import *
import pytest
from time import time
import numpy as np
from shapely.geometry import Polygon
from numpy.testing import assert_array_equal, assert_almost_equal
import subprocess
import sys
import os

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TEST_DIR, 'data')
OUTPUT_DIR = os.path.join(TEST_DIR, 'output')
                        
def read_weight_table(fname, ncol=7):

    data = np.genfromtxt(fname, delimiter=',', skip_header=1)
    data = data[:,:ncol]
    
    return data

def compare_weight_table_files(output_file, benchmark_file, ncol=7):
    output = read_weight_table(output_file, ncol=ncol)
    benchmark = read_weight_table(benchmark_file, ncol=ncol)
    assert_almost_equal(output, benchmark)
    
def test_generate_unique_id():
    """
    Verify that generate_unique_id() returns weight table ID in the correct
    format, i.e. 
    <catchment shapefile ID><3-digit latitude><3-digit longitude>
    as an integer.
    """
    rivid = 12345
    lat_index = 6
    lon_index = 7

    uid = generate_unique_id(rivid, lat_index, lon_index)

    expected = 12345006007
    
    assert (uid == expected)

def test_extract_lat_lon_from_nc():
    """
    Verify that extract_lat_lon_from_nc() correctly extracts latitude and
    longitude variables from a netCDF file.
    """
    filename = os.path.join(DATA_DIR, 'lsm_grids',
                            'gldas2', 'GLDAS_NOAH025_3H.A20101231.0000.020.nc4')
    lat_variable = 'lat'
    lon_variable = 'lon'

    lat, lon = extract_lat_lon_from_nc(filename, lat_variable=lat_variable,
                                       lon_variable=lon_variable)

    expected_lat_size = (600,)
    expected_lon_size =  (1440,)

    lat_lon_sizes = [lat.shape, lon.shape] 
    expected = [expected_lat_size, expected_lon_size]
    
    assert (lat_lon_sizes == expected)

def test_calculate_polygon_area():
    """
    Verify that a square of 30 arc-seconds is approximately 8.5 x 10^5 m^2 
    at the equator.
    """
    deg = 0.00833333 # 30 arc-seconds in degrees.
    polygon = Polygon([[0, 0], [deg, 0], [deg, deg], [0, deg]])

    area = calculate_polygon_area(polygon)
    expected = 854795.9840483834
    
    assert (area == expected)

def test_shift_longitude():
    """
    Verify that sample coordinates from a 0 to 360 degree longitude 
    coordinate system (e.g. the convention for some ECMWF grids) are 
    correctly mapped to a -180 to 180 longitude coordinate system.
    """
    lon = np.array([180, 360])

    shifted = shift_longitude(lon)
    expected = np.array([-180, 0])
    
    assert_array_equal(shifted, expected)

def test_define_geographic_spatial_reference():
    """
    Verify that a geographic spatial reference is defined according to
    convention (i.e. EPSG:4326/WGS84).
    """
    geographic_spatial_reference = define_geographic_spatial_reference(
        auth_code=4326)

    geographic_wkt = geographic_spatial_reference.ExportToWkt()
    
    expected = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

    assert (geographic_wkt == expected)

def test_reproject():
    """
    Verify that coordinates in the EPSG:4269/NAD83 coordinate reference 
    system are correctly reprojected to EPSG:4326/WGS84.
    """
    
    x = -106.52128819799998
    y = 38.21056641800004

    original_wkt = 'GEOGCRS["NAD83",DATUM["North American Datum 1983",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4269]]'

    original_crs = CRS.from_wkt(original_wkt)
    
    geographic_crs = CRS.from_epsg(4326)
    
    xy = reproject(x, y, original_crs, geographic_crs)
    
    expected = (-106.52128597511603, 38.21056645013602)
    
    assert (xy == expected)

def test_reproject_extent():
    """
    Verify that extents (i.e. [min lon, max lon, min lat, max lat]) in the 
    EPSG:4269/NAD83 coordinate reference system are correctly reprojected 
    to EPSG:4326/WGS84.
    """
    extent = [-106.52128819799998, -106.42948920899998, 38.21056641800004,
              38.32185130200008]

    original_wkt = 'GEOGCRS["NAD83",DATUM["North American Datum 1983",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4269]]'

    original_crs = CRS.from_wkt(original_wkt)
    
    geographic_crs = CRS.from_epsg(4326)
    
    extent = reproject_extent(extent, original_crs, geographic_crs)
    
    expected = [-106.52128597511603, -106.42948675047731,
                38.21056645013602, 38.32185167233476]
    
    assert (extent == expected)

def test_generate_rtree():
    """
    Verify that an Rtree spatial index populated with a single polygon
    has bounds consistent with that polygon.
    """
    poly = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

    feature_list = [{'polygon': poly}]
        
    tree = generate_rtree(feature_list)

    bounds = tree.bounds

    expected = [0.0, 0.0, 1.0, 1.0]
    
    assert (bounds == expected)

def test_get_lat_lon_indices():
    """
    Verify that get_lat_lon_indices returns the correct indices in 
    latitude/longitude arrays given sample coordinates.
    """

    lat_array = np.array([0, 45, 90])
    lon_array = np.array([0, 90, 180])

    lat = 45
    lon = 180

    latlon = get_lat_lon_indices(lat_array, lon_array, lat, lon)
    expected = (1, 2)

    assert (latlon == expected)

def test_write_weight_table():
    """
    Verify that write_weight_table produces a weight table consistent with
    benchmark.
    """
    lsm_file = os.path.join(DATA_DIR, 'lsm_grids', 'gldas2',
                            'GLDAS_NOAH025_3H.A20101231.0000.020.nc4')
    lsm_grid_lat, lsm_grid_lon = extract_lat_lon_from_nc(lsm_file)

    catchment_shapefile = os.path.join(DATA_DIR, 'catchment',
                                       'saguache_colorado', 'catchment.shp')
    shp = ogr.Open(catchment_shapefile)
    catchment_layer = shp.GetLayer()
    
    connect_rivid_list = [17880832]
    
    catchment_rivid_list = [17880282, 17880832, 17880836, 17880268,
                            17880834, 17880284, 17880830, 17880298]

    lsm_grid_polygon = Polygon([[-106.25, 38.25], [-106.5, 38.25],
                                [-106.5, 38.5], [-106.25, 38.5],
                                [-106.25, 38.25]])
    
    lsm_grid_voronoi_feature_list = [{
        'polygon': lsm_grid_polygon, 'lon': -106.375, 'lat': 38.375}]

    lsm_grid_rtree = rtree.index.Index()
    lsm_grid_rtree.insert(0, lsm_grid_polygon.bounds)
    
    out_weight_table_file = os.path.join(OUTPUT_DIR,
                                         'weight_table_test_gldas2_1.csv')

    if os.path.exists(out_weight_table_file):
        os.remove(out_weight_table_file)
        
    write_weight_table(catchment_layer, out_weight_table_file,
                       connect_rivid_list,
                       lsm_grid_lat, lsm_grid_lon, 
                       catchment_rivid_list, lsm_grid_rtree,
                       lsm_grid_voronoi_feature_list)

    entry = np.genfromtxt(out_weight_table_file, skip_header=1,
                          delimiter=',')

    expected = np.array([17880830, 17414823.8584, 294, 393, 1, -106.3750,
                         38.3750, 17880830393294])
    
    assert_array_equal(entry, expected)

def test_generate_weight_table():
    """
    Verify that the main routine in weight.py produces a weight table 
    consistent with benchmark.
    """
    catchment_shapefile = os.path.join(DATA_DIR, 'catchment',
                                       'saguache_colorado', 'catchment.shp')
    connectivity_file = os.path.join(DATA_DIR, 'connectivity',
                                     'rapid_connect_xx.csv')
    lsm_file = os.path.join(DATA_DIR, 'lsm_grids', 'gldas2',
                            'GLDAS_NOAH025_3H.A20101231.0000.020.nc4')
    out_weight_table_file = os.path.join(OUTPUT_DIR,
                                         'weight_table_test_gldas2_2.csv')
    benchmark_file = os.path.join(DATA_DIR, 'benchmark', 'weight_table',
                                  'weight_gldas2_check.csv')
    
    generate_weight_table(lsm_file, catchment_shapefile, connectivity_file,
                          out_weight_table_file)

    compare_weight_table_files(out_weight_table_file, benchmark_file)

def test_generate_weight_table_land_sea_mask():
    """
    Verify that the main routine in weight.py produces a weight table 
    consistent with benchmark using a land-sea mask.
    """
    catchment_shapefile = os.path.join(DATA_DIR, 'catchment',
                                       'mendocino_nhdplus',
                                       'NHDCat_mendocino_watershed_hopland_sample.shp')
    connectivity_file = os.path.join(DATA_DIR, 'connectivity',
                                     'rapid_connectivity_mendocino_sample.csv')
    lsm_file = os.path.join(DATA_DIR, 'lsm_grids', 'era5_land_mask',
                            'era5_land-sea_mask_mendocino_subset.nc')
    lsm_lat_variable = 'latitude'
    lsm_lon_variable = 'longitude'
    out_weight_table_file = os.path.join(OUTPUT_DIR,
                                         'weight_table_test_era5_land_mask.csv')
    benchmark_file = os.path.join(DATA_DIR, 'benchmark', 'weight_table',
                                  'weight_era5_land_mask_check.csv')
    longitude_shift = 1
    lsm_grid_mask_var = 'lsm'
    
    generate_weight_table(lsm_file, catchment_shapefile, connectivity_file,
                          out_weight_table_file,
                          lsm_lat_variable=lsm_lat_variable,
                          lsm_lon_variable=lsm_lon_variable,
                          longitude_shift=longitude_shift,
                          lsm_grid_mask_var=lsm_grid_mask_var)

    compare_weight_table_files(out_weight_table_file, benchmark_file)

def test_main():
    args = ['python', 'weight.py', 'weight_gldas2.yml']
    subprocess.call(args)
    
# test_generate_unique_id()
# test_extract_lat_lon_from_nc()
# test_extract_2d_lat_lon_from_nc()
# test_calculate_polygon_area()
# test_shift_longitude()
# test_define_geographic_spatial_reference()
# test_reproject()
# test_reproject_extent()
# test_generate_rtree()
# test_get_lat_lon_indices()
# test_write_weight_table()
# test_generate_weight_table()
# test_generate_weight_table_land_sea_mask()

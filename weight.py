#! /usr/bin/env python

from netCDF4 import Dataset, num2date, date2num
from osgeo import gdal, ogr, osr
from pyproj import CRS, Proj, Transformer, transform
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AlbersEqualAreaConversion
from catchment_shapefile import CatchmentShapefile
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
from shapely.ops import transform as shapely_transform
import numpy as np
from voronoi import pointsToVoronoiGridShapefile
from voronoi import pointsToVoronoiGridArray
from utils import read_yaml
import rtree
import sys

def extract_lat_lon_from_nc(filename, lat_variable='lat', lon_variable='lon'):
    data = Dataset(filename)
    lat = data[lat_variable][:]
    lon = data[lon_variable][:]

    return (lat, lon)

def calculate_polygon_area(polygon, native_auth_code=4326, always_xy=True):
    """
    Calculate the area in square meters of a Shapely Polygon.

    MPG TO DO: handle Shapely MultiPolygon class if necessary.
    """
    xmin, ymin, xmax, ymax = polygon.bounds

    original_crs = CRS.from_epsg(native_auth_code)

    x = polygon.exterior.coords.xy[0]
    y = polygon.exterior.coords.xy[1]

    latitude_false_origin = (ymin + ymax) / 2.0
    longitude_false_origin = (xmin + xmax) / 2.0
    latitude_first_parallel = ymin
    latitude_second_parallel = ymax
        
    equal_area_conversion = AlbersEqualAreaConversion(
        latitude_false_origin=latitude_false_origin,
        longitude_false_origin=longitude_false_origin,
        latitude_first_parallel=latitude_first_parallel,
        latitude_second_parallel=latitude_second_parallel)

    equal_area_crs = ProjectedCRS(equal_area_conversion)

    # Units are specified as 'm' for this family of crs, so area will be in
    # square meters by default.
    transform = Transformer.from_crs(original_crs, equal_area_crs,
                                       always_xy=always_xy).transform

    polygon = shapely_transform(transform, polygon)
    
    area = polygon.area

    return area

def define_geographic_spatial_reference(auth_code=4326):

    geographic_spatial_reference = osr.SpatialReference()
    
    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    geographic_spatial_reference.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER)
    geographic_spatial_reference.ImportFromEPSG(auth_code)

    return geographic_spatial_reference

def reproject(x, y, original_crs, projection_crs, always_xy=True):
    
    transformer = Transformer.from_crs(original_crs, projection_crs,
                                       always_xy=always_xy)

    x, y = transformer.transform(x,y)

    return (x, y)

def reproject_extent(extent, original_crs, reprojection_crs):
    x = extent[:2]
    y = extent[2:]

    x, y = reproject(x, y, original_crs, reprojection_crs)

    extent = [min(x), max(x), min(y), max(y)]
    
    return extent

def generate_rtree(feature_list):
    """
    Populate R-tree index with bounds of ECMWF grid cells.
    """
    index = rtree.index.Index()
    
    for idx, feature in enumerate(feature_list):
        index.insert(idx, feature['polygon'].bounds)

    return(index)

def get_lat_lon_indices(lsm_lat_array, lsm_lon_array, lat, lon):
    # MPG TO DO: Determine if we can constrain input lat/lon to be
    # 1D only to simplify.
    """
    Determine array indices for lat/lon coordinates.
    """
    if lsm_lat_array.ndim == 2 and lsm_lon_array.ndim == 2:
        lsm_lat_indices_from_lat, lsm_lon_indices_from_lat = \
            np.where((lsm_lat_array == lat))
        lsm_lat_indices_from_lon, lsm_lon_indices_from_lon = \
            np.where((lsm_lon_array == lon))

        index_lsm_grid_lat = np.intersect1d(lsm_lat_indices_from_lat,
                                            lsm_lat_indices_from_lon)[0]
        index_lsm_grid_lon = np.intersect1d(lsm_lon_indices_from_lat,
                                            lsm_lon_indices_from_lon)[0]

    elif lsm_lat_array.ndim == 1 and lsm_lon_array.ndim == 1:
        index_lsm_grid_lon = np.where(lsm_lon_array == lon)[0][0]
        index_lsm_grid_lat = np.where(lsm_lat_array == lat)[0][0]
    else:
        raise IndexError("Lat/Lon lists have invalid dimensions. "
                         "Only 1D or 2D arrays allowed ...")

    return index_lsm_grid_lat, index_lsm_grid_lon
                    
def write_weight_table(catchment_geospatial_layer, out_weight_table_file,
                       connect_rivid_list,
                       lsm_grid_lat, lsm_grid_lon, 
                       catchment_rivid_list, lsm_grid_rtree,
                       lsm_grid_voronoi_feature_list,
                       catchment_transformation=None,
                       catchment_has_area_id=False,
                       lsm_grid_mask=None):

    # MPG TO DO: determine if there is a generic object can manipulate
    # geospatial data in a way that is independent of file type. 
    
    # MPG: This line ending may need to be modified so that values are
    # recognized as valid in the inflow routine.
    dummy_row_end = '0,-9999,-9999,-9999,-9999,-9999'
    
    header = 'rivid,area_sqm,lon_index,lat_index,npoints,'
    header += 'lsm_grid_lon,lsm_grid_lat\n'
    
    with open(out_weight_table_file, 'w') as f:
        f.write(header)

        for (idx, connect_rivid) in enumerate(connect_rivid_list):
            intersection_feature_list = []
            try:
                catchment_idx = np.where(
                    catchment_rivid_list == connect_rivid)[0][0]
            except IndexError:
                # If the id from the connectivity file is not in the the
                # catchment id list, add dummy row in its place.
                f.write('{},{}\n'.format(connect_rivid, dummy_row_end))
                continue

            catchment_feature = catchment_geospatial_layer.GetFeature(
                catchment_idx)
            feature_geometry = catchment_feature.GetGeometryRef()

            if catchment_transformation is not None:
                feature_geometry.Transform(catchment_transformation)

            catchment_polygon = shapely_wkb.loads(
                bytes(feature_geometry.ExportToWkb()))

            catchment_polygon_bounds = catchment_polygon.bounds
            
            lsm_grid_intersection_index_generator = lsm_grid_rtree.intersection(
                catchment_polygon_bounds)
            
            for intersection_idx in lsm_grid_intersection_index_generator:
                lsm_grid_polygon = lsm_grid_voronoi_feature_list[
                    intersection_idx]['polygon']

                if catchment_polygon.intersects(lsm_grid_polygon):
                    try:
                        intersection_polygon = catchment_polygon.intersection(
                            lsm_grid_polygon)
                    except:
                        print(catchment_polygon)
                        # MPG: REVISIT THIS ERROR.
                        # except TopologicalError:
                        #     log('The catchment polygon with id {0} was '
                        #         'invalid. Attempting to self clean...'
                        #         .format(rapid_connect_rivid))
                        
                    if not catchment_has_area_id:
                        intersection_geometry_type = (
                            intersection_polygon.geom_type)
                        
                        if intersection_geometry_type == 'MultiPolygon':
                            intersection_area = 0
                            for p in intersection_polygon.geoms:
                                intersection_area += calculate_polygon_area(p)
                        else:
                            intersection_area = calculate_polygon_area(
                                intersection_polygon)
                    else:  
                        catchment_area = catchment_geospatial_layer.GetFeature(
                            catchment_idx).GetField(area_id)
                        intersect_fraction = (intersection_polygon.area /
                                              catchment_polygon.area)
                        intersection_area = catchment_area * intersect_fraction

                    lsm_grid_feature_lat = lsm_grid_voronoi_feature_list[
                        intersection_idx]['lat']
                    lsm_grid_feature_lon = lsm_grid_voronoi_feature_list[
                        intersection_idx]['lon']

                    lsm_grid_lat_idx = np.where(
                        lsm_grid_lat == lsm_grid_feature_lat)[0][0]
                    lsm_grid_lon_idx = np.where(
                        lsm_grid_lon == lsm_grid_feature_lon)[0][0]

                    if lsm_grid_mask is not None:
                        if lsm_grid_mask[index_lsm_grid_lat,
                                         index_lsm_grid_lon] > 0:
                            intersection_area /= lsm_grid_mask[
                                index_lsm_grid_lat, index_lsm_grid_lon]

                    intersection_dict = {
                        'rivid': connect_rivid,
                        'area': intersection_area,
                        'lsm_grid_lat': lsm_grid_feature_lat,
                        'lsm_grid_lon': lsm_grid_feature_lon,
                        'lsm_grid_lat_idx': lsm_grid_lat_idx,
                        'lsm_grid_lon_idx': lsm_grid_lon_idx}

                    intersection_feature_list.append(intersection_dict)

                    npoints = len(intersection_feature_list)

            for d in intersection_feature_list:
                if npoints < 1:
                    f.write('{},{}'.format(connect_rivid, dummy_row_end))
                else:
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        d['rivid'],
                        d['area'],
                        d['lsm_grid_lon_idx'],
                        d['lsm_grid_lat_idx'],
                        npoints,
                        d['lsm_grid_lon'],
                        d['lsm_grid_lat']))

def main(lsm_file, shapefile, connectivity_file, out_weight_table_file,
         lsm_lat_variable='lat', lsm_lon_variable='lon',
         geographic_auth_code=4326, catchment_has_area_id=False):

    catchment_data = CatchmentShapefile(shapefile)
    catchment_data.get_layer_info()
    
    catchment_spatial_reference = catchment_data.spatial_reference
    catchment_rivid_list = catchment_data.feature_id_list
    original_catchment_extent = catchment_data.extent
    original_catchment_crs = catchment_data.crs
    
    connect_rivid_list = np.genfromtxt(connectivity_file, delimiter=',',
                                       usecols=0, dtype=int)

    geographic_crs = CRS.from_epsg(geographic_auth_code)

    catchment_in_geographic_crs = (original_catchment_crs == geographic_crs)

    if not catchment_in_geographic_crs:
        geographic_spatial_reference = define_geographic_spatial_reference(
            auth_code=geographic_auth_code)
        catchment_extent = reproject_extent(original_catchment_extent,
                                            original_catchment_crs,
                                            geographic_crs)
        catchment_transformation = osr.CoordinateTransformation(
            catchment_spatial_reference, geographic_spatial_reference)
    else:
        catchment_extent = original_catchment_extent
        catchment_transformation = None

    lsm_grid_lat, lsm_grid_lon = extract_lat_lon_from_nc(
        lsm_file, lat_variable=lsm_lat_variable, lon_variable=lsm_lon_variable)
    
    lsm_grid_voronoi_feature_list = pointsToVoronoiGridArray(
        lsm_grid_lat, lsm_grid_lon, catchment_extent)

    lsm_grid_rtree = generate_rtree(lsm_grid_voronoi_feature_list)

    shp = ogr.Open(shapefile)
    catchment_geospatial_layer = shp.GetLayer()
    
    write_weight_table(catchment_geospatial_layer, out_weight_table_file,
                       connect_rivid_list,
                       lsm_grid_lat, lsm_grid_lon, 
                       catchment_rivid_list, lsm_grid_rtree,
                       lsm_grid_voronoi_feature_list,
                       catchment_transformation=catchment_transformation,
                       catchment_has_area_id=catchment_has_area_id)
    
if __name__=='__main__':
    yml = sys.argv[1]
    data = read_yaml(yml)

    catchment_has_area_id = data['catchment_has_area_id']
    catchment_shapefile = data['catchment_shapefile']
    connectivity_file = data['connectivity_file']
    lsm_file = data['lsm_file']
    lsm_lat_variable = data['lsm_lat_variable']
    lsm_lon_variable = data['lsm_lon_variable']
    out_weight_table_file = data['out_weight_table_file']
    
    main(lsm_file, catchment_shapefile, connectivity_file,
         out_weight_table_file, lsm_lat_variable=lsm_lat_variable,
         lsm_lon_variable=lsm_lon_variable,
         catchment_has_area_id=catchment_has_area_id)

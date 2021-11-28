#! /usr/bin/env python

from netCDF4 import Dataset, num2date, date2num
from osgeo import gdal, ogr, osr
from pyproj import CRS, Proj, Transformer, transform
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AlbersEqualAreaConversion
from catchment_shapefile import CatchmentShapefile
#import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
from shapely.ops import transform as shapely_transform
import numpy as np
from voronoi import pointsToVoronoiGridShapefile
from voronoi import pointsToVoronoiGridArray
import rtree
import sys

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

    # Units are specified as 'm' for this crs, so area will be in 'm^2'
    # by default.
    equal_area_crs = ProjectedCRS(equal_area_conversion)

    transform = Transformer.from_crs(original_crs, equal_area_crs,
                                       always_xy=always_xy).transform

    polygon = shapely_transform(transform, polygon)
    area = polygon.area

    return area

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
    index = rtree.index.Index()
    # Populate R-tree index with bounds of ECMWF grid cells
    for idx, feature in enumerate(feature_list):
        index.insert(idx, feature['polygon'].bounds)

    return(index)

def write_weight_table(shapefile, out_weight_table_file, connect_rivid_list,
                       lsm_grid_rtree, lsm_grid_voronoi_array,
                       catchment_transformation=None,
                       catchment_has_area_id=False):
    
    shp = ogr.Open(shapefile)
    lyr = shp.GetLayer()
    
    # MPG: This may need to be modified so that values are recognized as
    # valid in the inflow routine.
    dummy_row_end = '0,-9999,-9999,-9999,-9999,-9999'
    
    header = 'rivid,area_sqm,lon_index,lat_index,npoints,'
    header += 'lsm_grid_lon,lsm_grid_lat\n'
    
    with open(out_weight_table_file, 'w') as f:
        f.write(header)

        for (idx, connect_rivid) in enumerate(connect_rivid_list):
            intersect_grid_info_list = []
            try:
                catchment_idx = np.where(
                    catchment_rivid_list == connect_rivid)[0][0]
            except IndexError:
                # if it is not in the catchment, add dummy row in its place
                f.write('{},{}\n'.format(connect_rivid, dummy_row_end))
                continue

            catchment_feature = lyr.GetFeature(catchment_idx)
            feature_geometry = catchment_feature.GetGeometryRef()

            if catchment_transformation is not None:
                feature_geometry.Transform(catchment_transformation)

            catchment_polygon = shapely_wkb.loads(
                bytes(feature_geometry.ExportToWkb()))

            catchment_polygon_bounds = catchment_polygon.bounds

            lsm_grid_feature_generator  = lsm_grid_rtree.intersection(
                catchment_polygon_bounds)
            
            for lsm_grid_feature_idx in lsm_grid_feature_generator:
                lsm_grid_polygon = lsm_grid_voronoi_array[
                    lsm_grid_feature_idx]['polygon']

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
                        intersection_area = calculate_polygon_area(
                            intersection_polygon)
                    else:  
                        catchment_area = lyr.GetFeature(
                            catchment_idx).GetField(area_id)
                        intersect_fraction = (intersection_polygon.area /
                                              catchment_polygon.area)
                        intersection_area = catchment_area * intersect_fraction

                    print(type(catchment_polygon))
                    print(intersection_area)
                    
if __name__=='__main__':
    out_weight_table_file = 'weight_table_wip.csv'
    shapefile = 'catchment.shp'
    catchment_has_area_id = False
    lsm_file = 'GLDAS_NOAH025_3H.A20101231.0000.020.nc4'
    connectivity_file = 'rapid_connect_xx.csv'

    catchment_data = CatchmentShapefile(shapefile)
    catchment_data.get_layer_info()
    catchment_spatial_reference = catchment_data.spatial_reference
    catchment_rivid_list = catchment_data.feature_id_list
    
    connect_rivid_list = np.genfromtxt(connectivity_file, delimiter=',',
                                       usecols=0, dtype=int)

    original_catchment_extent = catchment_data.extent
    original_catchment_crs = catchment_data.crs
    
    geographic_auth_code = 4326  # EPSG:4326
    geographic_crs = CRS.from_epsg(geographic_auth_code)

    # print('orig_catchment_crs', type(original_catchment_crs))
    # print('geo_crs', type(geographic_crs))
    # sys.exit(0)
    
    geographic_spatial_reference = osr.SpatialReference()
    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    geographic_spatial_reference.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER)
    geographic_spatial_reference.ImportFromEPSG(4326)

    catchment_in_geographic_crs = (original_catchment_crs == geographic_crs)

    if not catchment_in_geographic_crs:
        catchment_extent = reproject_extent(original_catchment_extent,
                                            original_catchment_crs,
                                            geographic_crs)
        catchment_transformation = osr.CoordinateTransformation(
            catchment_spatial_reference, geographic_spatial_reference)
    else:
        catchment_extent = original_catchment_extent
        catchment_transformation = None
    
    lsm_data = Dataset(lsm_file)
    lsm_grid_lat = lsm_data['lat'][:]
    lsm_grid_lon = lsm_data['lon'][:]
    
    lsm_grid_voronoi_array = pointsToVoronoiGridArray(
        lsm_grid_lat, lsm_grid_lon, catchment_extent)

    lsm_grid_rtree = generate_rtree(lsm_grid_voronoi_array)
    
    write_weight_table(shapefile, out_weight_table_file, connect_rivid_list,
                       lsm_grid_rtree, lsm_grid_voronoi_array,
                       catchment_transformation=catchment_transformation,
                       catchment_has_area_id=catchment_has_area_id)
            

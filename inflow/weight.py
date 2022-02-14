from netCDF4 import Dataset, num2date, date2num
from osgeo import ogr
from pyproj import CRS, Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AlbersEqualAreaConversion
from shapely import wkb as shapely_wkb
from shapely.ops import transform as shapely_transform
import numpy as np
from inflow.voronoi import pointsToVoronoiGridShapefile
from inflow.voronoi import pointsToVoronoiGridArray
from inflow.utils import read_yaml
import rtree
import sys

def parse_coordinate_order(crs):
    axis_info = crs.axis_info
    
    first_direction = axis_info[0].direction

    if  first_direction.upper() == 'NORTH':
        coordinate_order = 'YX'
    elif first_direction.upper() == 'EAST':
        coordinate_order = 'XY'

    return coordinate_order

def generate_unique_id(rivid, lat_index, lon_index):

    rivid = abs(int(rivid))
    lat_index = abs(int(lat_index))
    lon_index = abs(int(lon_index))
    
    lat_idx_str = str(lat_index).zfill(4)
    lon_idx_str = str(lon_index).zfill(4)

    id_str = '{}{}{}'.format(rivid, lat_idx_str, lon_idx_str)

    uid = int(id_str)
                              
    return uid

def generate_dummy_row(rivid, invalid_value, area=0, lat=None, lon=None,
                       lat_index=None, lon_index=None):
    if lat is None:
        lat = invalid_value
    if lon is None:
        lon = invalid_value
    if lat_index is None:
        lat_index = invalid_value
    if lon_index is None:
        lon_index = invalid_value

    npoints = invalid_value
    uid = generate_unique_id(rivid, lat_index, lon_index)

    dummy_row = f'{rivid},{area},{lon_index},{lat_index},{npoints},{lon},{lat},'
    dummy_row += f'{uid}\n'

    return dummy_row

def generate_feature_id_list(geospatial_layer, id_field_name):
    feature_id_list = []
    for idx, feature in enumerate(geospatial_layer):
        feature_id = feature.GetField(id_field_name)
        feature_id_list.append(feature_id)

    return feature_id_list

def extract_lat_lon_from_nc(filename, lat_variable='lat', lon_variable='lon'):
    data = Dataset(filename)
    lat = data[lat_variable][:]
    lon = data[lon_variable][:]

    return (lat, lon)

def extract_nc_variable(filename, variable):
    data = Dataset(filename)
    variable = data[variable][:]

    return variable

def calculate_polygon_area(polygon, native_auth_code=4326):
    """
    Calculate the area in square meters of a Shapely Polygon.
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

    # Units are specified as 'm' for this family of crs, so area will be in
    # square meters by default.
    equal_area_crs = ProjectedCRS(equal_area_conversion)

    # original_crs (EPSG:4326) has orientation 'YX' (lat, lon) while
    # equal_area_crs (EPSG:9822) has orientation 'XY' (east, north).
    # Setting always_xy=True results in a transformed polygon with an area
    # that agrees with benchmark cases and manual verification.
    transformer = Transformer.from_crs(original_crs, equal_area_crs,
                                       always_xy=True)

    transform = transformer.transform

    polygon = shapely_transform(transform, polygon)
    
    area = polygon.area

    return area

def shift_longitude(longitude):

    longitude = (longitude + 180) % 360 - 180

    return longitude

def reproject(x, y, original_crs, projection_crs, always_xy=True):
    
    transformer = Transformer.from_crs(original_crs, projection_crs,
                                       always_xy=always_xy)

    x, y = transformer.transform(x,y)

    return (x, y)

def reproject_extent(extent, original_crs, reprojection_crs, always_xy=True):
    x = extent[:2]
    y = extent[2:]

    x, y = reproject(x, y, original_crs, reprojection_crs, always_xy=always_xy)

    extent = [min(x), max(x), min(y), max(y)]
    
    return extent

def generate_rtree(feature_list):
    """
    Populate R-tree index with bounds of ECMWF grid cells.
    """
    index = rtree.index.Index()

    for idx, feature in enumerate(feature_list):
        polygon_bounds =  feature['polygon'].bounds
        index.insert(idx, polygon_bounds)

    return(index)

def get_lat_lon_indices(lat_array_1d, lon_array_1d, lat, lon):
    """
    Determine array indices for lat/lon coordinates.
    """
    lat_idx = np.where(lat_array_1d == lat)[0][0]
    lon_idx = np.where(lon_array_1d == lon)[0][0]
        
    return (lat_idx, lon_idx)

def write_weight_table(catchment_geospatial_layer, out_weight_table_file,
                       connect_rivid_list,
                       lsm_grid_lat, lsm_grid_lon, 
                       catchment_id_list, lsm_grid_rtree,
                       lsm_grid_voronoi_feature_list,
                       catchment_transform=None,
                       catchment_has_area_id=False,
                       lsm_grid_mask=None,
                       invalid_value=-9999):

    # TODO: determine if there is a generic object can manipulate geospatial
    # data in a way that is independent of file type. 
        
    header = 'rivid,area_sqm,lon_index,lat_index,npoints,'
    header += 'lsm_grid_lon,lsm_grid_lat,uid\n'
    
    with open(out_weight_table_file, 'w') as f:
        f.write(header)

        for (idx, connect_rivid) in enumerate(connect_rivid_list):
            intersection_feature_list = []
            try:
                catchment_idx = catchment_id_list.index(connect_rivid)
            except ValueError:
                # If the id from the connectivity file is not in the the
                # catchment id list, add dummy row in its place.
                dummy_row = generate_dummy_row(connect_rivid, invalid_value)
                f.write(dummy_row)
                continue
            
            catchment_feature = catchment_geospatial_layer.GetFeature(
                catchment_idx)
            feature_geometry = catchment_feature.GetGeometryRef()

            catchment_polygon = shapely_wkb.loads(
                bytes(feature_geometry.ExportToWkb()))

            if catchment_transform is not None:
                catchment_polygon = shapely_transform(catchment_transform,
                                                      catchment_polygon)
            
            catchment_polygon_bounds = catchment_polygon.bounds
            
            lsm_grid_intersection_index_generator = lsm_grid_rtree.intersection(
                catchment_polygon_bounds)
            
            for intersection_idx in lsm_grid_intersection_index_generator:
                lsm_grid_polygon = lsm_grid_voronoi_feature_list[
                    intersection_idx]['polygon']

                if catchment_polygon.intersects(lsm_grid_polygon):
                    intersection_polygon = catchment_polygon.intersection(
                        lsm_grid_polygon)
                                            
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
                        # MPG: else clause not tested.
                        # Area fields are not present in any benchmark cases,
                        # but this functionality may be useful in the future.
                        catchment_area = catchment_geospatial_layer.GetFeature(
                            catchment_idx).GetField(area_id)
                        intersect_fraction = (intersection_polygon.area /
                                              catchment_polygon.area)
                        intersection_area = catchment_area * intersect_fraction
                    
                    lsm_grid_feature_lat = lsm_grid_voronoi_feature_list[
                        intersection_idx]['lat']
                    lsm_grid_feature_lon = lsm_grid_voronoi_feature_list[
                        intersection_idx]['lon']

                    lat_ndim = lsm_grid_lat.ndim

                    if lat_ndim == 1:
                        lsm_grid_lat_1d = lsm_grid_lat
                        lsm_grid_lon_1d = lsm_grid_lon
                    elif lat_ndim == 2:
                        lsm_grid_lat_1d = np.unique(lsm_grid_lat)
                        lsm_grid_lon_1d = np.unique(lsm_grid_lon)
                    
                    lat_idx_1d, lon_idx_1d = get_lat_lon_indices(
                        lsm_grid_lat_1d, lsm_grid_lon_1d,
                        lsm_grid_feature_lat, lsm_grid_feature_lon)

                    if lat_ndim == 1:
                        lsm_grid_lat_idx = lat_idx_1d
                        lsm_grid_lon_idx = lon_idx_1d
                    elif lat_ndim == 2:
                        lsm_grid_lat_idx = [lat_idx_1d, lon_idx_1d]
                        lsm_grid_lon_idx = [lat_idx_1d, lon_idx_1d]
 
                    if lsm_grid_mask is not None:
                        if lsm_grid_mask[lsm_grid_lat_idx,
                                         lsm_grid_lon_idx] > 0:
                            intersection_area /= lsm_grid_mask[
                                lsm_grid_lat_idx, lsm_grid_lon_idx]

                    try:
                        unique_id = generate_unique_id(connect_rivid,
                                                       lat_idx_1d, lon_idx_1d)
                    except:
                        # MPG: except clause not tested.
                        # This is a pathological case that should not occur.
                        unique_id = invalid_value
                        
                    intersection_dict = {
                        'rivid': connect_rivid,
                        'area': intersection_area,
                        'lsm_grid_lat': lsm_grid_feature_lat,
                        'lsm_grid_lon': lsm_grid_feature_lon,
                        'lsm_grid_lat_idx': lat_idx_1d,
                        'lsm_grid_lon_idx': lon_idx_1d,
                        'uid': unique_id}
                    
                    intersection_feature_list.append(intersection_dict)

                    npoints = len(intersection_feature_list)

            dummy_row = generate_dummy_row(connect_rivid, invalid_value,
                                           lat_index=lat_idx_1d,
                                           lon_index=lon_idx_1d,
                                           lat=lsm_grid_feature_lat,
                                           lon=lsm_grid_feature_lon)
            
            if not intersection_feature_list:
                f.write(dummy_row)
                
            for d in intersection_feature_list:
                f.write(
                    '{:d},{:0.4f},{:d},{:d},{:d},{:0.4f},{:0.4f},{:d}\n'.format(
                    d['rivid'],
                    d['area'],
                    d['lsm_grid_lon_idx'],
                    d['lsm_grid_lat_idx'],
                    npoints,
                    d['lsm_grid_lon'],
                    d['lsm_grid_lat'],
                    d['uid']))

def generate_weight_table(lsm_file, catchment_file, connectivity_file,
                          out_weight_table_file,
                          lsm_lat_variable='lat', lsm_lon_variable='lon',
                          geographic_auth_code=4326,
                          catchment_has_area_id=False,
                          catchment_id_field_name='FEATUREID',
                          longitude_shift=0,
                          lsm_grid_mask_var=None):

    catchment_file_obj = ogr.Open(catchment_file)
    catchment_geospatial_layer = catchment_file_obj.GetLayer()
    catchment_id_list = generate_feature_id_list(catchment_geospatial_layer,
                                                 catchment_id_field_name)

    # Catchment extent is a minumum bounding rectangle
    # (min(x), max(x), min(y), max(y)). This convention should not depend on the
    # order of feature coordinates (i.e. 'XY' vs. 'YX').
    original_catchment_extent = catchment_geospatial_layer.GetExtent()
    catchment_spatial_reference = catchment_geospatial_layer.GetSpatialRef()
    catchment_wkt = catchment_spatial_reference.ExportToWkt()
    original_catchment_crs = CRS.from_wkt(catchment_wkt)

    catchment_coordinate_order = parse_coordinate_order(original_catchment_crs)
    
    if catchment_coordinate_order == 'XY':
        always_xy = True
    elif catchment_coordinate_order == 'YX':
        always_xy = False
        
    connect_rivid_list = np.genfromtxt(connectivity_file, delimiter=',',
                                       usecols=0, dtype=int)

    geographic_crs = CRS.from_epsg(geographic_auth_code)

    catchment_in_geographic_crs = (original_catchment_crs == geographic_crs)

    if not catchment_in_geographic_crs:
        catchment_extent = reproject_extent(original_catchment_extent,
                                            original_catchment_crs,
                                            geographic_crs)

        transformer = Transformer.from_crs(original_catchment_crs,
                                           geographic_crs, always_xy=always_xy)

        catchment_transform = transformer.transform
        
    else:
        # MPG: else clause not tested. We perform coordinate transformations in
        # all benchmark cases.
        catchment_extent = original_catchment_extent
        catchment_transform = None

    lsm_grid_lat, lsm_grid_lon = extract_lat_lon_from_nc(
        lsm_file, lat_variable=lsm_lat_variable, lon_variable=lsm_lon_variable)

    if longitude_shift == 1:
        lsm_grid_lon = shift_longitude(lsm_grid_lon)
        
    lsm_grid_voronoi_feature_list = pointsToVoronoiGridArray(
        lsm_grid_lat, lsm_grid_lon, catchment_extent)

    lsm_grid_rtree = generate_rtree(lsm_grid_voronoi_feature_list)

    if lsm_grid_mask_var is not None:
        lsm_grid_mask = extract_nc_variable(lsm_file, lsm_grid_mask_var)
        if lsm_grid_mask.ndim == 3:
            lsm_grid_mask = lsm_grid_mask[0]
    else:
        lsm_grid_mask = None
        
    write_weight_table(catchment_geospatial_layer, out_weight_table_file,
                       connect_rivid_list,
                       lsm_grid_lat, lsm_grid_lon, 
                       catchment_id_list, lsm_grid_rtree,
                       lsm_grid_voronoi_feature_list,
                       catchment_transform=catchment_transform,
                       catchment_has_area_id=catchment_has_area_id,
                       lsm_grid_mask=lsm_grid_mask)


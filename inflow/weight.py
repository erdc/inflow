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

def generate_unique_id(rivid, lat_index, lon_index):

    rivid = int(rivid)
    lat_index = int(lat_index)
    lon_index = int(lon_index)
    
    lat_idx_str = str(lat_index).zfill(3)
    lon_idx_str = str(lon_index).zfill(3)

    id_str = '{}{}{}'.format(rivid, lat_idx_str, lon_idx_str)

    uid = int(id_str)
                              
    return uid

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

    x, y = reproject(x, y, original_crs, reprojection_crs,
                     always_xy=always_xy)

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

    # MPG TO DO: determine if there is a generic object can manipulate
    # geospatial data in a way that is independent of file type. 
    
    # MPG: This line ending may need to be modified so that values are
    # recognized as valid in the inflow routine.
    dummy_row_end = '0,' + ','.join([str(invalid_value)]*6) #-9999,-9999,-9999,-9999,-9999,-9999'
    
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
                f.write('{},{}\n'.format(connect_rivid, dummy_row_end))
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
                    try:
                        intersection_polygon = catchment_polygon.intersection(
                            lsm_grid_polygon)
                    except: # pragma: no cover
                        # MPG: except clause not tested.
                        # This is a pathological case that does not occur
                        # in any benchmark cases.

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
                    else: # pragma: no cover
                        # MPG: else clause not tested.
                        # Area fields are not present in any benchmark cases.
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
                    except: # pragma: no cover
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

            for d in intersection_feature_list:
                if npoints < 1:
                    f.write('{},{}'.format(connect_rivid, dummy_row_end))
                else:
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
                          catchment_lat_lon_order='lonlat',
                          catchment_has_area_id=False,
                          catchment_id_field_name='FEATUREID',
                          longitude_shift=0,
                          lsm_grid_mask_var=None):

    catchment_file_obj = ogr.Open(catchment_file)
    catchment_geospatial_layer = catchment_file_obj.GetLayer()
    catchment_id_list = generate_feature_id_list(catchment_geospatial_layer,
                                                 catchment_id_field_name)
    original_catchment_extent = catchment_geospatial_layer.GetExtent()
    catchment_spatial_reference = catchment_geospatial_layer.GetSpatialRef()
    catchment_wkt = catchment_spatial_reference.ExportToWkt()
    original_catchment_crs = CRS.from_wkt(catchment_wkt)
    
    connect_rivid_list = np.genfromtxt(connectivity_file, delimiter=',',
                                       usecols=0, dtype=int)

    geographic_crs = CRS.from_epsg(geographic_auth_code)

    catchment_in_geographic_crs = (original_catchment_crs == geographic_crs)

    if not catchment_in_geographic_crs:
        catchment_extent = reproject_extent(original_catchment_extent,
                                            original_catchment_crs,
                                            geographic_crs)

        catchment_transform = Transformer.from_crs(original_catchment_crs,
                                                   geographic_crs).transform
    else:
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
    
if __name__=='__main__':
    yml = sys.argv[1]
    data = read_yaml(yml)
    
    catchment_has_area_id = data['catchment_has_area_id']
    catchment_shapefile = data['catchment_shapefile']
    catchment_id_field_name = data['catchment_id_field_name']
    connectivity_file = data['connectivity_file']
    lsm_file = data['lsm_file']
    lsm_lat_variable = data['lsm_lat_variable']
    lsm_lon_variable = data['lsm_lon_variable']
    out_weight_table_file = data['out_weight_table_file']
    
    generate_weight_table(lsm_file, catchment_shapefile, connectivity_file,
                          out_weight_table_file,
                          lsm_lat_variable=lsm_lat_variable,
                          lsm_lon_variable=lsm_lon_variable,
                          catchment_id_field_name=catchment_id_field_name,
                          catchment_has_area_id=catchment_has_area_id)

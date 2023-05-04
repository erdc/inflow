"""
Routines for creating a weight table as a component of the workflow for
generating lateral inflow for a routing model.
"""
import logging
from datetime import datetime
from netCDF4 import Dataset
from osgeo import ogr
from pyproj import CRS, Transformer
from pyproj.crs import ProjectedCRS
# from pyproj.crs.coordinate_operation import AlbersEqualAreaConversion
from shapely import wkb as shapely_wkb
from shapely.ops import transform as shapely_transform
import numpy as np
import rtree

from inflow.voronoi import pointsToVoronoiGridArray

# The following import may be useful for debugging.
# from inflow.voronoi import pointsToVoronoiGridShapefile

# Configure logging.
current_timestr = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
logfile = f'weight_{current_timestr}.log'
MIN_LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s:%(name)s[%(levelname)s]: %(message)s'
logging.basicConfig(filename=logfile, level=MIN_LOGGING_LEVEL,
                    format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

def parse_coordinate_order(crs):
    """
    Return a string indicating the order of coordinate variables.

    Parameters
    ----------
    crs : pyproj.CRS
        Pyproj Coordinate Reference System manager.

    Returns
    -------
    coordinate_order : str
    String indicating the order in which latitude (Northing) and
    longitude (Easting) are reported.
    """
    axis_info = crs.axis_info

    first_direction = axis_info[0].direction

    if  first_direction.upper() == 'NORTH':
        coordinate_order = 'YX'
    elif first_direction.upper() == 'EAST':
        coordinate_order = 'XY'

    return coordinate_order

def generate_unique_id(rivid, lat_index, lon_index):
    """
    Return an identifier string.

    Parameters
    ----------
    rivid : int
        Feature identifier.
    lat_index : int
        Index of value in a latitude array.
    lon_index : int
        Index of value in a longitude array.

    Returns
    -------
    uid : str
        Unique identifier string.
    """

    rivid = abs(int(rivid))
    lat_index = abs(int(lat_index))
    lon_index = abs(int(lon_index))

    lat_idx_str = str(lat_index).zfill(4)
    lon_idx_str = str(lon_index).zfill(4)

    id_str = f'{rivid}{lat_idx_str}{lon_idx_str}'

    uid = int(id_str)

    return uid

def generate_dummy_row(rivid, invalid_value, area=0.0, lat=None, lon=None,
                       lat_index=None, lon_index=None):
    """
    Generate a string to be used as a placeholder in a weight table when no
    area intersection is detected.

    Parameters
    ----------
    rivid : int
        Feature identifier.
    invalid_value : int
        Missing data value (e.g. -9999).
    area : float, optional
        Area (should be zero).
    lat : float, optional
        Latitude coordinate.
    lon : float, optional
        Longitude coordinate.
    lat_index : int, optional
        Index of value in a latitude array.
    lon_index : int, optional
        Index of value in a longitude array.

    Returns
    -------
    dummy_row : str
        Weight-table placeholder string.
    """
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
    """
    Generate a list of feature identifiers from geospatial data.

    Parameters
    ----------
    geospatial_layer : osgeo.ogr.Layer
        Feature identifier.
    id_field_name : string
        Name of field containing the identifier (e.g. 'FEATUREID').

    Returns
    -------
    feature_id_list : list
        List of feature identifiers.
    """
    feature_id_list = []
    for feature in geospatial_layer:
        feature_id = feature.GetField(id_field_name)
        feature_id_list.append(feature_id)

    return feature_id_list

def extract_lat_lon_from_nc(filename, lat_variable='lat', lon_variable='lon'):
    """
    Generate a list of feature identifiers from geospatial data.

    Parameters
    ----------
    filename : str
        Path to file.
    lat_variable : str, optional
        Name of variable containing latitude array.
    lon_variable : str, optional
        Name of variable containing longitude array.

    Returns
    -------
    lat : ndarray
        Latitude array.
    lon : ndarray
        Longitude array.
    """
    data = Dataset(filename)
    lat = data[lat_variable][:]
    lon = data[lon_variable][:]

    return (lat, lon)

def extract_nc_variable(filename, variable):
    """
    Extract a variable from netCDF data.

    Parameters
    ----------
    filename : str
        Path to file.
    variable : str
        Name of netCDF variable.

    Returns
    -------
    variable : ndarray
        Array corresponding to the netCDF variable with name `variable`.
    """
    data = Dataset(filename)
    variable = data[variable][:]

    return variable

def calculate_polygon_area(polygon, native_auth_code=4326):
    """
    Calculate the area in square meters of a Shapely Polygon.

    Parameters
    ----------
    polygon : shapely.geometry.polygon.Polygon
        Polygon.
    native_auth_code : int
        Authority code indicating the CRS in which the polygon coordinates
        are reported.

    Returns
    -------
    area : float
        Polygon area in square meters.
    """
    xmin, ymin, xmax, ymax = polygon.bounds

    # original_crs = CRS.from_epsg(native_auth_code)
    original_crs_str = f'epsg:{native_auth_code}'

    lat_0 = (ymin + ymax) / 2.0
    lon_0 = (xmin + xmax) / 2.0
    lat_1 = ymin
    lat_2 = ymax

    # equal_area_conversion = AlbersEqualAreaConversion(
    #     latitude_false_origin=lat_0,
    #     longitude_false_origin=lon_0,
    #     latitude_first_parallel=lat_1,
    #     latitude_second_parallel=lat_2)

    # # Units are specified as 'm' for this family of crs, so area will be in
    # # square meters by default.
    # equal_area_crs = ProjectedCRS(equal_area_conversion)

    equal_area_crs_str = (f'+proj=aea +lat_0={lat_0} +lon_0={lon_0} ' +
                          f'+lat_1={lat_1} +lat_2={lat_2} +x_0=0 +y_0=0 ' +
                          '+datum=WGS84 +no_defs +type=crs +units=m')

    # original_crs (EPSG:4326) has orientation 'YX' (lat, lon) while
    # equal_area_crs (EPSG:9822) has orientation 'XY' (east, north).
    # Setting always_xy=True results in a transformed polygon with an area
    # that agrees with benchmark cases and manual verification.
    transformer = Transformer.from_crs(original_crs_str, equal_area_crs_str,
                                       always_xy=True)

    transform = transformer.transform

    polygon = shapely_transform(transform, polygon)

    area = polygon.area

    return area

def shift_longitude(longitude):
    """
    Transform longitude from a 0 to 360 degree reference system to a
    -180 to 180 degree reference system.

    Parameters
    ----------
    longitude : array_like
        Longitude in 0 to 360 degree reference system.

    Returns
    -------
    longitude : array_like
        Longitude in -180 to 180 degree reference system.
    """
    longitude = (longitude + 180) % 360 - 180

    return longitude

def reproject(x, y, original_crs, projection_crs, always_xy=True):
    """
    Transform coordinates from one CRS to another.

    Parameters
    ----------
    x : array_like
        x-coordinate (longitude or Easting)
    y : array_like
        y-coordinate (latitude or Northing)
    original_crs : pyproj.CRS
        Input coordinate reference system.
    projection_crs : pyproj.CRS
        Output coordinate reference system.
    always_xy : bool, optional
        Use order (x, y) or (longitude, latitude) or (Easing, Northing)
        for both input and output.

    Returns
    -------
    x : array_like
        Transformed x-coordinate (longitude or Easting).
    y : array_like
        Transformed y-coordinate (latitude or Northing).
    """
    transformer = Transformer.from_crs(original_crs, projection_crs,
                                       always_xy=always_xy)

    x, y = transformer.transform(x,y)

    return (x, y)

def reproject_extent(extent, original_crs, reprojection_crs, always_xy=True):
    """
    Transform the coordinates of a minimum bounding rectangle.

    Parameters
    ----------
    extent : tuple
        Minimum bounding rectangle representation
        (min(x), max(x), min(y), max(y)).
    original_crs : pyproj.CRS
        Input coordinate reference system.
    projection_crs : pyproj.CRS
        Output coordinate reference system.
    always_xy : bool, optional
        Use order (x, y) or (longitude, latitude) or (Easing, Northing)
        for both input and output.

    Returns
    -------
    extent : tuple
        Transformed minimum bounding rectangle representation
        (min(x), max(x), min(y), max(y)).
    """
    x = extent[:2]
    y = extent[2:]

    x, y = reproject(x, y, original_crs, reprojection_crs, always_xy=always_xy)

    extent = [min(x), max(x), min(y), max(y)]

    return extent

def generate_rtree(feature_list):
    """
    Populate R-tree index with polygon bounds.

    Parameters
    ----------
    feature_list : list
        List of dictionaries. Each dictionary should contain a polygon and
        lat/lon coordinates associated with the polygon centroid.

    Returns
    -------
    index : rtree.index.Index
        R-tree index structure.
    """
    index = rtree.index.Index()

    for idx, feature in enumerate(feature_list):
        polygon_bounds =  feature['polygon'].bounds
        index.insert(idx, polygon_bounds)

    return index

def get_lat_lon_indices(lat_array_1d, lon_array_1d, lat, lon):
    """
    Determine array indices for lat/lon coordinates.

    Parameters
    ----------
    lat_array_1d : ndarray
        One-dimensional latitude coordinate array.
    lon_array_1d : ndarray
        One-dimensional longitude coordinate array.
    lat : float
        Latitude coordinate.
    lon : float
        Longitude coordinate.

    Returns
    -------
    lat_idx : int
        Index of `lat` in `lat_array_1d`.
    lon_idx : int
        Index of `lon` in `lon_array_1d`.
    """
    lat_idx = np.where(lat_array_1d == lat)[0][0]
    lon_idx = np.where(lon_array_1d == lon)[0][0]

    return (lat_idx, lon_idx)

def write_weight_table(catchment_geospatial_layer, out_weight_table_file,
                       connect_rivid_array,
                       lsm_grid_lat, lsm_grid_lon,
                       catchment_id_list, lsm_grid_rtree,
                       lsm_grid_voronoi_feature_list,
                       catchment_transform=None,
                       catchment_area_field_name=None,
                       lsm_land_fraction_array=None,
                       invalid_value=-9999):

    """
    Write a weight-table CSV file. Each entry (row) in the file provides
    the area of intersection between a catchment polgon and a land surface model
    grid cell and corresponding metadata.

    Parameters
    ----------
    catchment_geospatial_layer : osgeo.ogr.Layer
        Layer from a geospatial data file (shapefile).
    out_weight_table_file : str
        Path to output file to which the weight table will be written.
    connect_rivid_array : ndarray
        Array containing stream-reach identifiers from a connectivity file.
    lsm_grid_lat : ndarray
        Land surface model latitude array (may be 1D or 2D).
    lsm_grid_lon : ndarray
        Land surface model longitude array (may be 1D or 2D).
    catchment_id_list : list
        List of catchment identifiers from a geospatial data file.
    lsm_grid_rtree : rtree.index.Index
        R-tree index structure containing land surface model grid cell data.
    lsm_grid_voronoi_feature_list : list
        List of dictionaries. Each dictionary should contain a polygon and
        lat/lon coordinates associated with the polygon centroid. The polygon
        (rectangle) corresponds to the extents of a land surface model grid
        cell.
    catchment_transform : pyproj.transformer.Transformer.transform, optional
        Method for transforming catchment polygon coordinates from native CRS
        to geographic CRS.
    catchment_area_field_name : str, optional
        Name of catchment area field in `catchment file`.
    lsm_land_fraction_array : ndarry, optional
        Array containing the fraction of land (vs. water) for each grid cell.
    invalid_value : int, optional
        Number to use in place of invalid values when writing the weight table.
    """
    # TODO: determine if there is a generic object can manipulate geospatial
    # data in a way that is independent of file type.

    header = 'rivid,area_sqm,lon_index,lat_index,npoints,'
    header += 'lsm_grid_lon,lsm_grid_lat,uid\n'

    with open(out_weight_table_file, 'w', encoding='UTF-8') as f:
        f.write(header)
        f.flush()

        for connect_rivid in connect_rivid_array:
            dummy_row = generate_dummy_row(connect_rivid, invalid_value)

            intersection_feature_list = []

            try:
                catchment_idx = catchment_id_list.index(connect_rivid)
            except ValueError:
                # If the id from the connectivity file is not in the the
                # catchment id list, add dummy row in its place.
                f.write(dummy_row)
                f.flush()
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

            # The `sorted` function accepts a generator but returns a list.
            # We sort the intersection indices to maintain a consistent order
            # for the entries (rows) in the weight-table.
            lsm_grid_intersection_index_list = sorted(
                lsm_grid_intersection_index_generator)

            for intersection_idx in lsm_grid_intersection_index_list:
                lsm_grid_polygon = lsm_grid_voronoi_feature_list[
                    intersection_idx]['polygon']

                if catchment_polygon.intersects(lsm_grid_polygon):
                    intersection_polygon = catchment_polygon.intersection(
                        lsm_grid_polygon)

                    if catchment_area_field_name is None:
                        intersection_geometry_type = (
                            intersection_polygon.geom_type)

                        if intersection_geometry_type == 'MultiPolygon':
                            intersection_area = 0
                            for geom in intersection_polygon.geoms:
                                intersection_area += calculate_polygon_area(
                                    geom)
                        else:
                            intersection_area = calculate_polygon_area(
                                intersection_polygon)
                    else:
                        # MPG: else clause not tested.
                        # Area fields are not present in any benchmark cases,
                        # but this functionality may be useful in the future.
                        catchment_area = catchment_geospatial_layer.GetFeature(
                            catchment_idx).GetField(
                                catchment_area_field_name)
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

                    if lsm_land_fraction_array is not None:
                        if lsm_land_fraction_array[lsm_grid_lat_idx,
                                         lsm_grid_lon_idx] > 0:
                            intersection_area /= lsm_land_fraction_array[
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

            if not intersection_feature_list:
                f.write(dummy_row)
                f.flush()

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
                f.flush()

def generate_weight_table(lsm_file, catchment_file, connectivity_file,
                          out_weight_table_file,
                          lsm_lat_variable='lat', lsm_lon_variable='lon',
                          geographic_auth_code=4326,
                          catchment_area_field_name=None,
                          catchment_id_field_name='FEATUREID',
                          lsm_longitude_shift=0,
                          lsm_land_fraction_var=None):

    """
    Generate a weight-table CSV file. Extract catchment, land surface model
    (LSM), and connectivity data from file. Construct the land surface model
    grid Voronoi array, feature list, and R-tree. Then call
    `write_weight_table` to write the weight-table file.

    Parameters
    ----------
    lsm_file : str
        Path to land surface model file.
    catchment_file : str
        Path to catchment file.
    connectivity_file : str
        Path to connectivity file.
    out_weight_table_file : str
        Path to output file to which the weight table will be written.
    lsm_lat_variable : str
        Name of the variable containing latitude in the LSM file.
    lsm_lon_variable : str
        Name of the variable containing longitude in the LSM file.
    connect_rivid_array : ndarray
        Array containing stream-reach identifiers from a connectivity file.
    geographic_auth_code : int, optional
        Authority code for geographic coordinate reference system.
    catchment_area_field_name : str, optional
        Name of catchment area field in `catchment file`.
    catchment_id_field_name : str, optional
        Name of catchment identifier field in `catchment file`.
    lsm_longitude_shift : int, optional
        If 1, call `shift_longitude` to transform land surface model longitude.
    lsm_land_fraction_var : str, optional
        Name of land-fraction variable in `lsm_file`.
    """
    logger.info('Reading catchment file %s.', catchment_file)
    catchment_file_obj = ogr.Open(catchment_file)
    catchment_geospatial_layer = catchment_file_obj.GetLayer()
    catchment_id_list = generate_feature_id_list(catchment_geospatial_layer,
                                                 catchment_id_field_name)

    # Catchment extent is a minimum bounding rectangle
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

    logger.info('Reading connectivity file %s.', connectivity_file)
    connect_rivid_array = np.genfromtxt(connectivity_file, delimiter=',',
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

    logger.info('Reading spatial coordinates from runoff grid file %s.',
                lsm_file)
    lsm_grid_lat, lsm_grid_lon = extract_lat_lon_from_nc(
        lsm_file, lat_variable=lsm_lat_variable, lon_variable=lsm_lon_variable)

    if lsm_longitude_shift == 1:
        lsm_grid_lon = shift_longitude(lsm_grid_lon)

    logger.info('Constructing Voronoi diagram from runoff grid coordinates.')
    lsm_grid_voronoi_feature_list = pointsToVoronoiGridArray(
        lsm_grid_lat, lsm_grid_lon, catchment_extent)

    logger.info('Constructing R-tree from Voronoi cells.')
    lsm_grid_rtree = generate_rtree(lsm_grid_voronoi_feature_list)

    if lsm_land_fraction_var is not None:
        lsm_land_fraction_array = extract_nc_variable(lsm_file,
                                                      lsm_land_fraction_var)
        if lsm_land_fraction_array.ndim == 3:
            lsm_land_fraction_array = lsm_land_fraction_array[0]
    else:
        lsm_land_fraction_array = None

    logger.info('Writing weight table to %s.', out_weight_table_file)
    write_weight_table(catchment_geospatial_layer, out_weight_table_file,
                       connect_rivid_array,
                       lsm_grid_lat, lsm_grid_lon,
                       catchment_id_list, lsm_grid_rtree,
                       lsm_grid_voronoi_feature_list,
                       catchment_transform=catchment_transform,
                       catchment_area_field_name=catchment_area_field_name,
                       lsm_land_fraction_array=lsm_land_fraction_array)

    logger.info('Done.')

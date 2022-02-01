#! /usr/bin/env python

from osgeo import gdal, ogr, osr
from pyproj import CRS, Proj, Transformer, transform

class CatchmentShapefile:

    def __init__(self, filename):
        self.filename = filename
        self.spatial_reference = None
        self.wkt = None
        self.auth_code = None
        self.proj4_string = None
        self.crs = None
        self.feature_count = None
        self.extent = None
        self.feature_id_list = []
        
    def get_layer_info(self, id_field_name='FEATUREID'):
        shp = ogr.Open(self.filename)

        # MPG: we do not assign the shapefile layer to a class attribute
        # because, when other methods attempt to access it, it results in a
        # segmentation fault.
        lyr = shp.GetLayer()
        self.spatial_reference = lyr.GetSpatialRef()
        self.wkt = self.spatial_reference.ExportToWkt()
        self.auth_code = int(self.spatial_reference.GetAuthorityCode(None))
        self.proj4_string = self.spatial_reference.ExportToProj4()
        self.crs = CRS.from_wkt(self.wkt)
        self.feature_count = lyr.GetFeatureCount()
        self.extent = lyr.GetExtent()
        
        feature_id_list = []
        for idx, feature in enumerate(lyr):
            feature_id_list.append(feature.GetField(id_field_name))
            
        self.feature_id_list = feature_id_list
        
if __name__=='__main__':
    filename = 'catchment.shp'
    shp = CatchmentShapefile(filename)
    shp.get_layer_info()

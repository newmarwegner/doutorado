import os
import shutil
from datetime import date

import ee
import geopandas as gpd
import geemap
from decouple import config
from shapely.geometry import LineString, MultiLineString


class SentinelHandler:
    def __init__(self):
        pass
    
    def get_abs_path(self, folder='agriculture'):
        """
        Absolute path of a especific folder
        :param folder: default its root of project
        :return: String of absolute path folder
        """
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', folder))
    
    def create_folder(self, path):
        """
        Method to create a folder in specific path
        :return: Folder created
        """
        if not os.path.exists(path):
            os.makedirs(path)
    
    def path_files(self, folder, endscaracters=('.tif',)):
        """
        Method to return a list of path files that endswith a specific caracters
        :param folder: Path folder
        :param endscaracters: Caracters that path needs endswith, default its .tif. Can be a tuple with mores endswith
        :return: A list of path files
        """
        paths = []
        for root, directory, files in os.walk(folder):
            for file in files:
                if file.endswith(endscaracters):
                    paths.append(os.path.join(folder, file))
        
        return paths
    
    def del_folders(self, folders=None):
        """
        Method to delete folders
        :param folders: A list of folders names to be delete, default its outputs and merged
        :return: Folders delete
        """
        if folders is None:
            folders = ['outputs', 'merged']
        for folder in folders:
            shutil.rmtree(self.get_abs_path(folder))
    
    def segment_vector(self, geodataframe):
        """
        Methon to divide vector in parts to download sentinel images
        :param geodataframe: geodataframe from geopackage in input directory
        :return: geodataframe divided
        """
        ds_in = geodataframe
        num_tiles = int(ds_in['geometry'].to_crs(epsg=5880).area[0] / 220300)
        minx, miny, maxx, maxy = [i[1][0] for i in ds_in.bounds.items()]
        dx = (maxx - minx) / num_tiles
        dy = (maxy - miny) / num_tiles
        
        lines = []
        for x in range(num_tiles + 1):
            lines.append(LineString([(minx + x * dx, miny), (minx + x * dx, maxy)]))
        for y in range(num_tiles + 1):
            lines.append(LineString([(minx, miny + y * dy), (maxx, miny + y * dy)]))
        
        grid = gpd.GeoDataFrame({'col1': ['id_pk'], 'geometry': [MultiLineString(lines)]}, crs="EPSG:4326")
        grid['geometry'] = grid['geometry'].buffer(0.0000000000001)
        gdf = gpd.overlay(ds_in, grid, how='difference').explode().reset_index()
        
        return gdf.rename(columns={"level_1": 'id_pk'}).drop(columns='level_0')
    
    def read_vectorlayer(self, vector, driver):
        """
        Method to read a vector layer
        :param vector: name of vector layer with extension inside o inputs folder
        :param drive: driver to open vector layer (gpkg,ogr,etc)
        :return:
        """
        gdf = gpd.read_file(os.path.join(self.get_abs_path('inputs'), vector), driver=driver)
        
        if gdf['geometry'].count() == 1:
            area = gdf['geometry'].to_crs(epsg=5880).area
            if area[0] < 220300:
                
                return gdf.reset_index().rename(columns={"index": 'id_pk'})
            else:
                return self.segment_vector(gdf)
        else:
            return 'Exist more than one features in vector limits, dissolve it and re-run the method'
    
    def get_class(self, geodataframe, filter_field):
        """
        Method to get each class for each row in geodataframe
        :param geodataframe: geodataframe
        :param filter_field: field to get unique classes
        :return: List with all unique classes in geodataframe field
        """
        
        return list(set([v[filter_field] for k, v in geodataframe.iterrows()]))
    
    def get_polygon_filtered(self, filter_field, feature_name, geodataframe):
        """
        Method to get polygon filtered considering loop with filter field
        :param filter_field: Field to filter geodataframe
        :param feature_name: Parameter to filter geodataframe
        :param geodataframe: Geodataframe to be filter
        """
        filter_polygon = geodataframe.loc[(geodataframe[filter_field] == feature_name)]
        xmin, ymin, xmax, ymax = filter_polygon.total_bounds
        box = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
        
        return filter_polygon, box


class SentinelDownloader:
    def __init__(self):
        self.handler = SentinelHandler()
    
    def authenticate_gee(self):
        """
        Method to authenticate gee to download images
        """
        credentials = ee.ServiceAccountCredentials(config('service_account'), config('credentials_api'))
        
        return ee.Initialize(credentials)
    
    def down_geemap(self, feature_name, box, start_date, end_date=date.today()):
        """
        Method to configure download image with geemap
        :param feature_name: Feature name obtained from filter field
        :param box: bounding box do crop download image
        :param start_date: start date to find images
        :param end_date: end date to find images
        :return:
        """
        self.handler.create_folder('output')
        out_dir = os.path.join(self.handler.get_abs_path('output'), str(feature_name))
        boundary = ee.Geometry.Polygon(box, None, False)
        collection = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(boundary) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 100) \
            .filterDate(start_date, str(end_date))
        
        geemap.ee_export_image_collection(collection, scale=10, crs='EPSG:4326', region=boundary, out_dir=out_dir)
    
    def download_sentinel(self, vector_limit, driver, filter_field, start_date):
        """
        Method to download sentinel images from gee
        :param vector_limit: string of name vector limit in input folder
        :param driver: driver to open vector (ogr,gpkg,etc)
        :param filter_field: column to be used as filter polygons
        :param start_date: start date to filter images (end date default is today)
        :return: Images downloaded in output folder
        """
        gdf = self.handler.read_vectorlayer(vector_limit, driver)
        feature_names = self.handler.get_class(gdf, filter_field)
        for feature_name in feature_names:
            filter_polygon, box = self.handler.get_polygon_filtered(filter_field, feature_name, gdf)
            self.down_geemap(feature_name, box, start_date)


if __name__ == '__main__':
    sd = SentinelDownloader()
    sd.authenticate_gee()
    sd.download_sentinel('vector_tasca_test.gpkg', 'gpkg', 'id_pk', '2021-07-01')

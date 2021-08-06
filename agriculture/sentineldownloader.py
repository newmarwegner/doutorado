import os
import shutil
import ee
import geopandas as gpd
import geemap
import rasterio
from rasterio.merge import merge
from itertools import groupby
from decouple import config
from shapely.geometry import LineString, MultiLineString
from datetime import date


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
        # divide = int(ds_in['geometry'].to_crs(epsg=5880).area[0] / 220300)
        num_tiles = int(ds_in['geometry'].to_crs(epsg=5880).area[0] / 500000)
        print(num_tiles)
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
    
    def rename_path_rasters(self):
        """
        Method to rename rasters according to date
        """
        output_paths = []
        for root, directory, files in os.walk('../output'):
            for file in files:
                if file.endswith('.tif'):
                    dir_path = root + '/'
                    os.rename(dir_path + file, dir_path + file[0:8] + '.tif')
                    full_path = f"{dir_path}{dir_path.split('/')[-2]}_{file.replace(file, file[0:8] + '.tif')}"
                    output_paths.append(full_path)
    
    def down_paths(self):
        """
        Method to create a list of download images paths to be used in create a mosaic
        :return: list of paths
        """
        self.rename_path_rasters()
        paths = []
        for root, directory, files in os.walk(self.get_abs_path('output')):
            if not root.endswith('output'):
                for file in files:
                    paths.append(os.path.join(root, file))
        
        return paths
    
    def grouppath_mosaic(self, paths_to_mosaic):
        """
            Method to grouppaths to create a input to mosaic
            """
        func = lambda x: x[-11:]
        temp = sorted(paths_to_mosaic, key=func)
        
        return [list(paths_to_mosaic) for i, paths_to_mosaic in groupby(temp, func)]


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
        out_dir = os.path.join(self.handler.get_abs_path('output'), str(feature_name))
        boundary = ee.Geometry.Polygon(box, None, False)
        collection = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(boundary) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 100) \
            .filterDate(start_date, str(end_date))
        
        geemap.ee_export_image_collection(collection, scale=10, crs='EPSG:4326', region=boundary, out_dir=out_dir)
    
    def create_mosaic(self):
        """
        Method to create mosaic with all download images
        """
        self.handler.create_folder('../merged')
        src_files_to_mosaic = []
        out_meta = []
        for i in self.handler.grouppath_mosaic(self.handler.down_paths()):
            ano = i[0].split('/')[-1][-12:-4]
            
            for file in i:
                full = ''.join([file[:file.rfind('/') + 1], ano + '.tif'])
                src = rasterio.open(full)
                src_files_to_mosaic.append(src)
                out_meta = src.meta.copy()
            
            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta.update({"driver": "GTiff",
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans,
                             "crs": "+proj=longlat +datum=WGS84 +no_defs"
                             }
                            )
            with rasterio.open(f'../merged/{ano}.tif', "w", **out_meta) as dest:
                dest.write(mosaic)
        
        self.handler.del_folders(['output', ])
    
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
        
        self.create_mosaic()


class SentinelIndexes:
    def __init__(self):
        self.handler = SentinelHandler()
        self.profile = None
    
    def get_bands(self, path):
        """
        Method to get bands from mosaics in list on order below
        [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12,AOT,WVP,SCL,TCI_R,TCI_G,TCI_B,MSK_CLDPRB,MSK_SNWPRB,QA10,QA20,QA60]
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
        :return: A list of array bands
        """
        src = rasterio.open(path, 'r')
        bands = [src.read(1, masked=True) / 10000,
                 src.read(2, masked=True) / 10000,
                 src.read(3, masked=True) / 10000,
                 src.read(4, masked=True) / 10000,
                 src.read(5, masked=True) / 10000,
                 src.read(6, masked=True) / 10000,
                 src.read(7, masked=True) / 10000,
                 src.read(8, masked=True) / 10000,
                 src.read(9, masked=True) / 10000,
                 src.read(10, masked=True) / 10000,
                 src.read(11, masked=True) / 10000,
                 src.read(12, masked=True) / 10000,
                 src.read(13, masked=True) / 10000,
                 src.read(14, masked=True) / 10000,
                 src.read(15, masked=True) / 10000,
                 src.read(16, masked=True) / 10000,
                 src.read(17, masked=True) / 10000,
                 src.read(18, masked=True) / 10000,
                 src.read(19, masked=True) / 10000,
                 src.read(20, masked=True) / 10000,
                 src.read(21, masked=True) / 10000,
                 src.read(22, masked=True) / 10000,
                 src.read(4, masked=True) / 10000]
        
        self.profile = src.profile
        src.close()
        
        return bands, self.profile


class Statistics:
    def __init__(self):
        pass


if __name__ == '__main__':
    # sd = SentinelDownloader()
    # sd.authenticate_gee()
    # sd.download_sentinel('teste.gpkg', 'gpkg', 'id_pk', '2021-07-15')
    si = SentinelIndexes()
    si.handler.create_folder(si.handler.get_abs_path('indexes'))
    for i in si.handler.path_files(si.handler.get_abs_path('merged')):
        bands, profile = si.get_bands(i)
        print(profile)
        break

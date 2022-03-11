import os
import shutil
import psycopg2
import ee
import geopandas as gpd
import geemap
import rasterio
import numpy as np
import webbrowser
from rasterio.merge import merge
from itertools import groupby
from decouple import config
from shapely.geometry import LineString, MultiLineString
from datetime import date
from jinja2 import Environment, FileSystemLoader


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
    def __init__(self, path_raster):
        self.date = path_raster.split('/')[-1][:8]
        self.handler = SentinelHandler()
        self.path_raster = path_raster
        self.profile = None
        self.b1, self.b2, self.b3, self.b4, self.b5, \
        self.b6, self.b7, self.b8, self.b8A, self.b9, \
        self.b11, self.b12, self.AOT, self.WVP, self.SCL, \
        self.TCI_R, self.TCI_G, self.TCI_B, self.MSK_CLDPRB, \
        self.MSK_SNWPRB, self.QA10, self.QA20, self.QA60 = self.get_bands()[0]
    
    def get_bands(self):
        """
        Method to get bands from mosaics in list on order below
        [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12,AOT,WVP,SCL,TCI_R,TCI_G,TCI_B,MSK_CLDPRB,MSK_SNWPRB,QA10,QA20,QA60]
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
        :return: A list of array bands
        """
        src = rasterio.open(self.path_raster, 'r')
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
                 src.read(13, masked=True) / 1000,
                 src.read(14, masked=True) / 1000,
                 src.read(15, masked=True),
                 src.read(16, masked=True),
                 src.read(17, masked=True),
                 src.read(18, masked=True),
                 src.read(19, masked=True),
                 src.read(20, masked=True),
                 src.read(21, masked=True),
                 src.read(22, masked=True),
                 src.read(23, masked=True)]
        
        self.profile = src.profile
        src.close()
        
        return bands, self.profile
    
    def get_evi(self):
        """
        Method to generate evi index. The formula used:
        2.5*(self.b8-self.b4)/((self.b8+6*self.b4-7.5*self.b2)+1)
        :return: Array with index values
        """
        numerator = np.multiply(2.5, np.subtract(self.b8, self.b4))
        denominator = np.add(np.add(self.b8, np.subtract(np.multiply(6, self.b8),
                                                         np.multiply(7.5, self.b2))), 1)
        
        return np.divide(numerator, denominator)
    
    def get_atsavi(self):
        """
        Method to generate atsavi index. The formula used:
        1.22 * (self.b8 - 1.22 * self.b4 - 0.03) / (self.b8 + self.b4 - 1.22 * 0.03 +
        0.08 * (1.0 + 1.22**2))
        :return: Array with index values
        """
        numerator = np.multiply(1.22,
                                (np.subtract(np.subtract(self.b8, np.multiply(1.22, self.b4)), 0.03)))
        
        denominator = np.add(np.subtract(np.add(self.b8, self.b4), np.multiply(1.22, 0.03)),
                             np.multiply(0.08, np.add(1.0, np.power(1.22, 2))))
        
        return np.divide(numerator, denominator)
    
    def get_ari(self):
        """
        Method to generate ari2 index. The formula used:
        (1.0 / self.b3) - (1.0 / self.b5)
       
        :return: Array with index values
        """
        
        return np.subtract(np.divide(1, self.b3), np.divide(1, self.b5))
    
    def get_avi(self):
        """
        Method to generate avi index. The formula used:
        2.0 * self.b9 - self.b4
        :return: Array with index values
        """
        
        return np.subtract(np.multiply(2, self.b9), self.b4)
    
    def get_arvi(self):
        """
        Method to generate arvi index. The formula used:
        (self.b9 - self.b4 - 0.106 * (self.b4 - self.b2)) / (self.b9 + self.b4 - 0.106 * (self.b4 - self.b2))
        :return: Array with index values
        """
        numerator = np.subtract(self.b9, self.b4, np.multiply(0.106, np.subtract(self.b4, self.b2)))
        denominator = np.add(self.b9,
                             np.subtract(self.b4, np.multiply(0.106, np.subtract(self.b4, self.b2))))
        
        return np.divide(numerator, denominator)
    
    def get_chlgreen(self):
        """
        Method to generate chlgreen index. The formula used:
        (self.b7/self.b3)**-1
        :return: Array with index values
        """
        
        return np.power(np.divide(self.b7, self.b3), -1)
    
    def get_fe3(self):
        """
        Method to generate fe3+ index. The formula used:
        self.b4/self.b3
        :return: Array with index values
        """
        
        return np.divide(self.b4, self.b3)
    
    def get_fo(self):
        """
        Method to generate fe3+ index. The formula used:
        self.b11/self.b8
        :return: Array with index values
        """
        
        return np.divide(self.b11, self.b8)
    
    def get_gvmi(self):
        """
        Method to generate gvmi index. The formula used:
        ((self.b8 + 0.1) - (self.b12 + 0.02)) / ((self.b8 + 0.1) + (self.b12 + 0.02))
        :return: Array with index values
        """
        numerator = np.subtract(np.add(self.b8, 0.1), np.add(self.b12, 0.02))
        denominator = np.add(np.add(self.b8, 0.1), np.add(self.b12, 0.02))
        
        return np.divide(numerator, denominator)
    
    def get_gndvi(self):
        """
        Method to generate gndvi index. The formula used:
        (self.b8 - self.b3) / (self.b8 + self.b3)
        :return: Array with index values
        """
        numerator = np.subtract(self.b8, self.b3)
        denominator = np.add(self.b8, self.b3)
        
        return np.divide(numerator, denominator)
    
    def get_lci(self):
        """
        Method to generate lci index. The formula used:
        (self.b8 - self.b5) / (self.b8 + self.b4)
        :return: Array with index values
        """
        numerator = np.subtract(self.b8, self.b5)
        denominator = np.add(self.b8, self.b4)
        
        return np.divide(numerator, denominator)
    
    def get_ndmi(self):
        """
        Method to generate ndmi index. The formula used:
        (self.b8 - self.b11) / (self.b8 + self.b11)
        :return: Array with index values
        """
        numerator = np.subtract(self.b8, self.b11)
        denominator = np.add(self.b8, self.b11)
        
        return np.divide(numerator, denominator)
    
    def get_ndvi(self):
        """
        Method to generate ndvi index. The formula used:
        (self.b8 - self.b4 / (self.b8 + self.b4))
        :return: Array with index values
        """
        numerator = np.subtract(self.b8, self.b4)
        denominator = np.add(self.b8, self.b4)
        
        return np.divide(numerator, denominator)
    
    def get_sci(self):
        """
        Method to generate sci index. The formula used:
        (self.b11 - self.b8 / (self.b11 + self.b8))
        :return: Array with index values
        """
        numerator = np.subtract(self.b11, self.b8)
        denominator = np.add(self.b11, self.b8)
        
        return np.divide(numerator, denominator)
    
    def get_savi(self):
        """
        Method to generate savi index. The formula used:
        (self.b8 - self.b4) / (self.b8 + self.b4 + 0.428) * (1.0 + 0.428)
        :return: Array with index values
        """
        numerator = np.subtract(self.b8, self.b4)
        denominator = np.multiply(np.add(np.add(self.b8, self.b4), 0.428), np.add(1.0, 0.428))
        
        return np.divide(numerator, denominator)
    
    def get_ctvi(self):
        """
        Method to generate ctvi index. The formula used:
        (((b4 - b3) / (b4 + b3)) + 0.5) / abs(((b4 - b3) / (b4 + b3)) + 0.5) * sqrt(abs((((b4 - b3) / (b4 + b3))) + 0.5))
        :return: Array with index values
        """
        p1 = np.add(np.divide(np.subtract(self.b4, self.b3), np.add(self.b4, self.b3)), 0.5)
        p2 = np.abs(np.add(np.divide(np.subtract(self.b4, self.b3), np.add(self.b4, self.b3)), 0.5))
        p3 = np.sqrt(np.abs(np.add(np.divide(np.subtract(self.b4, self.b3), np.add(self.b4, self.b3)), 0.5)))
        
        return np.multiply(np.divide(p1, p2), p3)
    
    def get_indexes(self, indexes=None):
        """
        Method to export all indexes as tiff in indexes folder
        :param indexes: indexes to be exported. if values is none the function will export all (16) indexes
        :return: tiff files exported
        """
        if indexes is None:
            indexes = ['atsavi', 'ari', 'avi', 'arvi', 'chlgreen',
                       'fe3', 'fo', 'gvmi', 'gndvi', 'lci', 'ndmi',
                       'evi', 'ndvi', 'sci', 'savi', 'ctvi']
        for index in indexes:
            self.export_index(index)
    
    def export_index(self, index):
        """
        Method to run each get index a time and export to minimize use of memory
        :param key: index to run locale function
        :return: Tiff exported
        """
        self.handler.create_folder('../indexes')
        
        def export_atsavi():
            return self.get_atsavi()
        
        def export_ari():
            return self.get_ari()
        
        def export_avi():
            return self.get_avi()
        
        def export_arvi():
            return self.get_arvi()
        
        def export_chlgreen():
            return self.get_chlgreen()
        
        def export_fe3():
            return self.get_fe3()
        
        def export_fo():
            return self.get_fo()
        
        def export_gvmi():
            return self.get_gvmi()
        
        def export_gndvi():
            return self.get_gndvi()
        
        def export_lci():
            return self.get_lci()
        
        def export_ndmi():
            return self.get_ndmi()
        
        def export_evi():
            return self.get_evi()
        
        def export_ndvi():
            return self.get_ndvi()
        
        def export_sci():
            return self.get_sci()
        
        def export_savi():
            return self.get_savi()
        
        def export_ctvi():
            return self.get_ctvi()
        
        self.profile.update({'dtype': 'float32', 'count': 1})
        with rasterio.open(self.handler.get_abs_path('indexes') + '/' + self.date + '_' + index + '.tif', 'w',
                           **self.profile) as dst:
            dst.write(locals()['export_' + index](), 1)


class Postgresql:
    def __init__(self):
        self.conn = self.conn_postgres()
        self.create_table()
    
    def conn_postgres(self):
        """
        Method to connect with PostgreSQL Database
        :return: Conection with Database
        """
        return psycopg2.connect(host=config('host'), database=config('database'), user=config('user'),
                                password=config('password'))
    
    def disconn(self):
        """
        Method to disconnet database
        :return: database closed
        """
        self.conn.close()
    
    def drop_table(self, table):
        """
        Method to drop table in PostgreSQL
        :param table: Name of table to be truncate
        :return: Table droped
        """
        cur = self.conn.cursor()
        sql = f'drop table if exists {table}'
        cur.execute(sql)
        self.conn.commit()
        cur.close()
    
    def create_table(self, tables=None):
        """
        Method to create table from a dictionary
        :param tables: Dictionary with keys from name of tables and keys of fields and type fields
        :return: Tables created in PostgreSQL database
        """
        if tables is None:
            tables = [{"table": "stats", "fields":
                {"id": 'serial primary key',
                 "data": 'text not null',
                 "index": 'text not null',
                 "raster_array": 'text not null',
                 "raster_profile": 'text not null',
                 "zonal_stats": 'text not null',
                 "profile": 'text not null',
                 }},
                      {"table": "stats_estimated", "fields":
                          {"id": 'serial primary key',
                           "data": 'text not null',
                           "index": 'text not null',
                           "raster_array": 'text not null',
                           "raster_profile": 'text not null',
                           "zonal_stats": 'text not null',
                           "profile": 'text not null',
                           }}]
        
        for table in tables:
            self.drop_table(table["table"])
            fields = [(i + ' ' + k) for i, k in table["fields"].items()]
            cur = self.conn.cursor()
            cur.execute(f'create table if not exists {table["table"]} ({",".join(fields)})')
            self.conn.commit()
            cur.close()


class Statistics:
    def __init__(self):
        self.database = Postgresql()
##TODO: Criar tabela com campos id, zonal_stats(dicionario), profile, bandas, data


class Diagnosys:
##TODO: Preparar saidas para html
    def __init__(self, html):
        self.text = html
        self.run = self.print_html_doc()
        
    def print_html_doc(self):
        env = Environment(loader=FileSystemLoader(os.getcwd()),
                          trim_blocks=True)
        template = env.get_template('./templates/index.html')
        
        web = template.render(title='estatisticas', run=self.text)
        with open("./templates/plots.html", "w") as fh:
            fh.write(web)

        return webbrowser.open('file://' + os.path.realpath('./templates/index.html'))




if __name__ == '__main__':
    ## Download de imagens desde 15-07-2021
    # sd = SentinelDownloader()
    # sd.authenticate_gee()
    # sd.download_sentinel('teste.gpkg', 'gpkg', 'id_pk', '2021-07-15')
    ## Geração dos indices para os downloads de imagens
    # sh = SentinelHandler()
    # for i in sh.path_files(sh.get_abs_path('merged')):
    #     si = SentinelIndexes(i)
    #     si.get_indexes()
    ## Criação de tabelas e inserção de dados no banco
    stats = Statistics()
    text = 'este é um texto aleatorio'
    run = Diagnosys(text)

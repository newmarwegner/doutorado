import os
import shutil
import ee
from decouple import config


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
            
class SentinelDownloader:
    def __init__(self):
        pass
    
    def authenticate_gee(self):
        """
        Method to authenticate gee to download images
        """
        credentials = ee.ServiceAccountCredentials(config('service_account'), config('credentials_api'))
        
        return ee.Initialize(credentials)



if __name__ == '__main__':
    sh = SentinelHandler()
    sd = SentinelDownloader()
    sd.authenticate_gee()

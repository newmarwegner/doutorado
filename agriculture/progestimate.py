import glob
import numpy as np
import rasterio

def filter_images(path, months, years=None):
    files = glob.glob(path + '*_ndvi.tif')
    order = lambda x: x.split('/')[-1].split('_')[0]
    files.sort(key=order)

    images = []
    for file in files:
        year_file = int(file.split('/')[-1].split('_')[0][0:4])
        month_file = int(file.split('/')[-1].split('_')[0][4:6])
        year_month = int(str(year_file) + str(month_file))
        if not years:
            if month_file in months:
                images.append(file)
        else:
            if year_month in years:
                images.append(file)

    return sorted(images, key=order)

def create_np_images(list_images,output_file):

    database = []
    for i in list_images:
        with rasterio.open(i) as dst:
            src = dst.read()
            database.append(src[0])

    with open(f'./templates/{output_file}.npy', 'wb') as f:
        np.save(f, database)



if __name__ == '__main__':
    ## variables
    images_folder = '/home/newmar/Downloads/projetos_python/doutorado/indexes/'
    month_global = [9, 10, 11, 12, 1, 2, 3]
    monthyear_corn = [201812, 20191, 20192, 20193, 202011, 202012, 20211, 20212, 20213]
    monthyear_soybean = [201911, 201912, 20201, 20202, 20203, 202111, 202112, 20221, 20222, 20223]

    ## filter datasets to use as Train dataset
    list_images = filter_images(images_folder, month_global)
    create_np_images(list_images, 'dados_global')

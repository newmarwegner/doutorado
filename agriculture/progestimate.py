import os

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from sentineldownloader import PlotHtml


class IndexPrognosys:
    def __init__(self):
        self.db = PlotHtml()

    def table(self):
        return self.db.raster_from_db()


if __name__ == '__main__':
    ip = IndexPrognosys()
    print(ip.table())
    #print('olá')
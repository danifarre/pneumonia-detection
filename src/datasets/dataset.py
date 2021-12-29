from os.path import abspath
import pandas as pd
import numpy as np


class Dataset(object):

    def __init__(self, path, image_size):
        self.ds_path = path
        self.image_size = image_size

        self.image = 'image'
        self.path = 'path'
        self.diagnosis = 'diagnosis'

        self.train_df = pd.read_csv(abspath(self.ds_path + '/train.csv'), header=0)
        self.test_df = pd.read_csv(abspath(self.ds_path + '/test.csv'), header=0)
        self.val_df = pd.read_csv(abspath(self.ds_path + '/val.csv'), header=0)

        self.split_train = None
        self.iter_train = None

    def __iter__(self):
        return self

    def __next__(self):
        dataset = next(self.iter_train)
        return self._load_data(dataset)

    def get_train_df(self):
        return self.train_df.copy()

    def get_test_df(self):
        return self.test_df.copy()

    def get_val_df(self):
        return self.val_df.copy()

    def load_train_data(self):
        return self._load_data(self.train_df.copy())

    def load_test_data(self):
        return self._load_data(self.test_df.copy())

    def load_val_data(self):
        return self._load_data(self.val_df.copy())

    def split(self, sections):
        self.split_train = np.array_split(self.train_df, sections)
        self.iter_train = iter(self.split_train)
        return self.__iter__()

    def _load_data(self, ds):
        pass

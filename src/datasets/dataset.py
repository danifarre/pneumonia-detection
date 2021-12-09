from os.path import abspath
import pandas as pd


class Dataset(object):

    def __init__(self, path):
        self.ds_path = path

        self.image = 'image'
        self.path = 'path'
        self.diagnosis = 'diagnosis'

        self.train_df = pd.read_csv(abspath(self.ds_path + '/train.csv'), header=0)
        self.test_df = pd.read_csv(abspath(self.ds_path + '/test.csv'), header=0)
        self.val_df = pd.read_csv(abspath(self.ds_path + '/val.csv'), header=0)

    def get_train_df(self):
        return self.train_df.copy()

    def get_test_df(self):
        return self.test_df.copy()

    def get_val_df(self):
        return self.test_df.copy()

    def load_train_data(self, image_size):
        return self._load_data(self.train_df.copy(), image_size)

    def load_test_data(self, image_size):
        return self._load_data(self.test_df.copy(), image_size)

    def load_val_data(self, image_size):
        return self._load_data(self.val_df.copy(), image_size)

    def _load_data(self, df, image_size):
        pass

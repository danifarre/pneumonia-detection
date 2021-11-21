import pydicom
import cv2

import numpy as np
import pandas as pd

class PneumoniaDetectionChallenge:

    def __init__(self, path, seed=1):
        self.path = path
        self.labels_df = self.__normalize(pd.read_csv(self.path  + 'stage_2_train_labels.csv', header=0, usecols=['patientId', 'Target']))
        self.train_labels_df  = self.labels_df.sample(frac=0.75, random_state=seed)
        self.test_labels_df = self.labels_df.drop(self.train_labels_df.index)
        self.train_labels_df = pd.concat([self.train_labels_df[self.train_labels_df["Target"] == 1], self.train_labels_df[self.train_labels_df["Target"] == 0].sample(frac=0.35, random_state=seed)])

    def get_labels_df(self):
        return self.labels_df.copy()

    def get_train_labels_df(self):
        return self.train_labels_df.copy()

    def get_test_labels_df(self):
        return self.test_labels_df.copy()

    def load_train_data(self):
        return self.__load_data(self.train_labels_df.copy())

    def load_test_data(self):
        return self.__load_data(self.test_labels_df.copy())

    def __load_data(self, df):
        df["path"] = df['patientId'].map(lambda _: self.path + 'stage_2_train_images/' + _ + '.dcm')

        patients_id = list(df["patientId"])
        images_paths = list(df['path'])
        x_train = list()
        y_train = list(df['Target'])

        for path in images_paths:
            img = pydicom.read_file(path).pixel_array
            img = cv2.resize(img, (256, 256))
            x_train.append(img)

        return np.array(x_train), np.array(y_train), patients_id

    def load_train_metadata(self):
        #f = pydicom.read_file(filePath, stop_before_pixels=True)
        pass

    def load_test_metadata(self):
        pass

    def __normalize(self, df):
        df = df.groupby("patientId", as_index=False).sum()
        df['Target'] = np.where(df['Target'] >= 1, 1, 0)
        return df

import pydicom
import cv2
import numpy as np

from src.datasets.dataset import Dataset


class PneumoniaDetectionChallenge(Dataset):

    def __init__(self, path, image_size):
        super().__init__(path, image_size)

    def _load_data(self, df):
        images = list(df[self.image])
        images_paths = list(df[self.path])
        x_train = list()
        y_train = list(df[self.diagnosis])

        for path in images_paths:
            img = pydicom.read_file(path).pixel_array
            img = cv2.resize(img, self.image_size)
            x_train.append(img)

        x_train = np.array(x_train)
        x_train = np.repeat(x_train[..., np.newaxis], 3, -1)

        return x_train, np.array(y_train), images


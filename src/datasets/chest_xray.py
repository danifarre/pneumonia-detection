import cv2
import numpy as np

from src.datasets.dataset import Dataset


class ChestXRay(Dataset):

    def __init__(self, path, image_size):
        super().__init__(path, image_size)

    def _load_data(self, df):
        images = list(df[self.image])
        images_paths = list(df[self.path])
        x_train = list()
        y_train = list(df[self.diagnosis])

        for path in images_paths:
            img = cv2.imread(path)
            img = cv2.resize(img, self.image_size)
            x_train.append(img)

        return np.array(x_train), np.array(y_train), images

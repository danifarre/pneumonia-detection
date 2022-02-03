import os
from shutil import copyfile

import pandas as pd
import numpy as np

from configuration.paths import *

SEED = 1


def normalize(ds):
    ds = ds.groupby("patientId", as_index=False).sum()
    ds['diagnosis'] = np.where(ds['Target'] >= 1, 1, 0)
    ds = ds.rename(columns={'patientId': 'image'})
    ds = ds[['image', 'diagnosis']]
    return ds


def copy_files(src, dst):
    for p1, p2 in zip(src, dst):
        copyfile(p1, p2)


if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    if not os.path.exists(DATASET_PNEUMONIA_DETECTION_CHALLENGE_PATH):
        os.mkdir(DATASET_PNEUMONIA_DETECTION_CHALLENGE_PATH)
    else:
        exit()

    if not os.path.exists(DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH):
        os.makedirs(DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH)
        os.makedirs(DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH + '/train')
        os.makedirs(DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH + '/test')
        os.makedirs(DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH + '/val')
    else:
        exit()

    cols = ['image', 'path', 'diagnosis']

    ds = normalize(pd.read_csv(abspath(RSNA_PNEUMONIA_DETECTION_CHALLENGE_PATH + '/stage_2_train_labels.csv'),
                               header=0, usecols=['patientId', 'Target']))
    ds = ds.sample(frac=1, random_state=SEED)
    ds["l_path"] = ds['image'].map(lambda _: RSNA_PNEUMONIA_DETECTION_CHALLENGE_PATH + '/stage_2_train_images/' + _ + '.dcm')
    ds['extended'] = False

    train_ds = ds.sample(frac=0.75, random_state=SEED)

    val_ds = train_ds.sample(frac=0.001, random_state=SEED)

    test_ds = ds.drop(train_ds.index)

    train_ds = pd.concat([train_ds[train_ds["diagnosis"] == 1], train_ds[train_ds["diagnosis"] == 0].sample(frac=0.30, random_state=SEED)])

    train_ds['path'] = train_ds['image'].map(lambda _: DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH + '/train/' + _ + '.dcm')
    train_ds.to_csv(abspath(DATASET_PNEUMONIA_DETECTION_CHALLENGE_PATH + '/train.csv'), index=False, columns=cols)

    test_ds['path'] = test_ds['image'].map(lambda _: DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH + '/test/' + _ + '.dcm')
    test_ds = pd.concat([test_ds[test_ds["diagnosis"] == 1], test_ds[test_ds["diagnosis"] == 0].sample(frac=0.30, random_state=SEED)])

    test_ds.to_csv(abspath(DATASET_PNEUMONIA_DETECTION_CHALLENGE_PATH + '/test.csv'), index=False, columns=cols)

    val_ds['path'] = val_ds['image'].map(lambda _: DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH + '/val/' + _ + '.dcm')
    val_ds.to_csv(abspath(DATASET_PNEUMONIA_DETECTION_CHALLENGE_PATH + '/val.csv'), index=False, columns=cols)

    copy_files(list(train_ds['l_path']), list(train_ds['path']))
    copy_files(list(test_ds['l_path']), list(test_ds['path']))
    copy_files(list(val_ds['l_path']), list(val_ds['path']))

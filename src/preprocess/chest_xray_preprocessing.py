import os
from shutil import copyfile

import pandas as pd

from configuration.paths import *

SEED = 1


def charge_dataset(normal, pneumonia, type):
    ds1 = pd.DataFrame(os.listdir(normal), columns=['image'])
    ds1["path"] = ds1['image'].map(lambda _: abspath(DATASET_CHEST_XRAY_DATA_PATH + '/' + type + '/' + _))
    ds1["l_path"] = ds1['image'].map(lambda _: abspath(normal + '/' +_))
    ds1['image'] = ds1['image'].map(lambda _: _[:-5])
    ds1['diagnosis'] = 0
    ds1['extended'] = False

    ds2 = pd.DataFrame(os.listdir(pneumonia), columns=['image'])
    ds2["path"] = ds2['image'].map(lambda _: abspath(DATASET_CHEST_XRAY_DATA_PATH + '/' + type + '/' + _))
    ds2["l_path"] = ds2['image'].map(lambda _: abspath(pneumonia + '/' +_))
    ds2['image'] = ds2['image'].map(lambda _: _[:-5])
    ds2['diagnosis'] = 1
    ds2['extended'] = False

    ds = pd.concat([ds1, ds2])
    ds = ds.sample(frac=1, random_state=SEED)

    return ds


def copy_files(src, dst):
    for p1, p2 in zip(src, dst):
        copyfile(p1, p2)


if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    if not os.path.exists(DATASET_CHEST_XRAY_PATH):
        os.mkdir(DATASET_CHEST_XRAY_PATH)
    else:
        exit()

    if not os.path.exists(DATASET_CHEST_XRAY_DATA_PATH):
        os.makedirs(DATASET_CHEST_XRAY_DATA_PATH)
        os.makedirs(DATASET_CHEST_XRAY_DATA_PATH + '/train')
        os.makedirs(DATASET_CHEST_XRAY_DATA_PATH + '/test')
        os.makedirs(DATASET_CHEST_XRAY_DATA_PATH + '/val')
    else:
        exit()

    cols = ['image', 'path', 'diagnosis']

    train_ds = charge_dataset(TRAIN_NORMAL_PATH, TRAIN_PNEUMONIA_PATH, 'train')
    train_ds.to_csv(abspath(DATASET_CHEST_XRAY_PATH + '/train.csv'), index=False, columns=cols)

    test_ds = charge_dataset(TEST_NORMAL_PATH, TEST_PNEUMONIA_PATH, 'test')
    test_ds.to_csv(abspath(DATASET_CHEST_XRAY_PATH + '/test.csv'), index=False, columns=cols)

    val_ds = charge_dataset(VAL_NORMAL_PATH, VAL_PNEUMONIA_PATH, 'val')
    val_ds.to_csv(abspath(DATASET_CHEST_XRAY_PATH + '/val.csv'), index=False, columns=cols)

    copy_files(list(train_ds['l_path']), list(train_ds['path']))
    copy_files(list(test_ds['l_path']), list(test_ds['path']))
    copy_files(list(val_ds['l_path']), list(val_ds['path']))
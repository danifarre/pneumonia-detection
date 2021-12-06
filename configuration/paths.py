from os.path import abspath

# General
RSNA_PNEUMONIA_DETECTION_CHALLENGE_PATH = abspath('kaggle/rsna-pneumonia-detection-challenge/')
CHEST_XRAY_PATH = abspath("kaggle/chest_xray")

# chest_xray
TRAIN_NORMAL_PATH = abspath(CHEST_XRAY_PATH + '/train/NORMAL/')
TRAIN_PNEUMONIA_PATH = abspath(CHEST_XRAY_PATH + '/train/PNEUMONIA/')
TEST_NORMAL_PATH = abspath(CHEST_XRAY_PATH + '/test/NORMAL/')
TEST_PNEUMONIA_PATH = abspath(CHEST_XRAY_PATH + '/test/PNEUMONIA/')
VAL_NORMAL_PATH = abspath(CHEST_XRAY_PATH + '/val/NORMAL/')
VAL_PNEUMONIA_PATH = abspath(CHEST_XRAY_PATH + '/val/PNEUMONIA/')

# Datasets
DATASET_PATH = abspath('datasets')
DATASET_CHEST_XRAY_PATH = abspath(DATASET_PATH + "/chest_xray")
DATASET_CHEST_XRAY_DATA_PATH = abspath(DATASET_CHEST_XRAY_PATH + "/data")

DATASET_PNEUMONIA_DETECTION_CHALLENGE_PATH = abspath(DATASET_PATH + "/pneumonia-detection-challenge")
DATASET_PNEUMONIA_DETECTION_CHALLENGE_DATA_PATH = abspath(DATASET_PNEUMONIA_DETECTION_CHALLENGE_PATH + "/data")


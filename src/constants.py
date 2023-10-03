import numpy as np

X_RANGE_EulerGPU = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
Y_RANGE_EulerGPU = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
THETA_RANGE_EulerGPU = np.linspace(0, 360, 360, dtype=np.float32)
BATCH_SIZE = 200
DEVICE = 'cuda:0'
TARGET_DIM = (208, 96, 56)
SAVE_PATH = '/home/alicia/data_personal/registered/euler_grid_search_full-v3'
SOURCE_PATH = '/data1/jungsoo/data/2023-01-16-reg-data/h5_resize'
TRAIN_DS = [
        "2022-01-09-01",
        "2022-01-23-04",
        "2022-01-27-04",
        "2022-06-14-01",
        "2022-07-15-06",
        "2022-01-17-01",
        "2022-01-27-01",
        "2022-03-16-02",
        "2022-06-28-01"
]
VALID_DS = [
        "2022-02-16-04",
        "2022-04-05-01",
        "2022-07-20-01",
        "2022-03-22-01",
        "2022-04-12-04",
        "2022-07-26-01"
]
TEST_DS = [
        "2022-04-14-04",
        "2022-04-18-04",
        "2022-08-02-01"
]
DS_DICT = {
           'train': TRAIN_DS,
           'valid': VALID_DS,
           'test': TEST_DS}

from julia.api import Julia
from scipy import ndimage
from tqdm import tqdm
import SimpleITK as sitk
import glob
import h5py
import numpy as np
import os
import re
import shutil
import yaml

jl = Julia(compiled_modules=False)
jl.eval('include("/home/alicia/notebook/register/adjust.jl")')
julia_resize_func_new = jl.eval("adjust_image_cm")
#TARGET_DIM = (208, 96, 56)
TARGET_DIM = (290, 120, 64)
manifest = {
        "train": ["/data1/prj_kfc/data_processed/2022-03-16-02_output",
            "/data1/prj_kfc/data_processed/2022-01-09-01_output",
            "/data1/prj_kfc/data_processed/2022-01-17-01-SWF415-animal1_output",
            "/data1/prj_kfc/data_processed/2022-01-23-04_output",
            "/data1/prj_kfc/data_processed/2022-01-27-01_output",
            "/data1/prj_kfc/data_processed/2022-01-27-04_output",
            "/data1/prj_neuropal/data_processed/2022-06-14-01_output",
            "/data1/prj_neuropal/data_processed/2022-06-28-01_output",
            "/data1/prj_neuropal/data_processed/2022-07-15-06_output"],
        "valid": ["/data1/prj_kfc/data_processed/2022-04-05-01_output",
            "/data1/prj_kfc/data_processed/2022-04-12-04_output",
            "/data1/prj_kfc/data_processed/2022-02-16-04_output",
            "/data1/prj_kfc/data_processed/2022-03-22-01_output",
            "/data1/prj_neuropal/data_processed/2022-07-20-01_output",
            "/data1/prj_neuropal/data_processed/2022-07-26-01_output"],
        "test": ["/data1/prj_kfc/data_processed/2022-04-14-04_output",
            "/data1/prj_kfc/data_processed/2022-04-18-04_output",
            "/data1/prj_neuropal/data_processed/2022-08-02-01_output"]
        }

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
DATASETS_SPLIT_DICT = {
           'train': TRAIN_DS,
           'valid': VALID_DS,
           'test': TEST_DS
        }


def filter_image(image, threshold):
    filtered_image = image - threshold
    filtered_image[filtered_image < 0] = 0

    return filtered_image


def get_cropped_image(image_T, center, projection=2):

    if projection >= 0:
        return julia_resize_func_new(
                image_T,
                center,
                TARGET_DIM).max(projection)
    elif projection == -1:
        return julia_resize_func_new(
                image_T,
                center,
                TARGET_DIM)


def get_image_T(image_path):

    '''Given the path, read image of .nrrd format as numpy array
    '''

    image_nrrd = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image_nrrd)
    if image.ndim == 4:
        image = image.squeeze()
    image_T = np.transpose(image, (2,1,0))
    return image_T


def get_image_CM(image_T):

    '''Taking image of shape in the order (x, y, z) and find its center of mass
    after filtering
    '''

    # subtract the median pixel from the image; zero out the negative pixels
    image_T_wo_background = image_T - np.median(image_T)
    image_T_wo_background[image_T_wo_background < 0] = 0
    x, y, z = ndimage.measurements.center_of_mass(image_T_wo_background)
    return (round(x), round(y), round(z))


def io_config_yaml(ds_base,
                   train_datasets,
                   valid_datasets,
                   test_datasets,
                   outfile):

    data = {
        "dataset": {
            "train": {
                "dir": [],
                "format": "h5",
                "labeled": False
            },
            "valid": {
                "dir": [],
                "format": "h5",
                "labeled": False
            },
            "test": {
                "dir": [],
                "format": "h5",
                "labeled": False
            },
            "type": "paired",
            "moving_image_shape": [208,96,56],
            "fixed_image_shape": [208,96,56]
        },
        "train": {
            "method": "ddf",
            "backbone": {
                "name": "local",
                "num_channel_initial": 16,
                "extract_levels": [0, 1, 2, 3]
            },
            "loss": {
                "image": {
                    "name": "lncc",
                    "kernel_size": 16,
                    "weight": 1.0
                },
                "regularization": {
                    "weight": 0.2,
                    "name": "bending"
                }
            },
            "preprocess": {
                "data_augmentation": {
                    "name": "affine"
                },
                "batch_size": 4,
                "shuffle_buffer_num_batch": 2,
                "num_parallel_calls": -1
            },
            "optimizer": {
                "name": "Adam",
                "learning_rate": 1.0e-3
            },
            "epochs": 300,
            "save_period": 1
        }
    }

    # fill in the datasets
    data['dataset']['train']['dir'] = [f'{ds_base}/{train_ds}' for train_ds in train_datasets]
    data['dataset']['valid']['dir'] = [f'{ds_base}/{valid_ds}' for valid_ds in valid_datasets]
    data['dataset']['test']['dir'] = [f'{ds_base}/{test_ds}' for test_ds in test_datasets]

    with open('data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def copy_raw_nrrd_images(target_dim):

    dir_save_to = \
            "/home/alicia/data_personal/regnet_dataset/2023-01-16-reg-data/h5_resize"

    for ds_type, ds_paths in manifest.items():

        for ds_path in tqdm(ds_paths):

            match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}', ds_path)
            if match:
                ds_date = match.group()

            destination = f"{dir_save_to}/{ds_type}/{ds_date}"
            if not os.path.exists(destination):
                os.makedirs(destination)

            # create moving images h5 file folder
            hdf5_m_file = h5py.File(f'{destination}/moving_images.h5', 'w')

            problem_path = \
                f"/data1/jungsoo/data/2023-01-16-reg-data/h5_resize/{ds_type}/{ds_date}"
            with h5py.File(f"{problem_path}/fixed_images.h5") as f:
                problems = list(f.keys())

            for problem in problems:
                t_moving, _ = problem.split("to")
                t_moving = t_moving.zfill(4)
                nrrd_image_path = \
                        glob.glob(f"{ds_path}/NRRD_filtered/*t{t_moving}*")[0]
                nrrd_image = sitk.ReadImage(nrrd_image_path)
                arr_image = sitk.GetArrayFromImage(nrrd_image)
                # flip the image dimension order from (z, y, x) to (x, y, z)
                arr_image_T = np.transpose(arr_image, (2, 1, 0))
                # adjust image size
                julia_resize_func = jl.eval("adjust_img")
                resized_arr_image = julia_resize_func(arr_image_T, target_dim)
                hdf5_m_file.create_dataset(problem, data = resized_arr_image)

                # copy fixed image from Jungsoo's directory
                shutil.copy(f"{problem_path}/fixed_images.h5",
                        f"{destination}/fixed_images.h5")
            hdf5_m_file.close()


def calculate_gncc(fixed, moving):

    mu_f = np.mean(fixed)
    mu_m = np.mean(moving)
    a = np.sum(abs(fixed - mu_f) * abs(moving - mu_m))
    b = np.sqrt(np.sum((fixed - mu_f) ** 2) * np.sum((moving - mu_m) ** 2))
    return a / b


def calculate_ncc(fixed, moving):
    assert fixed.shape == moving.shape

    med_f = np.median(np.max(fixed, axis=2))
    med_m = np.median(np.max(moving, axis=2))
    fixed_new = np.maximum(fixed - med_f, 0)
    moving_new = np.maximum(moving - med_m, 0)

    mu_f = np.mean(fixed_new)
    mu_m = np.mean(moving_new)
    fixed_new = fixed_new / mu_f - 1
    moving_new = moving_new / mu_m - 1
    numerator = np.sum(fixed_new * moving_new)
    denominator = np.sqrt(np.sum(fixed_new ** 2) * np.sum(moving_new ** 2))

    return numerator / denominator


def locate_directory(ds_date):

    '''
    Given the date when the dataset was collected, this function locates which
    directory this data file can be found
    '''

    neuropal_dir = '/home/alicia/data_prj_neuropal/data_processed'
    non_neuropal_dir = '/home/alicia/data_prj_kfc/data_processed'
    for directory in os.listdir(neuropal_dir):
        if ds_date in directory:
            return os.path.join(neuropal_dir, directory)

    for directory in os.listdir(non_neuropal_dir):
        if ds_date in directory:
            return os.path.join(non_neuropal_dir, directory)

    raise Exception(f'Dataset {ds_date} cannot be founed.')

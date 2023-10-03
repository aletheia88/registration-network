"""Utilties for registering channel-1-images to channel-2-images"""

from euler_gpu.grid_search import grid_search
from euler_gpu.preprocess import initialize, max_intensity_projection_and_downsample
from euler_gpu.transform import transform_image_3d, translate_along_z
from preprocess import filter_and_crop
from tqdm import tqdm
from util import DATASETS_SPLIT_DICT, get_image_T, get_cropped_image, get_image_CM, locate_directory
import glob
import h5py
import json
import numpy as np
import os
import torch


def get_elastix_solutions():

    with open('jungsoo_registration_problems.json', 'r') as f:
        registration_problem_dict = json.load(f)

    for dataset_type_n_name, problems in registration_problem_dict.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')
        save_path = f'{save_directory}/{dataset_type}/{dataset_name}'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_elastix_file = h5py.File(f'{save_path}/ch1-elastix_images.h5', 'w')
        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            t_ch1, t_ch2 = problem.split('to')
            ch2_image_paths = glob.glob(
                    f'{dataset_path}/Registered_RG/{t_ch1}to{t_ch1}/result.0.R2.nrrd'
            )
            if len(ch2_image_paths) > 0:
                ch2_image_path = ch2_image_paths[0]
                ch2_image_T = get_image_T(ch2_image_path)
                ch2_image_median = np.median(ch2_image_T)
                resized_ch2_image_xyz = filter_and_crop(ch2_image_T,
                            ch2_image_median)

                if f'{t_ch1}to{t_ch1}' not in hdf5_elastix_file:
                    hdf5_elastix_file.create_dataset(
                                f'{t_ch1}to{t_ch1}',
                                data=resized_ch2_image_xyz
                )

    hdf5_elastix_file.close()


def preprocess(downsample_factor,
               resolution_factor,
               x_translation_range,
               y_translation_range,
               z_translation_range,
               theta_rotation_range,
               batch_size,
               device_name,
               save_directory):

    with open('jungsoo_registration_problems.json', 'r') as f:
        registration_problem_dict = json.load(f)

    # Initialize memory dictionaries for performing Euler transformation
    width = int(208 / downsample_factor)
    height = int(96 / downsample_factor)
    z_dim = 56
    memory_dict = initialize(
                            np.zeros((width, height)).astype(np.float32),
                            np.zeros((width, height)).astype(np.float32),
                            x_translation_range,
                            y_translation_range,
                            theta_rotation_range,
                            batch_size,
                            device_name
    )
    print(f'memory_dict initialized!')
    memory_dict_ = initialize(
                            np.zeros((208, 96)).astype(np.float32),
                            np.zeros((208, 96)).astype(np.float32),
                            torch.zeros(z_dim, device=device_name),
                            torch.zeros(z_dim, device=device_name),
                            torch.zeros(z_dim, device=device_name),
                            z_dim,
                            device_name
    )

    for dataset_type_n_name, problems in registration_problem_dict.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')
        save_path = f'{save_directory}/{dataset_type}/{dataset_name}'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_ch1_file = h5py.File(f'{save_path}/ch1_images.h5', 'w')
        hdf5_ch2_file = h5py.File(f'{save_path}/ch2_images.h5', 'w')

        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            t_ch1, t_ch2 = problem.split('to')
            t_ch1_4 = t_ch1.zfill(4)
            t_ch2_4 = t_ch2.zfill(4)
            ch2_image_path = glob.glob(
                    f'{dataset_path}/NRRD/*_t{t_ch2_4}_ch2.nrrd'
            )[0]
            ch1_image_path = glob.glob(
                    f'{dataset_path}/NRRD/*_t{t_ch1_4}_ch1.nrrd'
            )[0]

            ch2_image_T = get_image_T(ch2_image_path)
            ch2_image_median = np.median(ch2_image_T)
            ch1_image_T = get_image_T(ch1_image_path)
            ch1_image_median = np.median(ch1_image_T)

            resized_ch2_image_xyz = filter_and_crop(ch2_image_T,
                        ch2_image_median)
            resized_ch1_image_xyz = filter_and_crop(ch1_image_T,
                        ch1_image_median)

            downsampled_resized_ch2_image_xy = \
                        max_intensity_projection_and_downsample(
                                resized_ch2_image_xyz,
                                downsample_factor).astype(np.float32)
            downsampled_resized_ch1_image_xy = \
                        max_intensity_projection_and_downsample(
                                resized_ch1_image_xyz,
                                downsample_factor).astype(np.float32)
            memory_dict["fixed_images_repeated"] = torch.tensor(
                        downsampled_resized_ch2_image_xy,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            memory_dict["moving_images_repeated"] = torch.tensor(
                        downsampled_resized_ch1_image_xy,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            best_score, best_transformation = grid_search(memory_dict)
            transformed_ch1_image_xyz = transform_image_3d(
                        resized_ch1_image_xyz,
                        memory_dict_,
                        best_transformation,
                        device_name)
            dz, gncc, final_ch1_image_xyz = translate_along_z(
                        z_translation_range,
                        resized_ch2_image_xyz,
                        transformed_ch1_image_xyz,
                        ch1_image_median)
            hdf5_ch1_file.create_dataset(
                        f'{t_ch1}to{t_ch2}',
                        data=final_ch1_image_xyz
            )
            hdf5_ch2_file.create_dataset(
                        f'{t_ch1}to{t_ch2}',
                        data=resized_ch2_image_xyz
            )

        hdf5_ch1_file.close()
        hdf5_ch2_file.close()


if __name__ == "__main__":

    resolution_factor = 1
    x_translation_range = np.linspace(-1, 1, int(100/resolution_factor),
            dtype=np.float32)
    y_translation_range = np.linspace(-1, 1, int(100/resolution_factor),
            dtype=np.float32)
    theta_rotation_range = np.linspace(0, 360, int(360/resolution_factor),
            dtype=np.float32)
    z_translation_range = range(-50, 50)
    batch_size = 200
    device_name = torch.device("cuda:1")
    downsample_factor = 1
    eulergpu_version = 'd1_r1_xy1'
    save_directory = \
            f'/home/alicia/data_personal/regnet_dataset/2023-01-16-ch1toch2/{eulergpu_version}'
    '''preprocess(downsample_factor, resolution_factor, x_translation_range,
        y_translation_range, z_translation_range, theta_rotation_range,
        batch_size, device_name, save_directory)'''
    get_elastix_solutions()


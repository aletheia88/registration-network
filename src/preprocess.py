"""Utilities for preprocessing given image pairs by doing Euler transformation
on them"""

from euler_gpu.grid_search import grid_search
from euler_gpu.preprocess import initialize, max_intensity_projection_and_downsample
from euler_gpu.transform import transform_image_3d, translate_along_z
from tqdm import tqdm
from util import DATASETS_SPLIT_DICT, get_image_T, get_cropped_image, get_image_CM, locate_directory, calculate_gncc
import glob
import h5py
import json
import numpy as np
import os
import torch


def preprocess_raw():

    """Preprocess (crop only) and generated pairs of fixed and moving images
    """
    source_path = '/data1/jungsoo/data/2023-01-16-reg-data/h5_resize'
    save_directory = \
        '/home/alicia/data_personal/regnet_dataset/2023-01-16_raw_crop-v1'
    with open('jungsoo_registration_problems.json', 'r') as f:
        registration_problem_dict = json.load(f)

    for dataset_type_n_name, problems in registration_problem_dict.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')
        save_path = f'{save_directory}/{dataset_type}/{dataset_name}'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            t_moving, t_fixed = problem.split('to')
            t_moving_4 = t_moving.zfill(4)
            t_fixed_4 = t_fixed.zfill(4)
            fixed_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
            )[0]
            moving_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
            )[0]

            fixed_image_T = get_image_T(fixed_image_path)
            fixed_image_median = np.median(fixed_image_T)
            moving_image_T = get_image_T(moving_image_path)
            moving_image_median = np.median(moving_image_T)

            resized_fixed_image_xyz = crop(fixed_image_T,
                        fixed_image_median)
            resized_moving_image_xyz = crop(moving_image_T,
                        moving_image_median)

            hdf5_m_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_moving_image_xyz)
            hdf5_f_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_fixed_image_xyz)

        hdf5_m_file.close()
        hdf5_f_file.close()


def preprocess_euler_elastix():

    """Preprocess a selected pairs of raw fixed and moving images from each
    dataset (e.g. 'train/2022-07-26-01', 'test/2022-04-14-04'); then assemble
    them into .h5 files (e.g. 'train/2022-07-26-01/fixed_images.h5',
    'test/2022-04-14-04/moving_images.h5').

    Specifically, preprocessing consists of the following steps:
        - filter out the background pixels;
        - cropped to the target size (208, 96, 56);
        - transform the image with eulerElastix
    """

    elastix_failures = ['2022-01-17-01/908to1018', '2022-07-26-01/107to815']
    source_path = '/data1/jungsoo/data/2023-01-16-reg-data/h5_resize'
    save_directory = \
        '/home/alicia/data_personal/regnet_dataset/2023-01-16_raw_crop-v1'

    for ds_type, ds_names in DATASETS_SPLIT_DICT.items():

        for ds_name in tqdm(ds_names):
            src_file = f'{source_path}/{ds_type}/{ds_name}'
            with h5py.File(f'{src_file}/fixed_images.h5', 'r') as f:
                problems = list(f.keys())

            save_path = f'{save_directory}/{ds_type}/{ds_name}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
            hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

            for problem in problems:
                t_moving, t_fixed = problem.split('to')
                euler_home = \
                    f"/home/alicia/data_personal/registered/euler_transformed_{ds_name}"
                euler_path = f"{euler_home}/{t_moving}to{t_fixed}/result.nrrd"

                if f'{ds_name}/{problem}' not in elastix_failures:
                    if euler_path:
                        t_fixed_4 = t_fixed.zfill(4)
                        ds_path = locate_directory(ds_name)
                        fixed_image_path = glob.glob(
                            f'{ds_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
                        )[0]
                        fixed_image_T = get_image_T(fixed_image_path)
                        fixed_image_median = np.median(fixed_image_T)
                        moving_image_T = get_image_T(euler_path)
                        moving_image_median = np.median(moving_image_T)

                        resized_fixed_image_xyz = crop(fixed_image_T,
                                    fixed_image_median)
                        resized_moving_image_xyz = crop(moving_image_T,
                                    moving_image_median)

                        hdf5_m_file.create_dataset(f'{t_moving}to{t_fixed}',
                                data = resized_moving_image_xyz)
                        hdf5_f_file.create_dataset(f'{t_moving}to{t_fixed}',
                                data = resized_fixed_image_xyz)

            hdf5_m_file.close()
            hdf5_f_file.close()


def preprocess_euler_gpu(downsample_factor,
               resolution_factor,
               x_translation_range,
               y_translation_range,
               z_translation_range,
               theta_rotation_range,
               batch_size,
               device_name):

    with open('jungsoo_registration_problems.json', 'r') as f:
        registration_problem_dict = json.load(f)

    x_dim = 208
    y_dim = 96
    z_dim = 56

    memory_dict_xy = initialize(
                np.zeros((x_dim, y_dim)).astype(np.float32),
                np.zeros((x_dim, y_dim)).astype(np.float32),
                x_translation_range,
                y_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )
    _memory_dict_xy = initialize(
                np.zeros((x_dim, y_dim, z_dim)).astype(np.float32),
                np.zeros((x_dim, y_dim, z_dim)).astype(np.float32),
                x_translation_range,
                y_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )
    memory_dict_xz = initialize(
                np.zeros((x_dim, z_dim)).astype(np.float32),
                np.zeros((x_dim, z_dim)).astype(np.float32),
                x_translation_range,
                y_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )
    _memory_dict_xz = initialize(
                np.zeros((x_dim, z_dim, y_dim)).astype(np.float32),
                np.zeros((x_dim, z_dim, y_dim)).astype(np.float32),
                x_translation_range,
                y_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )
    memory_dict_yz = initialize(
                np.zeros((y_dim, z_dim)).astype(np.float32),
                np.zeros((y_dim, z_dim)).astype(np.float32),
                x_translation_range,
                y_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )
    _memory_dict_yz = initialize(
                np.zeros((y_dim, z_dim, x_dim)).astype(np.float32),
                np.zeros((y_dim, z_dim, x_dim)).astype(np.float32),
                x_translation_range,
                y_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )

    outcomes = dict()

    #for dataset_type_n_name, problems in registration_problem_dict.items():
    for dataset_type_n_name, problems in \
            {'train/2022-01-09-01': ['102to675','104to288']}.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')

        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            outcomes[f"{dataset_name}/{problem}"] = dict()

            t_moving, t_fixed = problem.split('to')
            t_moving_4 = t_moving.zfill(4)
            t_fixed_4 = t_fixed.zfill(4)
            fixed_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
            )[0]
            moving_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
            )[0]

            fixed_image_T = get_image_T(fixed_image_path)
            fixed_image_median = np.median(fixed_image_T)
            moving_image_T = get_image_T(moving_image_path)
            moving_image_median = np.median(moving_image_T)

            resized_fixed_image_xyz = filter_and_crop(fixed_image_T,
                        fixed_image_median)
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
                        moving_image_median)

            # project onto the x-y plane along the z axis with the maximum value
            downsampled_resized_fixed_image_xy = \
                    max_intensity_projection_and_downsample(
                            resized_fixed_image_xyz,
                            downsample_factor,
                            projection_axis=2).astype(np.float32)
            downsampled_resized_moving_image_xy = \
                    max_intensity_projection_and_downsample(
                            resized_moving_image_xyz,
                            downsample_factor,
                            projection_axis=2).astype(np.float32)

            # update the memory dictionary for grid search on x-y image
            memory_dict_xy["fixed_images_repeated"][:] = torch.tensor(
                        downsampled_resized_fixed_image_xy,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            memory_dict_xy["moving_images_repeated"][:] = torch.tensor(
                        downsampled_resized_moving_image_xy,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            # search optimal parameters with projected image on the x-y plane
            best_score_xy, best_transformation_xy = grid_search(memory_dict_xy)
            outcomes[f"{dataset_name}/{problem}"]["x-y_score_best"] = best_score_xy.item()

            print(f"x-y score (best): {best_score_xy}")
            print(f"best_transformation_xy: {best_transformation_xy}")
            # transform the 3d image with the searched parameters
            transformed_moving_image_xyz_max2 = transform_image_3d(
                        resized_moving_image_xyz,
                        _memory_dict_xy,
                        best_transformation_xy,
                        device_name)
            gncc = calculate_gncc(
                    resized_fixed_image_xyz.max(0),
                    transformed_moving_image_xyz_max2.max(0)
            )
            outcomes[f"{dataset_name}/{problem}"]["y-z_score_max2"] = gncc.item()
            print(f"y-z score: {gncc}")
            gncc = calculate_gncc(
                    resized_fixed_image_xyz.max(1),
                    transformed_moving_image_xyz_max2.max(1)
            )
            outcomes[f"{dataset_name}/{problem}"]["x-z_score_max2"] = gncc.item()
            print(f"x-z score: {gncc}")

            best_score_xyz_max2 = calculate_gncc(
                    resized_fixed_image_xyz,
                    transformed_moving_image_xyz_max2
            )
            outcomes[f"{dataset_name}/{problem}"]["full_image_score_max2"] = \
                    gncc.item()
            print(f"full image score: {best_score_xyz_max2}")

            # project onto the x-z plane along the y axis with the maximum value
            downsampled_resized_fixed_image_xz = \
                        max_intensity_projection_and_downsample(
                                resized_fixed_image_xyz,
                                downsample_factor,
                                projection_axis=1).astype(np.float32)

            downsampled_resized_moving_image_xz = \
                        max_intensity_projection_and_downsample(
                                transformed_moving_image_xyz_max2,
                                downsample_factor,
                                projection_axis=1).astype(np.float32)

            # update the memory dictionary for grid search on x-z image
            memory_dict_xz["fixed_images_repeated"][:] = torch.tensor(
                        downsampled_resized_fixed_image_xz,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            memory_dict_xz["moving_images_repeated"][:] = torch.tensor(
                        downsampled_resized_moving_image_xz,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            memory_dict_xz['output_tensor'][:] = torch.zeros_like(
                        memory_dict_xz["moving_images_repeated"],
                        device=device_name,
                        dtype=torch.float32)
            print(f'\n')
            # search optimal parameters with projected image on the x-y plane
            best_score_xz, best_transformation_xz = grid_search(memory_dict_xz)
            outcomes[f"{dataset_name}/{problem}"]["x-z_score_best"] = best_score_xz.item()
            print(f"x-z score (best): {best_score_xz}")
            print(f"best_transformation_xz: {best_transformation_xz}")
            # transform the 3d image with the searched parameters
            transformed_moving_image_xyz_max1 = transform_image_3d(
                        transformed_moving_image_xyz_max2,
                        _memory_dict,
                        best_transformation_xz,
                        device_name)
            gncc = calculate_gncc(
                    resized_fixed_image_xyz.max(0),
                    transformed_moving_image_xyz_max1.max(0)
            )
            outcomes[f"{dataset_name}/{problem}"]["y-z_score_max1"] = gncc.item()
            print(f"y-z score: {gncc}")
            gncc = calculate_gncc(
                    resized_fixed_image_xyz.max(2),
                    transformed_moving_image_xyz_max1.max(2)
            )
            outcomes[f"{dataset_name}/{problem}"]["x-y_score_max1"] = gncc.item()
            print(f"x-y score: {gncc}")

            best_score_xyz_max1 = calculate_gncc(resized_fixed_image_xyz,
                                            transformed_moving_image_xyz_max1)
            outcomes[f"{dataset_name}/{problem}"]["full_image_score_max1"] = best_score_xyz_max1.item()
            print(f"full image score: {best_score_xyz_max1}")

            # project onto the y-z plane along the x axis with the maximum value
            downsampled_resized_fixed_image_yz = \
                        max_intensity_projection_and_downsample(
                                resized_fixed_image_xyz,
                                downsample_factor,
                                projection_axis=0).astype(np.float32)
            downsampled_resized_moving_image_yz = \
                        max_intensity_projection_and_downsample(
                                transformed_moving_image_xyz_max1,
                                downsample_factor,
                                projection_axis=0).astype(np.float32)

            memory_dict_yz["fixed_images_repeated"][:] = torch.tensor(
                        downsampled_resized_fixed_image_yz,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            memory_dict_yz["moving_images_repeated"][:] = torch.tensor(
                        downsampled_resized_moving_image_yz,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            memory_dict_yz['output_tensor'][:] = torch.zeros_like(
                        memory_dict_yz["moving_images_repeated"][:],
                        device=device_name,
                        dtype=torch.float32)
            print(f'\n')
            # search optimal parameters with projected image on the y-z plane
            best_score_yz, best_transformation_yz = grid_search(memory_dict_yz)
            outcomes[f"{dataset_name}/{problem}"]["y-z_score_best"] = best_score_yz.item()
            print(f"y-z score (best): {best_score_yz}")
            print(f"best_transformation_yz: {best_transformation_yz}")
            # transform the 3d image with the searched parameters
            transformed_moving_image_xyz_max0 = transform_image_3d(
                        transformed_moving_image_xyz_max1,
                        _memory_dict_yz,
                        best_transformation_yz,
                        device_name)
            gncc = calculate_gncc(
                    resized_fixed_image_xyz.max(1),
                    transformed_moving_image_xyz_max0.max(1)
            )
            outcomes[f"{dataset_name}/{problem}"]["x-z_score_max0"] = gncc.item()
            print(f"x-z score: {gncc}")
            gncc = calculate_gncc(
                    resized_fixed_image_xyz.max(2),
                    transformed_moving_image_xyz_max0.max(2)
            )
            outcomes[f"{dataset_name}/{problem}"]["x-y_score_max0"] = \
                    gncc.item()
            print(f"x-y score: {gncc}")

            best_score_xyz_max0 = calculate_gncc(resized_fixed_image_xyz,
                                            transformed_moving_image_xyz_max0)
            outcomes[f"{dataset_name}/{problem}"]["full_image_score_max0"] = \
                    best_score_xyz_max0.item()
            print(f"full image score: {best_score_xyz_max0}")

            # search for the optimal dz translation
            dz, gncc, final_moving_image_xyz = translate_along_z(
                        z_translation_range,
                        resized_fixed_image_xyz,
                        transformed_moving_image_xyz_max0,
                        moving_image_median)
            final_score = calculate_gncc(resized_fixed_image_xyz,
                        final_moving_image_xyz)
            outcomes[f"{dataset_name}/{problem}"]["final_full_image_score"] = \
                    final_score.item()
            print(f"final_score: {final_score}")

    with open(f"outcomes.json", "w") as f:
        json.dump(outcomes, f, indent=4)


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

        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            t_moving, t_fixed = problem.split('to')
            t_moving_4 = t_moving.zfill(4)
            t_fixed_4 = t_fixed.zfill(4)
            fixed_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
            )[0]
            moving_image_path = glob.glob(
                    f'{dataset_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
            )[0]

            fixed_image_T = get_image_T(fixed_image_path)
            fixed_image_median = np.median(fixed_image_T)
            moving_image_T = get_image_T(moving_image_path)
            moving_image_median = np.median(moving_image_T)

            resized_fixed_image_xyz = filter_and_crop(fixed_image_T,
                        fixed_image_median)
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
                        moving_image_median)

            downsampled_resized_fixed_image_xy = \
                        max_intensity_projection_and_downsample(
                                resized_fixed_image_xyz,
                                downsample_factor).astype(np.float32)
            downsampled_resized_moving_image_xy = \
                        max_intensity_projection_and_downsample(
                                resized_moving_image_xyz,
                                downsample_factor).astype(np.float32)
            memory_dict["fixed_images_repeated"] = torch.tensor(
                        downsampled_resized_fixed_image_xy,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            memory_dict["moving_images_repeated"] = torch.tensor(
                        downsampled_resized_moving_image_xy,
                        device=device_name,
                        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            best_score, best_transformation = grid_search(memory_dict)
            transformed_moving_image_xyz = transform_image_3d(
                        resized_moving_image_xyz,
                        memory_dict_,
                        best_transformation,
                        device_name)
            dz, gncc, final_moving_image_xyz = translate_along_z(
                        z_translation_range,
                        resized_fixed_image_xyz,
                        transformed_moving_image_xyz,
                        moving_image_median)
            hdf5_m_file.create_dataset(
                        f'{t_moving}to{t_fixed}',
                        data=final_moving_image_xyz
            )
            hdf5_f_file.create_dataset(
                        f'{t_moving}to{t_fixed}',
                        data=resized_fixed_image_xyz
            )

        hdf5_m_file.close()
        hdf5_f_file.close()


def crop(image_T, image_median):

    filtered_image_T = filter_image(image_T, image_median)
    filtered_image_CM = get_image_CM(filtered_image_T)

    return get_cropped_image(image_T, filtered_image_CM,
            -1).astype(np.float32)


def filter_and_crop(image_T, image_median):

    filtered_image_T = filter_image(image_T, image_median)
    filtered_image_CM = get_image_CM(filtered_image_T)

    return get_cropped_image(filtered_image_T, filtered_image_CM,
            -1).astype(np.float32)


def filter_image(image, threshold):
    filtered_image = image - threshold
    filtered_image[filtered_image < 0] = 0

    return filtered_image


def get_registration_problems():

    source_path = '/data1/jungsoo/data/2023-01-16-reg-data/h5_resize'
    registration_problems = dict()
    for ds_type, ds_names in DATASETS_SPLIT_DICT.items():
        for ds_name in ds_names:
            print(ds_name)
            src_file = f'{source_path}/{ds_type}/{ds_name}'
            with h5py.File(f'{src_file}/fixed_images.h5', 'r') as f:
                problems = list(f.keys())
            registration_problems[f'{ds_type}/{ds_name}'] = problems

    with open('jungsoo_registration_problems.json', 'w') as f:
        json.dump(registration_problems, f, indent=4)

    return registration_problems


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
    eulergpu_version = 'd1_r1_xy1_v2'
    save_directory = \
        f'/home/alicia/data_personal/regnet_dataset/2023-01-16-eulerGPU-ddf/{eulergpu_version}'
    '''preprocess(downsample_factor, resolution_factor, x_translation_range,
        y_translation_range, z_translation_range, theta_rotation_range,
        batch_size, device_name, save_directory)'''
    #preprocess_raw()
    preprocess_euler_gpu(downsample_factor,
               resolution_factor,
               x_translation_range,
               y_translation_range,
               z_translation_range,
               theta_rotation_range,
               batch_size,
               device_name)


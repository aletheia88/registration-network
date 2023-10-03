''' This script estimates the difficulty of the given registration problems by
comparing the GNCC of the fixed image and the GNCC of the affine transformed
image; a higher GNCC score implies an easier problem and vice versa.'''

from euler_gpu.grid_search import grid_search
from euler_gpu.preprocess import initialize, max_intensity_projection_and_downsample
from euler_gpu.transform import transform_image_3d, translate_along_z
#from euler_search import build_transform_matrix
from tqdm import tqdm
from util import get_image_T, get_cropped_image, get_image_CM
from util import locate_directory, calculate_gncc,DATASETS_SPLIT_DICT
import SimpleITK as sitk
import glob
import h5py
import json
import nibabel as nib
import numpy as np
import os
import random
import subprocess
import time
import torch


ALICIA_PATH = "/home/alicia/data_personal"

def evaluate_euler_gpu(num_problems_per_dataset,
            downsample_factor,
            resolution_factor,
            x_translation_range,
            y_translation_range,
            z_translation_range,
            theta_rotation_range,
            batch_size,
            device_name):

    z_dim = 56
    # initialized memory dictionary for holding updates during grid search
    width = int(208 / downsample_factor)
    height = int(96 / downsample_factor)
    memory_dict = initialize(np.zeros((width, height)).astype(np.float32),
                             np.zeros((width, height)).astype(np.float32),
                             x_translation_range,
                             y_translation_range,
                             theta_rotation_range,
                             batch_size,
                             device_name
                            )
    # initialized memory dictionary for holding the original images
    memory_dict_ = initialize(np.zeros((208, 96)).astype(np.float32),
                             np.zeros((208, 96)).astype(np.float32),
                             torch.zeros(z_dim, device=device_name),
                             torch.zeros(z_dim, device=device_name),
                             torch.zeros(z_dim, device=device_name),
                             z_dim,
                             device_name
                             )
    euler_gpu_evaluation = dict()
    ds_name_problems = sample_datasets(num_problems_per_dataset)
    #ds_name_problems = ['2022-07-15-06_963to1361', '2022-01-09-01_633to791']
    for ds_name_problem in tqdm(ds_name_problems):

        ds_name, problem = ds_name_problem.split('_')
        ds_path = locate_directory(ds_name)

        # prepare input image pairs
        t_moving, t_fixed = problem.split('to')
        t_moving_4 = t_moving.zfill(4)
        t_fixed_4 = t_fixed.zfill(4)

        time_start_prep_inputs = time.time()
        fixed_image_path = glob.glob(
            f'{ds_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
        )[0]
        moving_image_path = glob.glob(
            f'{ds_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
        )[0]
        fixed_image_T = get_image_T(fixed_image_path)
        moving_image_T = get_image_T(moving_image_path)

        # substract the moving image median from both fixed and moving images
        moving_image_median = np.median(moving_image_T)
        moving_image_T = filter_image(moving_image_T, moving_image_median)
        fixed_image_T = filter_image(fixed_image_T, moving_image_median)

        fixed_image_CM = get_image_CM(fixed_image_T)
        moving_image_CM = get_image_CM(moving_image_T)
        resized_fixed_image_xyz = get_cropped_image(
                        fixed_image_T,
                        fixed_image_CM, -1).astype(np.float32)
        resized_moving_image_xyz = get_cropped_image(
                        moving_image_T,
                        moving_image_CM, -1).astype(np.float32)
        time_end_prep_inputs = time.time()
        time_prep_inputs = time_end_prep_inputs - time_start_prep_inputs

        # downsample before initializing `memory_dict`
        time_start_downsample = time.time()
        downsampled_resized_fixed_image_xy = \
                max_intensity_projection_and_downsample(
                    resized_fixed_image_xyz,
                    downsample_factor).astype(np.float32)

        downsampled_resized_moving_image_xy = \
                max_intensity_projection_and_downsample(
                    resized_moving_image_xyz,
                    downsample_factor).astype(np.float32)
        time_end_downsample = time.time()
        time_downsample = time_end_downsample - time_start_downsample

        time_start_update_memory_dict = time.time()
        # modify the images kept in the memory dictionary
        memory_dict["fixed_images_repeated"] = torch.tensor(
                    downsampled_resized_fixed_image_xy,
                    device=device_name,
                    dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        memory_dict["moving_images_repeated"] = torch.tensor(
                    downsampled_resized_moving_image_xy,
                    device=device_name,
                    dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        time_end_update_memory_dict = time.time()
        time_update_memory_dict = time_end_update_memory_dict - time_start_update_memory_dict

        time_start_grid_search = time.time()
        best_score, best_transformation = grid_search(memory_dict)
        time_end_grid_search = time.time()
        time_grid_search = time_end_grid_search - time_start_grid_search

        dx_gpu, dy_gpu, angles_rad = best_transformation

        # transform moving image with the best parameters searched
        transformed_moving_image_xyz = transform_image_3d(
                    resized_moving_image_xyz,
                    memory_dict_,
                    best_transformation,
                    device_name)

        # translate the image along z-axis and get GNCC
        time_start_search_z = time.time()
        dz, gncc, final_moving_image_xyz = translate_along_z(
                    z_translation_range,
                    resized_fixed_image_xyz,
                    transformed_moving_image_xyz,
                    moving_image_median)
        time_end_search_z = time.time()
        time_search_z = time_end_search_z - time_start_search_z

        euler_gpu_evaluation[ds_name_problem] = {
                    'parameters': [dx_gpu.item(),
                                   dy_gpu.item(),
                                   angles_rad.item(),
                                   dz],
                    'euler_gncc': calculate_gncc(
                                    resized_fixed_image_xyz,
                                    final_moving_image_xyz),
                    'raw_gncc': calculate_gncc(
                                    resized_fixed_image_xyz,
                                    resized_moving_image_xyz),
                    'time_prep_inputs': time_prep_inputs,
                    'time_grid_search': time_grid_search,
                    'time_search_z': time_search_z,
                    'time_downsample': time_downsample,
                    'time_update_memory_dict': time_update_memory_dict
                }
        print(ds_name_problem)
        print(euler_gpu_evaluation[ds_name_problem])

    save_path = f'{ALICIA_PATH}/benchmark'
    file_name = f'euler_gpu_evaluation_d{downsample_factor}_r{resolution_factor}.json'
    with open(f'{save_path}/{file_name}', 'w') as f:
        json.dump(euler_gpu_evaluation, f, default=handle_numpy, indent=4)


def filter_image(image, threshold):
    filtered_image = image - threshold
    filtered_image[filtered_image < 0] = 0

    return filtered_image


def sample_datasets(num_problems_per_dataset):

    '''Sample registration problems from the chosen datasets'''
    ds_name_problems = []
    source_path = "/data1/jungsoo/data/2023-01-16-reg-data/h5_resize"
    for ds_type, ds_names in DATASETS_SPLIT_DICT.items():
        for ds_name in ds_names:
            source_file = f'{source_path}/{ds_type}/{ds_name}'
            with h5py.File(f'{source_file}/fixed_images.h5', 'r') as f:
                problems = list(f.keys())
                random.seed(420)
                sampled_problems = random.sample(problems,
                        num_problems_per_dataset)
                ds_name_problems += [f'{ds_name}_{problem}' for problem in
                        sampled_problems]

    return ds_name_problems


def handle_numpy(obj):

    # Custom function to handle numpy data types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def evaluate_ddf_net(test_results_path, test_datasets):
    """
    Compute the GNCC score for datasets registered via the network solution of
    finding a DDF mapping between the fixed and moving images
    """

    ddf_net_gncc_dict = dict()

    ### evaluate ddf net
    for results_path in tqdm(results_paths):
        fixed_image = nib.load(f"{results_path}/fixed_image.nii.gz").get_fdata()
        pred_fixed_image = \
            nib.load(f"{results_path}/pred_fixed_image.nii.gz").get_fdata()
        ddf_net_gncc_dict[results_path.split('/')[-1]] = \
                    calculate_gncc(fixed_image, pred_fixed_image)

    # save results to a json file
    with open(f"{ALICIA_PATH}/ddfnet_eval/ddf_net_gncc.json", "w") as f:
        json.dump(ddf_net_gncc_dict, f, indent=4)


def evaluate_elastix_bspline(test_ds_path, test_datasets):
    """
    Compute the GNCC score for datasets registered via elastix B-spline
    """

    elastix_gncc_dict = dict()

    # path of the test datasets
    for test_ds in tqdm(test_datasets):

        # get all the registration problems
        fixed_images_path = f"{test_ds_path}/{test_ds}/fixed_images.h5"
        bspline_images_path = \
                    f"{test_ds_path}/{test_ds}/moving_images_bspline.h5"
        with h5py.File(fixed_images_path, "r") as f:
            registration_problems = list(f.keys())

        # get fixed & affine transformed moving image in each registration
        # problem
        for problem in registration_problems:

            t_moving, t_fixed = problem.split('to')
            with h5py.File(fixed_images_path, "r") as f:
                fixed_image = f[problem][:]
            with h5py.File(bspline_images_path, "r") as f:
                bspline_image = f[f"{problem}_bspline"][:]

            elastix_gncc_dict[f"{test_ds}/{problem}"] = \
                        calculate_gncc(fixed_image, bspline_image)

    # save results to a json file
    with open(f"{ALICIA_PATH}/ddfnet_eval/bspline_gncc.json", "w") as f:
        json.dump(elastix_gncc_dict, f, indent=4)


def estimate_dataset_complexity(datasets_to_evaluate, ds_root_path):

    evaluation_dict = dict()
    ds_type = ds_root_path.split('/')[-1] # train, valid, or test
    all_ds_paths = glob.glob("/home/alicia/data_prj_kfc/data_processed/*") + \
            glob.glob("/home/alicia/data_prj_neuropal/data_processed/*")
    ds_paths = [
                ds_path for ds_path in all_ds_paths
                if any(ds in ds_path for ds in datasets_to_evaluate)
               ]
    for ds_path in ds_paths:

        # get the registration problems
        ds_date = ds_path.split('/')[-1][:13]
        fixed_images_path = f"{ds_root_path}/{ds_date}/fixed_images.h5"
        moving_images_path = f"{ds_root_path}/{ds_date}/moving_images.h5"
        with h5py.File(fixed_images_path, "r") as f:
            registration_problems = list(f.keys())

        # affine transform the moving images in each registration problem
        for problem in registration_problems:
            t_moving, t_fixed = problem.split('to')
            affine_image_path = affine_transform(ds_path, ds_date,
                        t_moving, t_fixed)
            # if affine image generated correctly
            if affine_image_path:
                affine_image = \
                        sitk.GetArrayFromImage(sitk.ReadImage(affine_image_path))
                with h5py.File(fixed_images_path, "r") as f:
                    fixed_image = f[problem][:]

                print(f"affine_image: {affine_image.shape}")
                print(f"fixed_image: {fixed_image.shape}")

                # adjust image size
                affine_image_adjusted = adjust_img(affine_image,
                            fixed_image.shape)
                if affine_image_adjusted.size > 0:
                    evaluation_dict[f'{ds_type}/{ds_date}/{problem}'] = \
                                calculate_gncc(fixed_image,
                                    affine_image_adjusted)

    # save results to a json file
    with open(f"{ALICIA_PATH}/ddfnet_eval/{ds_type}_ds_complexity.json", "w") as f:
        json.dump(evaluation_dict, f, indent=4)


def affine_transform(ds_path, ds_date, t_moving, t_fixed):

    # directory where to extract the parameters for running transformix
    params_path = \
            f"{ds_path}/Registered/{t_moving}to{t_fixed}/TransformParameters.1.txt"
    # check if the parameter file exists
    if os.path.isfile(params_path):
        # directory where saves the affine transformed images
        affine_home = f"{ALICIA_PATH}/registered/affine_transformed_{ds_date}"
        affine_path = f"{affine_home}/{t_moving}to{t_fixed}"
        # check if the affine path was already created
        if os.path.isfile(f"{affine_path}/result.nrrd"):
            return f"{affine_path}/result.nrrd"
        # otherwise generate affine image from parameters
        else:
            t_moving_4 = str(t_moving).zfill(4)
            image_path = glob.glob(
                            f"{ds_path}/NRRD_filtered/{ds_date}*_t{t_moving_4}_ch2.nrrd"
                        )[0]
            commands = [
                    f"mkdir {affine_home}",
                    f"mkdir {affine_path}",
                    f"transformix -out {affine_path} -tp {params_path} -in {image_path}"
            ]
            for command in commands:
                subprocess.run(command, shell=True)
            print(f"Affine transformed image generated!\nFind it at {affine_path}")
            return f"{affine_path}/result.nrrd"
    # returns False if the parameter file does not exist
    else:
        return False


if __name__ == "__main__":
    ds_root_path = "/data1/jungsoo/data/2023-01-16-reg-data/h5_resize/test"
    train_datasets = [
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
    valid_datasets = [
            "2022-02-16-04",
            "2022-04-05-01",
            "2022-07-20-01",
            "2022-03-22-01",
            "2022-04-12-04",
            "2022-07-26-01"
    ]
    test_datasets = [
            "2022-04-14-04",
            "2022-04-18-04",
            "2022-08-02-01"
    ]
    num_problems_per_dataset = 30
    resolution_factor = 10
    x_translation_range = np.linspace(-0.25, 0.25, int(100/resolution_factor),
            dtype=np.float32)
    y_translation_range = np.linspace(-0.25, 0.25, int(100/resolution_factor),
            dtype=np.float32)
    theta_rotation_range = np.linspace(0, 360, int(360/resolution_factor),
            dtype=np.float32)
    z_translation_range = range(-50, 50)
    batch_size = 200
    device_name = torch.device("cuda:0")
    downsample_factor = 8
    evaluate_euler_gpu(num_problems_per_dataset,
            downsample_factor,
            resolution_factor,
            x_translation_range,
            y_translation_range,
            z_translation_range,
            theta_rotation_range,
            batch_size,
            device_name)

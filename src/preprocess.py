"""Utilities for preprocessing given image pairs"""

from euler_gpu.grid_search import grid_search
from euler_gpu.preprocess import initialize, max_intensity_projection_and_downsample
from euler_gpu.transform import transform_image_3d, translate_along_z
from julia.api import Julia
from tqdm import tqdm
from util import DATASETS_SPLIT_DICT, get_image_T, get_cropped_image, get_image_CM, locate_directory, calculate_gncc
import glob
import h5py
import json
import numpy as np
import os
import subprocess
import torch

jl = Julia(compiled_modules=False)
jl.eval('include("/home/alicia/notebook/register/modify_elastix_parameters.jl")')
from julia import Main

def affine_transform_image(ds_date,
                          t_moving,
                          t_fixed,
                          nrrd_type='filtered',
                          ch_num=2,
                          ds_path=None):


    modify_parameters = jl.eval("modify_parameter_file")

    if ds_path == None:
        neuropal_dir = '/home/alicia/data_prj_neuropal/data_processed'
        non_neuropal_dir = '/home/alicia/data_prj_kfc/data_processed'
        ds_path = locate_directory(ds_date)

    # directory where to extract the parameters for running transformix
    params_path = \
        f"{ds_path}/Registered/{t_moving}to{t_fixed}/TransformParameters.1.txt"

    # check if the parameter file exists
    if os.path.isfile(params_path):

        # directory where saves the affine-transformed images
        base = f"/home/alicia/data_personal/registered/affine_transformed_roi_{ds_date}"
        problem_path = f"{base}/{t_moving}to{t_fixed}"
        t_moving_4 = str(t_moving).zfill(4)

        image_path = glob.glob(
                    f"{ds_path}/NRRD_{nrrd_type}/{ds_date}*_t{t_moving_4}_ch{ch_num}.nrrd"
                )[0]
        roi_image_path = f"{ds_path}/img_roi_watershed/{t_moving}.nrrd"
        roi_params_path = f"{problem_path}/TransformParameters.1.txt"

        julia_dict = Main.eval('Dict("FinalBSplineInterpolationOrder" => 0, "DefaultPixelValue" => 0)')
        modify_parameters(params_path, roi_params_path, julia_dict)

        commands = [
                f"mkdir {base}",
                f"mkdir {problem_path}",
                f"transformix -out {problem_path} -tp {roi_params_path} -in {roi_image_path}"
        ]
        for command in commands:
            subprocess.run(command, shell=True)
        print(f"Affine-transformed image generated!\nFind it at {problem_path}")
        return f"{problem_path}/result.nrrd"
    else:
        return False


def preprocess_raw(save_folder_name):

    """Preprocess (crop only) and generated pairs of fixed and moving images
    """
    save_directory = \
        f'/home/alicia/data_personal/regnet_dataset/{save_folder_name}'

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

            resized_fixed_image_xyz = filter_and_crop(fixed_image_T,
                        fixed_image_median)
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
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
                base = \
                    f"/home/alicia/data_personal/registered/euler_transformed_{ds_name}"
                problem_path = f"{base}/{t_moving}to{t_fixed}/result.nrrd"

                if f'{ds_name}/{problem}' not in elastix_failures:
                    if problem_path:
                        t_fixed_4 = t_fixed.zfill(4)
                        ds_path = locate_directory(ds_name)
                        fixed_image_path = glob.glob(
                            f'{ds_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
                        )[0]
                        fixed_image_T = get_image_T(fixed_image_path)
                        fixed_image_median = np.median(fixed_image_T)
                        moving_image_T = get_image_T(problem_path)
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


def preprocess_euler_gpu_newranges(
               downsample_factor,
               batch_size,
               device_name):

    save_directory = \
        '/home/alicia/data_personal/regnet_dataset/2023-01-16_euler-gpu-v2_filter-crop-v1_size-v2'
    with open('jungsoo_registration_problems.json', 'r') as f:
        registration_problem_dict = json.load(f)

    x_dim = 290
    y_dim = 120
    z_dim = 64

    z_translation_range = range(-60, 60)

    x_translation_range_xy = np.sort(np.concatenate((
                np.linspace(-0.24, 0.24, 49),
                np.linspace(-0.46, -0.25, 8),
                np.linspace(0.25, 0.46, 8),
                np.linspace(0.5, 1, 3),
                np.linspace(-1, -0.5, 3))))
    y_translation_range_xy = np.sort(np.concatenate((
                np.linspace(-0.28, 0.28, 29),
                np.linspace(-0.54, -0.3, 5),
                np.linspace(0.3, 0.54, 5),
                np.linspace(0.6, 1.4, 3),
                np.linspace(-1.4, -0.6, 3))))
    theta_rotation_range_xy = np.sort(np.concatenate((
                np.linspace(0, 19, 20),
                np.linspace(20, 160, 29),
                np.linspace(161, 199, 39),
                np.linspace(200, 340, 29),
                np.linspace(341, 359, 19))))

    y_translation_range_yz = np.linspace(-0.1, 0.1, 11)
    z_translation_range_yz = np.linspace(-1, 1, 51)
    theta_rotation_range_yz = np.concatenate((
                np.linspace(-40, -20, 5),
                np.linspace(-19, 19, 39),
                np.linspace(20, 40, 5)))

    x_translation_range_xz = np.linspace(-0.1, 0.1, 21)
    z_translation_range_xz = np.linspace(-0.1, 0.1, 9)
    theta_rotation_range_xz = np.concatenate((
                np.linspace(-40, -20, 5),
                np.linspace(-19, 19, 39),
                np.linspace(20, 40, 5)))

    memory_dict_xy = initialize(
                np.zeros((x_dim, y_dim)).astype(np.float32),
                np.zeros((x_dim, y_dim)).astype(np.float32),
                x_translation_range_xy,
                y_translation_range_xy,
                theta_rotation_range_xy,
                batch_size,
                device_name
    )
    _memory_dict_xy = initialize(
                np.zeros((x_dim, y_dim)).astype(np.float32),
                np.zeros((x_dim, y_dim)).astype(np.float32),
                np.zeros(z_dim),
                np.zeros(z_dim),
                np.zeros(z_dim),
                z_dim,
                device_name
    )
    memory_dict_xz = initialize(
                np.zeros((x_dim, z_dim)).astype(np.float32),
                np.zeros((x_dim, z_dim)).astype(np.float32),
                x_translation_range_xz,
                z_translation_range_xz,
                theta_rotation_range_xz,
                batch_size,
                device_name
    )
    _memory_dict_xz = initialize(
                np.zeros((x_dim, z_dim)).astype(np.float32),
                np.zeros((x_dim, z_dim)).astype(np.float32),
                np.zeros(y_dim),
                np.zeros(y_dim),
                np.zeros(y_dim),
                y_dim,
                device_name
    )
    memory_dict_yz = initialize(
                np.zeros((y_dim, z_dim)).astype(np.float32),
                np.zeros((y_dim, z_dim)).astype(np.float32),
                y_translation_range_yz,
                z_translation_range_yz,
                theta_rotation_range_yz,
                batch_size,
                device_name
    )
    _memory_dict_yz = initialize(
                np.zeros((y_dim, z_dim)).astype(np.float32),
                np.zeros((y_dim, z_dim)).astype(np.float32),
                np.zeros(x_dim),
                np.zeros(x_dim),
                np.zeros(x_dim),
                x_dim,
                device_name
    )

    outcomes = dict()

    #for dataset_type_n_name, problems in registration_problem_dict.items():
    for dataset_type_n_name, problems in {'train/2022-01-09-01':
            ['102to675']}.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')
        save_path = f'{save_directory}/{dataset_type}/{dataset_name}'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            problem_id = f"{dataset_name}/{problem}"
            outcomes[problem_id] = dict()

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

            # prepare reshaped fixed images for later use

            resized_fixed_image_xzy = np.transpose(resized_fixed_image_xyz,
                        (0, 2, 1))
            resized_fixed_image_yzx = np.transpose(resized_fixed_image_xyz,
                        (1, 2, 0))
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
                        moving_image_median)

            #########################################
            #########################################
            #########################################

            # project onto the x-y plane along the maximum z

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
            outcomes[problem_id]["registered_image_xyz_gncc_xy"] = best_score_xy.item()

            #print(f"x-y score (best): {best_score_xy}")
            #print(f"best_transformation_xy: {best_transformation_xy}")

            # transform the 3d image with the searched parameters

            transformed_moving_image_xyz = transform_image_3d(
                        resized_moving_image_xyz,
                        _memory_dict_xy,
                        best_transformation_xy,
                        device_name
            )

            registered_image_xyz_gncc_yz = calculate_gncc(
                    resized_fixed_image_xyz.max(0),
                    transformed_moving_image_xyz.max(0)
            )

            #print(f"y-z score: {registered_image_xyz_gncc_yz}")
            outcomes[problem_id]["registered_image_xyz_gncc_yz"] = \
                    registered_image_xyz_gncc_yz.item()
            registered_image_xyz_gncc_xz = calculate_gncc(
                    resized_fixed_image_xyz.max(1),
                    transformed_moving_image_xyz.max(1)
            )
            #print(f"x-z score: {registered_image_xyz_gncc_xz}")
            outcomes[problem_id]["registered_image_xyz_gncc_xz"] = \
                    registered_image_xyz_gncc_xz.item()

            registered_image_xyz_gncc_xyz = calculate_gncc(
                    resized_fixed_image_xyz,
                    transformed_moving_image_xyz
            )
            print(f"full image score xyz: {registered_image_xyz_gncc_xyz}")
            outcomes[problem_id]["registered_image_xyz_gncc_xyz"] = \
                    registered_image_xyz_gncc_xyz.item()

            #########################################
            #########################################
            #########################################

            # project onto the x-z plane along the maximum y

            downsampled_resized_fixed_image_xz = \
                        max_intensity_projection_and_downsample(
                                resized_fixed_image_xyz,
                                downsample_factor,
                                projection_axis=1).astype(np.float32)

            downsampled_resized_moving_image_xz = \
                        max_intensity_projection_and_downsample(
                                transformed_moving_image_xyz,
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
                        memory_dict_xz["moving_images_repeated"][:],
                        device=device_name,
                        dtype=torch.float32)

            # search optimal parameters with projected image on the x-y plane

            best_score_xz, best_transformation_xz = grid_search(memory_dict_xz)

            #print(f"x-z score (best): {best_score_xz}")
            #print(f"best_transformation_xz: {best_transformation_xz}")

            outcomes[problem_id]["x-z_score_best"] = best_score_xz.item()

            # transform the 3d image with the searched parameters

            transformed_moving_image_xzy = transform_image_3d(
                        np.transpose(transformed_moving_image_xyz, (0, 2, 1)),
                        _memory_dict_xz,
                        best_transformation_xz,
                        device_name
            )

            transformed_moving_image_xzy_gncc_yz = calculate_gncc(
                        resized_fixed_image_xzy.max(0),
                        transformed_moving_image_xzy.max(0)
            )
            #print(f"y-z score: {transformed_moving_image_xzy_gncc_yz}")
            outcomes[problem_id]["transformed_moving_image_xzy_gncc_yz"] = transformed_moving_image_xzy_gncc_yz.item()

            transformed_moving_image_xzy_gncc_xy = calculate_gncc(
                    resized_fixed_image_xzy.max(1),
                    transformed_moving_image_xzy.max(1)
            )
            #print(f"x-y score: {transformed_moving_image_xzy_gncc_xy}")
            outcomes[problem_id]["transformed_moving_image_xzy_gncc_xy"] = transformed_moving_image_xzy_gncc_xy.item()

            registered_image_xzy_gncc_xzy = calculate_gncc(
                    resized_fixed_image_xzy,
                    transformed_moving_image_xzy)
            print(f"full image score xzy: {registered_image_xzy_gncc_xzy}")
            outcomes[problem_id]["registered_image_xzy_gncc_xzy"] = registered_image_xzy_gncc_xzy.item()

            #########################################
            #########################################
            #########################################

            # project onto the y-z plane along the maximum x

            downsampled_resized_fixed_image_yz = \
                        max_intensity_projection_and_downsample(
                                resized_fixed_image_xyz,
                                downsample_factor,
                                projection_axis=0).astype(np.float32)
            downsampled_resized_moving_image_yz = \
                        max_intensity_projection_and_downsample(
                                np.transpose(transformed_moving_image_xzy, (0, 2, 1)),
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

            # search optimal parameters with projected image on the y-z plane

            best_score_yz, best_transformation_yz = grid_search(memory_dict_yz)
            outcomes[problem_id]["y-z_score_best"] = best_score_yz.item()
            #print(f"y-z score (best): {best_score_yz}")
            #print(f"best_transformation_yz: {best_transformation_yz}")

            # transform the 3d image with the searched parameters

            transformed_moving_image_yzx = transform_image_3d(
                        np.transpose(transformed_moving_image_xzy, (2, 1, 0)),
                        _memory_dict_yz,
                        best_transformation_yz,
                        device_name
            )

            transformed_moving_image_yzx_gncc_xz = calculate_gncc(
                    resized_fixed_image_yzx.max(0),
                    transformed_moving_image_yzx.max(0)
            )
            #print(f"x-z score: {transformed_moving_image_yzx_gncc_xz}")
            outcomes[problem_id]["transformed_moving_image_yzx_gncc_xz"] = transformed_moving_image_yzx_gncc_xz.item()

            transformed_moving_image_yzx_gncc_xy = calculate_gncc(
                    resized_fixed_image_yzx.max(1),
                    transformed_moving_image_yzx.max(1)
            )
            outcomes[problem_id]["transformed_moving_image_yzx_gncc_xy"] = transformed_moving_image_yzx_gncc_xy.item()
            #print(f"x-y score: {transformed_moving_image_yzx_gncc_xy}")

            registered_image_yzx_gncc_yzx = calculate_gncc(
                    resized_fixed_image_yzx,
                    transformed_moving_image_yzx
            )
            print(f"full image score yzx: {registered_image_yzx_gncc_yzx}")
            outcomes[problem_id]["registered_image_yzx_gncc_yzx"] = registered_image_yzx_gncc_yzx.item()

            # search for the optimal dz translation
            dz, gncc, final_moving_image_xyz = translate_along_z(
                        z_translation_range,
                        resized_fixed_image_xyz,
                        np.transpose(transformed_moving_image_yzx, (2, 0, 1)),
                        moving_image_median
            )

            final_score = calculate_gncc(
                        resized_fixed_image_xyz,
                        final_moving_image_xyz)
            outcomes[problem_id]["final_full_image_score"] = final_score.item()
            print(f"final_score: {final_score}")

            # write dataset to .hdf5 file

            hdf5_m_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = final_moving_image_xyz)
            hdf5_f_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_fixed_image_xyz)

        hdf5_m_file.close()
        hdf5_f_file.close()

    with open(f"outcomes_newranges.json", "w") as f:
        json.dump(outcomes, f, indent=4)


def preprocess_euler_gpu(downsample_factor,
               resolution_factor,
               x_translation_range,
               y_translation_range,
               z_translation_range,
               theta_rotation_range,
               batch_size,
               device_name):

    save_directory = \
        '/home/alicia/data_personal/regnet_dataset/2023-01-16_euler-gpu-v2_crop-v1'
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
                np.zeros((x_dim, y_dim)).astype(np.float32),
                np.zeros((x_dim, y_dim)).astype(np.float32),
                np.zeros(z_dim),
                np.zeros(z_dim),
                np.zeros(z_dim),
                z_dim,
                device_name
    )
    memory_dict_xz = initialize(
                np.zeros((x_dim, z_dim)).astype(np.float32),
                np.zeros((x_dim, z_dim)).astype(np.float32),
                x_translation_range,
                z_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )
    _memory_dict_xz = initialize(
                np.zeros((x_dim, z_dim)).astype(np.float32),
                np.zeros((x_dim, z_dim)).astype(np.float32),
                np.zeros(y_dim),
                np.zeros(y_dim),
                np.zeros(y_dim),
                y_dim,
                device_name
    )
    memory_dict_yz = initialize(
                np.zeros((y_dim, z_dim)).astype(np.float32),
                np.zeros((y_dim, z_dim)).astype(np.float32),
                y_translation_range,
                z_translation_range,
                theta_rotation_range,
                batch_size,
                device_name
    )
    _memory_dict_yz = initialize(
                np.zeros((y_dim, z_dim)).astype(np.float32),
                np.zeros((y_dim, z_dim)).astype(np.float32),
                np.zeros(x_dim),
                np.zeros(x_dim),
                np.zeros(x_dim),
                x_dim,
                device_name
    )

    outcomes = dict()

    for dataset_type_n_name, problems in registration_problem_dict.items():
    #for dataset_type_n_name, problems in {'train/2022-01-09-01':
    #        ['102to675','104to288']}.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')
        save_path = f'{save_directory}/{dataset_type}/{dataset_name}'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

        dataset_path = locate_directory(dataset_name)

        for problem in tqdm(problems):

            problem_id = f"{dataset_name}/{problem}"
            outcomes[problem_id] = dict()

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

            # prepare reshaped fixed images for later use

            resized_fixed_image_xzy = np.transpose(resized_fixed_image_xyz,
                        (0, 2, 1))
            resized_fixed_image_yzx = np.transpose(resized_fixed_image_xyz,
                        (1, 2, 0))
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
                        moving_image_median)

            #########################################
            #########################################
            #########################################

            # project onto the x-y plane along the maximum z

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
            outcomes[problem_id]["registered_image_xyz_gncc_xy"] = best_score_xy.item()

            #print(f"x-y score (best): {best_score_xy}")
            #print(f"best_transformation_xy: {best_transformation_xy}")

            # transform the 3d image with the searched parameters

            transformed_moving_image_xyz = transform_image_3d(
                        resized_moving_image_xyz,
                        _memory_dict_xy,
                        best_transformation_xy,
                        device_name
            )

            registered_image_xyz_gncc_yz = calculate_gncc(
                    resized_fixed_image_xyz.max(0),
                    transformed_moving_image_xyz.max(0)
            )

            #print(f"y-z score: {registered_image_xyz_gncc_yz}")
            outcomes[problem_id]["registered_image_xyz_gncc_yz"] = \
                    registered_image_xyz_gncc_yz.item()
            registered_image_xyz_gncc_xz = calculate_gncc(
                    resized_fixed_image_xyz.max(1),
                    transformed_moving_image_xyz.max(1)
            )
            #print(f"x-z score: {registered_image_xyz_gncc_xz}")
            outcomes[problem_id]["registered_image_xyz_gncc_xz"] = \
                    registered_image_xyz_gncc_xz.item()

            registered_image_xyz_gncc_xyz = calculate_gncc(
                    resized_fixed_image_xyz,
                    transformed_moving_image_xyz
            )
            print(f"full image score xyz: {registered_image_xyz_gncc_xyz}")
            outcomes[problem_id]["registered_image_xyz_gncc_xyz"] = \
                    registered_image_xyz_gncc_xyz.item()

            #########################################
            #########################################
            #########################################

            # project onto the x-z plane along the maximum y

            downsampled_resized_fixed_image_xz = \
                        max_intensity_projection_and_downsample(
                                resized_fixed_image_xyz,
                                downsample_factor,
                                projection_axis=1).astype(np.float32)

            downsampled_resized_moving_image_xz = \
                        max_intensity_projection_and_downsample(
                                transformed_moving_image_xyz,
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
                        memory_dict_xz["moving_images_repeated"][:],
                        device=device_name,
                        dtype=torch.float32)

            # search optimal parameters with projected image on the x-y plane

            best_score_xz, best_transformation_xz = grid_search(memory_dict_xz)

            #print(f"x-z score (best): {best_score_xz}")
            #print(f"best_transformation_xz: {best_transformation_xz}")

            outcomes[problem_id]["x-z_score_best"] = best_score_xz.item()

            # transform the 3d image with the searched parameters

            transformed_moving_image_xzy = transform_image_3d(
                        np.transpose(transformed_moving_image_xyz, (0, 2, 1)),
                        _memory_dict_xz,
                        best_transformation_xz,
                        device_name
            )

            transformed_moving_image_xzy_gncc_yz = calculate_gncc(
                        resized_fixed_image_xzy.max(0),
                        transformed_moving_image_xzy.max(0)
            )
            #print(f"y-z score: {transformed_moving_image_xzy_gncc_yz}")
            outcomes[problem_id]["transformed_moving_image_xzy_gncc_yz"] = transformed_moving_image_xzy_gncc_yz.item()

            transformed_moving_image_xzy_gncc_xy = calculate_gncc(
                    resized_fixed_image_xzy.max(1),
                    transformed_moving_image_xzy.max(1)
            )
            #print(f"x-y score: {transformed_moving_image_xzy_gncc_xy}")
            outcomes[problem_id]["transformed_moving_image_xzy_gncc_xy"] = transformed_moving_image_xzy_gncc_xy.item()

            registered_image_xzy_gncc_xzy = calculate_gncc(
                    resized_fixed_image_xzy,
                    transformed_moving_image_xzy)
            print(f"full image score xzy: {registered_image_xzy_gncc_xzy}")
            outcomes[problem_id]["registered_image_xzy_gncc_xzy"] = registered_image_xzy_gncc_xzy.item()

            #########################################
            #########################################
            #########################################

            # project onto the y-z plane along the maximum x

            downsampled_resized_fixed_image_yz = \
                        max_intensity_projection_and_downsample(
                                resized_fixed_image_xyz,
                                downsample_factor,
                                projection_axis=0).astype(np.float32)
            downsampled_resized_moving_image_yz = \
                        max_intensity_projection_and_downsample(
                                np.transpose(transformed_moving_image_xzy, (0, 2, 1)),
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

            # search optimal parameters with projected image on the y-z plane

            best_score_yz, best_transformation_yz = grid_search(memory_dict_yz)
            outcomes[problem_id]["y-z_score_best"] = best_score_yz.item()
            #print(f"y-z score (best): {best_score_yz}")
            #print(f"best_transformation_yz: {best_transformation_yz}")

            # transform the 3d image with the searched parameters

            transformed_moving_image_yzx = transform_image_3d(
                        np.transpose(transformed_moving_image_xzy, (2, 1, 0)),
                        _memory_dict_yz,
                        best_transformation_yz,
                        device_name
            )

            transformed_moving_image_yzx_gncc_xz = calculate_gncc(
                    resized_fixed_image_yzx.max(0),
                    transformed_moving_image_yzx.max(0)
            )
            #print(f"x-z score: {transformed_moving_image_yzx_gncc_xz}")
            outcomes[problem_id]["transformed_moving_image_yzx_gncc_xz"] = transformed_moving_image_yzx_gncc_xz.item()

            transformed_moving_image_yzx_gncc_xy = calculate_gncc(
                    resized_fixed_image_yzx.max(1),
                    transformed_moving_image_yzx.max(1)
            )
            outcomes[problem_id]["transformed_moving_image_yzx_gncc_xy"] = transformed_moving_image_yzx_gncc_xy.item()
            #print(f"x-y score: {transformed_moving_image_yzx_gncc_xy}")

            registered_image_yzx_gncc_yzx = calculate_gncc(
                    resized_fixed_image_yzx,
                    transformed_moving_image_yzx
            )
            print(f"full image score yzx: {registered_image_yzx_gncc_yzx}")
            outcomes[problem_id]["registered_image_yzx_gncc_yzx"] = registered_image_yzx_gncc_yzx.item()

            # search for the optimal dz translation
            dz, gncc, final_moving_image_xyz = translate_along_z(
                        z_translation_range,
                        resized_fixed_image_xyz,
                        np.transpose(transformed_moving_image_yzx, (2, 0, 1)),
                        moving_image_median
            )

            final_score = calculate_gncc(
                        resized_fixed_image_xyz,
                        final_moving_image_xyz)
            outcomes[problem_id]["final_full_image_score"] = final_score.item()
            print(f"final_score: {final_score}")

            # write dataset to .hdf5 file

            hdf5_m_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = final_moving_image_xyz)
            hdf5_f_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_fixed_image_xyz)

        hdf5_m_file.close()
        hdf5_f_file.close()

    with open(f"outcomes0.json", "w") as f:
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

    filtered_image_CM = get_image_CM(image_T)
    filtered_image_T = filter_image(image_T, image_median)

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
    device_name = torch.device("cuda:0")
    downsample_factor = 1
    eulergpu_version = 'd1_r1_xy1_v2'
    save_directory = \
        f'/home/alicia/data_personal/regnet_dataset/2023-01-16-eulerGPU-ddf/{eulergpu_version}'

    #preprocess_raw("raw_crop-filter_size-v1")

    '''preprocess(downsample_factor, resolution_factor, x_translation_range,
        y_translation_range, z_translation_range, theta_rotation_range,
        batch_size, device_name, save_directory)'''
    '''
    preprocess_euler_gpu(downsample_factor,
               resolution_factor,
               x_translation_range,
               y_translation_range,
               z_translation_range,
               theta_rotation_range,
               batch_size,
               device_name)'''
    """preprocess_euler_gpu_newranges(
               downsample_factor,
               batch_size,
               device_name)
    """
    ds_date = "2022-04-14-04"
    t_moving = "1118"
    t_fixed = "1535"
    affine_transform_image(ds_date,
                          t_moving,
                          t_fixed,
                          nrrd_type='filtered',
                          ch_num=2,
                          ds_path=None)

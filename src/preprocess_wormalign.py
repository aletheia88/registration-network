from euler_gpu.grid_search import grid_search
from euler_gpu.preprocess import initialize, max_intensity_projection_and_downsample
from euler_gpu.transform import transform_image_3d, translate_along_z
from evaluate import calculate_gncc
from tqdm import tqdm
from utils import locate_dataset, filter_and_crop, get_image_T, get_image_CM
import glob
import h5py
import json
import numpy as np
import os
import torch


def resize_images(save_directory):

    with open("resources/registration_problems.json", 'r') as f:
        registration_problem_dict = json.load(f)

    for dataset_type_n_name, problems in registration_problem_dict.items():

        dataset_type, dataset_name = dataset_type_n_name.split('/')
        save_path = f"{save_directory}/{dataset_type}/{dataset_name}"

        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        if not os.path.exists(f"{save_directory}/{dataset_type}"):
            os.mkdir(f"{save_directory}/{dataset_type}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

        dataset_path = locate_dataset(dataset_name)

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

            target_dim = (208, 96, 56)
            resized_fixed_image_xyz = filter_and_crop(fixed_image_T,
                        fixed_image_median, target_dim)
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
                        moving_image_median, target_dim)

            hdf5_m_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_moving_image_xyz)
            hdf5_f_file.create_dataset(f'{t_moving}to{t_fixed}',
                    data = resized_fixed_image_xyz)

        hdf5_m_file.close()
        hdf5_f_file.close()


def generate_pregistered_images(
               target_image_shape,
               save_directory,
               downsample_factor,
               batch_size,
               device_name):

    """
    Generate pre-registered images for a given dataset.

    Args:
        target_image_shape (tuple): the target shape of preregistered images
        save_directory (str): the directory where the pre-registered images will be saved
        downsample_factor (int): the factor of downsampling images
        batch_size (int): the batch size for processing images
        device_name (str): the name of the device (e.g.,'cuda:1')
    """
    # dictionary that saves the CM of each problem in the test set
    CM_dict = dict()
    # dictionary that saves the Euler parameters for test problems
    euler_parameters_dict = dict()

    with open("resources/registration_problems.json", 'r') as f:
        registration_problem_dict = json.load(f)

    x_dim, y_dim, z_dim = target_image_shape

    z_translation_range = range(-z_dim, z_dim)

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
    hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
    hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

    for dataset_name, problems in registration_problem_dict.items():

        save_path = f'{save_directory}/{dataset_type}/nonaugmented/{dataset_name}'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        hdf5_m_file = h5py.File(f'{save_path}/moving_images.h5', 'w')
        hdf5_f_file = h5py.File(f'{save_path}/fixed_images.h5', 'w')

        dataset_path = locate_dataset(dataset_name)
        for problem in tqdm(problems):

            problem_id = f"{dataset_name}/{problem}"
            outcomes[problem_id] = dict()
            euler_parameters_dict[problem_id] = dict()
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

            CM_dict[problem_id] =[
                    get_image_CM(moving_image_T),
                    get_image_CM(fixed_image_T)
            ]

            resized_fixed_image_xyz = filter_and_crop(fixed_image_T,
                        fixed_image_median, target_dim)

            # prepare reshaped fixed images for later use

            resized_fixed_image_xzy = np.transpose(resized_fixed_image_xyz,
                        (0, 2, 1))
            resized_fixed_image_yzx = np.transpose(resized_fixed_image_xyz,
                        (1, 2, 0))
            resized_moving_image_xyz = filter_and_crop(moving_image_T,
                        moving_image_median, target_dim)

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

            euler_parameters_dict[problem_id]["xy"] = [
                    score.item() for score in list(best_transformation_xy)
            ]
            print(best_transformation_xy)
            print(euler_parameters_dict[problem_id]["xy"])
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
            euler_parameters_dict[problem_id]["xz"] = [
                    score.item() for score in list(best_transformation_xz)
            ]

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
            euler_parameters_dict[problem_id]["yz"] = [
                    score.item() for score in list(best_transformation_yz)
            ]
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
            euler_parameters_dict[problem_id]["dz"] = dz

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

    write_to_json(outcomes, "eulergpu_outcomes_2022-01-09-01")
    write_to_json(CM_dict, "center_of_mass_2022-01-09-01")
    write_to_json(euler_parameters_dict, "euler_parameters_2022-01-09-01")


def write_to_json(input_, output_file):

    with open(f"resources/{output_file}.json", "w") as f:
        json.dump(input_, f, indent=4)

    print(f"{output_file} written under resources.")


if __name__ == "__main__":

    save_directory = "/home/alicia/data_personal/regnet_dataset/euler-gpu_size-v1"
    downsample_factor = 1
    batch_size = 200
    device_name = torch.device("cuda:1")

    euler_transform_images(
               save_directory,
               downsample_factor,
               batch_size,
               device_name)


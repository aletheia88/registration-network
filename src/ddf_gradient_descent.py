"""Utilties for
    * preprocessing images via affine transformation implemented with Elastix
    * searching for a DDF warping function through gradient descent
    * computing the GNCC (Global Normalization Correlation Coefficient) score
        between a fixed image and a warped moving image attained from DDF &
        Affine-Elastix transformation for each chosen registration pair in a
        dataset
    * stroing the GNCC scores in a JSON file written in the following format:
        { <ds_name>:
               {
                   (t_moving, t_fixed): {'ddf': ddf_gncc, 'elastix': elastix_gncc}
                   (t_moving, t_fixed): {'ddf': ddf_gncc, 'elastix': elastix_gncc}
                   ...
               }
        }
"""

from deepreg.registry import REGISTRY
from tqdm import tqdm
import SimpleITK as sitk
from util import manifest
import deepreg.model.layer as layer
import deepreg.model.layer as layer
import glob
import json
import numpy as np
import os
import random
import re
import subprocess
import tensorflow as tf
import time
import h5py


root_path = "/home/alicia/data_prj_neuropal/data_processed"
alicia_path = "/home/alicia/data_personal"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
# dataset(s) to exclude due to issues with Elastix transformation
excluded_datasets = ["2022-06-14-07"]


def register_selected(ds_home_dir):

    # register datasets in `ds_home_dir`
    ds_paths = manifest['test']
    ddf_gncc = dict()

    for ds_path in ds_paths:
        # get dataset date from dataset path
        match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}', ds_path)
        if match:
            ds_date = match.group()

        fixed_images_path = f"{ds_home_dir}/test/{ds_date}/fixed_images.h5"
        moving_images_path = f"{ds_home_dir}/test/{ds_date}/moving_images.h5"

        with h5py.File(fixed_images_path) as f:
            problems = list(f.keys())

        # solve each registration problem with ddf
        for problem in tqdm(problems):
            t_moving, t_fixed = problem.split('to')

            # get the fixed image
            with h5py.File(fixed_images_path, "r") as f:
                fixed_image = f[problem][:]
            # get the affine-transformed moving image
            with h5py.File(moving_images_path, "r") as f:
                moving_image = f[problem][:]

            fixed_image_tf = tf.cast(tf.expand_dims(fixed_image, axis=0),
                        dtype=tf.float32)
            moving_image_tf = tf.cast(tf.expand_dims(moving_image, axis=0),
                        dtype=tf.float32)
            """
            affine_path = affine_transform_image(ds_path,
                                                 ds_date,
                                                 t_moving,
                                                 t_fixed)
            """
            warped_moving_image, register_time, _ = register_warp(
                        fixed_image_tf,
                        moving_image_tf)
            img_m_tfm = warped_moving_image.numpy()[0]
            ddf_gncc[f'{ds_date}/{problem}'] = calculate_gncc(fixed_image,
                                                img_m_tfm)
            print(f"{ds_date}/{problem}: ", ddf_gncc[f'{ds_date}/{problem}'])

    with open(f"{alicia_path}/ddfnet_eval/ddf_gncc.json", "w") as f:
        json.dump(ddf_gncc, f, indent=4)


def register_all():

    dates_pattern = "202[0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
    ds_paths = glob.glob(
                f"{root_path}/2022-06-22-animal1-all_output/{dates_pattern}")+\
               glob.glob(f"{root_path}/{dates_pattern}*_output")
    filtered_ds_paths = [ds_path for ds_path in ds_paths if not
            any(ds in ds_path for ds in excluded_datasets)]

    # filter out datasets that elastix failed on generating transformations
    nonempty_elastix_ds_paths = filter_empty_datasets(filtered_ds_paths)
    ds_pairs_dict = sample_register_pairs(nonempty_elastix_ds_paths)
    gncc_per_pair_per_ds = dict()

    for ds_path, register_pairs in ds_pairs_dict.items():

        image_path = f"{ds_path}/NRRD_filtered"
        if os.path.exists(image_path):
            date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}-\d{2}')
            pattern_match = date_pattern.search(image_path)
            if pattern_match:
                ds_date = pattern_match.group()

            moving_images, fixed_images = pair_image(image_path,
                                                     ds_date,
                                                     register_pairs)
            ds_name = "/".join(ds_path.split('/')[5:])
            gncc_per_pair_per_ds[ds_name] = dict()
            num_pairs = len(register_pairs)
            print(f"{num_pairs} register pairs for {ds_name}")
            print(f"pairs to be registered: {register_pairs}")

            for i in tqdm(range(num_pairs)):

                print(f"Current pair to register: {register_pairs[i]}")
                t_moving, t_fixed = register_pairs[i].split()
                fixed_image = tf.cast(tf.expand_dims(fixed_images[i], axis=0),
                                        dtype=tf.float32)
                affine_path = affine_transform_image(ds_path,
                                                     ds_date,
                                                     t_moving,
                                                     t_fixed)
                if affine_path:
                    img_a = sitk.GetArrayFromImage(sitk.ReadImage(affine_path))
                    affine_moving_image = tf.cast(tf.expand_dims(img_a, axis=0), dtype=tf.float32)
                    warped_moving_image, register_time, _ = register_warp(
                                                fixed_image, affine_moving_image)
                    img_m_tfm = warped_moving_image.numpy()[0]
                    ddf_gncc = calculate_gncc(fixed_images[i], img_m_tfm)

                    elastix_root = \
                            f"{ds_path}/Registered/{t_moving}to{t_fixed}"
                    elastix_paths = glob.glob(f"{elastix_root}/result.2.R*.nrrd")
                    elastix_gnccs = [
                                        calculate_gncc(
                                            fixed_images[i],
                                            sitk.GetArrayFromImage(sitk.ReadImage(elastix_path))
                                        )
                                        for elastix_path in elastix_paths
                                    ]
                    if not elastix_gnccs:
                        elastix_gncc = -1
                    else:
                        elastix_gncc = max(elastix_gnccs)
                    print(f"elastix_gncc: {elastix_gncc}")
                    print(f"ddf_gncc: {ddf_gncc}")
                    print(f"register time: {register_time}")
                    gncc_per_pair_per_ds[ds_name][register_pairs[i]] = {
                                                        "ddf_gncc": ddf_gncc,
                                                        "elastix_gncc": elastix_gncc,
                                                        "register_time": register_time
                                                        }
                else:
                    print(f"Not Registered: {ds_path}/Registered/{t_moving}to{t_fixed}")

    with open(f"{alicia_path}/registered/ddf_vs_elastix_all_ds.json", "w") as f:
        json.dump(gncc_per_pair_per_ds, f, indent=4)
    print("gncc results written to .json!")


def filter_empty_datasets(ds_paths):

    empty_dates = []

    # find datasets that are empty
    current_dir = "/storage/fs/data1/prj_kfc/data_processed"
    subdirs = [d for d in os.listdir(current_dir) if
                os.path.isdir(os.path.join(current_dir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(current_dir, subdir)
        # if the subdirectory is empty
        if not os.listdir(subdir_path):
            date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
            pattern_match = date_pattern.search(subdir_path)
            if pattern_match:
                empty_dates.append(pattern_match.group())

    # filter the empty datasets out from the list of paths
    return [ds_path for ds_path in ds_paths if not
            any(date in ds_path for date in empty_dates)]


def sample_register_pairs(ds_paths, sample_ratio=0.005):

    ds_pairs_dict = dict()
    # read in pairs of time points at which to register images
    for ds_path in ds_paths:
        register_problem_path = f"{ds_path}/registration_problems.txt"
        if os.path.exists(register_problem_path):
            with open(register_problem_path, "r") as f:
                file_contents = f.read()
                pairs = file_contents.split('\n')[:-1]
                register_pairs = random.sample(pairs, int(len(pairs) * sample_ratio))
                ds_pairs_dict[ds_path] = register_pairs

    return ds_pairs_dict


def pair_image(image_path, ds_date, register_pairs):

    moving_images = []
    fixed_images = []
    for pair in register_pairs:
        t_moving, t_fixed = pair.split(' ')
        t_moving = str(t_moving).zfill(4)
        t_fixed = str(t_fixed).zfill(4)
        moving_image_path = glob.glob(
                            f"{image_path}/{ds_date}*_t{t_moving}_ch2.nrrd"
                        )
        fixed_image_path = glob.glob(
                            f"{image_path}/{ds_date}*_t{t_fixed}_ch2.nrrd"
                        )
        moving_image = np.squeeze(
                        sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path)),
                        axis=0)
        fixed_image = np.squeeze(
                        sitk.GetArrayFromImage(sitk.ReadImage(fixed_image_path)),
                        axis=0)
        moving_images.append(moving_image)
        fixed_images.append(fixed_image)

    return moving_images, fixed_images


def affine_transform_image(ds_path,
                           ds_date,
                           t_moving,
                           t_fixed,
                           nrrd_type='filtered',
                           ch_num=2):

    # directory where to extract the parameters for running transformix
    params_path = \
            f"{ds_path}/Registered/{t_moving}to{t_fixed}/TransformParameters.1.txt"
    # check if the parameter file exists
    if os.path.isfile(params_path):
        # directory where saves the affine transformed images
        affine_home = f"{alicia_path}/registered/affine_transformed_{ds_date}"
        affine_path = f"{affine_home}/{t_moving}to{t_fixed}"
        t_moving_4 = str(t_moving).zfill(4)

        image_path = glob.glob(
                    f"{ds_path}/NRRD_{nrrd_type}/{ds_date}*_t{t_moving_4}_ch{ch_num}.nrrd"
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
    else:
        return False


def register_warp(fixed_image, moving_image):

    # registration parameters
    config_loss_img_1 = {"name": "lncc", "kernel_size": 16}
    config_loss_img_2 = {"name": "gncc"}
    config_loss_deform_1 = {"name": "bending"}
    weight_loss_deform_1 = 1
    learning_rate = 0.1
    total_iter = int(300)

    # optimisation
    @tf.function
    def train_step(warper, weights, optimizer, mov, fix) -> tuple:
        """
        Train step function for backprop using gradient tape
        :param warper: warping function returned from layer.Warping
        :param weights: trainable ddf [1, f_dim1, f_dim2, f_dim3, 3]
        :param optimizer: tf.optimizers
        :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
        :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
        :return:
            a tuple:
                - loss: overall loss to optimise
                - loss_image: image dissimilarity
                - loss_deform: deformation regularisation
        """
        with tf.GradientTape() as tape:
            pred = warper(inputs=[weights, mov])
            loss_img_1 = REGISTRY.build_loss(config=config_loss_img_1)(
                y_true=fix,
                y_pred=pred,
            )
            loss_img_2 = REGISTRY.build_loss(config=config_loss_img_2)(
                y_true=fix,
                y_pred=pred,
            )
            loss_deform_1 = REGISTRY.build_loss(config=config_loss_deform_1)(
                inputs=weights,
            )
            loss =  1 * loss_img_1 + 2 * loss_img_2 + weight_loss_deform_1 * loss_deform_1
        gradients = tape.gradient(loss, [weights])
        optimizer.apply_gradients(zip(gradients, [weights]))

        return loss, loss_img_1, loss_deform_1

    fixed_image_size = fixed_image.shape
    initializer = tf.random_normal_initializer(mean=0, stddev=1e-3)
    warping = layer.Warping(fixed_image_size=fixed_image_size[1:4])
    var_ddf = tf.Variable(initializer(fixed_image_size + [3]), name="ddf", trainable=True)
    optimiser = tf.optimizers.Adam(learning_rate)

    t0 = time.time()
    for step in range(total_iter):
        loss_opt, loss1_opt, loss_deform_opt = train_step(
            warping, var_ddf, optimiser, moving_image, fixed_image
        )
    t1 = time.time()

    warped_moving_image = warping(inputs=[var_ddf, moving_image])

    return warped_moving_image, t1 - t0, var_ddf.numpy()


def calculate_gncc(fixed, moving):

    mu_f = np.mean(fixed)
    mu_m = np.mean(moving)
    a = np.sum((fixed - mu_f) * (moving - mu_m)) 
    b = np.sqrt(np.sum((fixed - mu_f) ** 2) * np.sum((moving - mu_m) ** 2))
    return a / b


if __name__ == "__main__":
    # register_all()
    ds_home_dir = "/data1/jungsoo/data/2023-01-16-reg-data/h5_resize"
    register_selected(ds_home_dir)

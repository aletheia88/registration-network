"""Utilities for searching the optimal parameters (dθ, dx, dy) for Euler
transformation through a brute-force grid search in a specified parameter
space done on CPU (not GPU-accelerated)
"""

from julia.api import Julia
from scipy import ndimage
from tqdm import tqdm
from util import calculate_gncc, locate_directory
from util import get_cropped_image, get_image_T, get_image_CM
import SimpleITK as sitk
import constants
import copy
import cv2
import glob
import h5py
import json
import numpy as np
import os
import random


jl = Julia(compiled_modules=False)
jl.eval('include("/home/alicia/notebook/register/adjust.jl")')

TARGET_DIM = (208, 96, 56)
julia_resize_func_new = jl.eval("adjust_image_cm")
julia_adjust_point = jl.eval("adjust_point")

# a subspace search region found from 188 sampled problems across 18 datasets
X_RANGE = range(0, 13)  #range(0, 200)
Y_RANGE = range(0, 26)       #range(0, 90)
THETA_RANGE = range(0, 354)     #range(0, 355)

'''
# a corrected full-space search
X_RANGE = range(-200, 200)
Y_RANGE = range(-90, 90)
THETA_RANGE = range(0, 360)
'''

def euler_search_all(num_problem_per_ds):

    '''This function solves the Euler transformation by doing a grid search
    over the specified parameter space; the Euler transformation problems are
    selected though registration problem sampling from all datasets; after the
    search completes, the grid is saved for each pair of "dataset/problem".
    '''

    failures = ['2022-01-17-01/908to1018', '2022-07-26-01/107to815']
    for ds_type, ds_names in DS_DICT.items():
        for ds_name in tqdm(ds_names):
            src_file = f'{SOURCE_PATH}/{ds_type}/{ds_name}'
            with h5py.File(f'{src_file}/fixed_images.h5', 'r') as f:
                problems = list(f.keys())
                subset_problems = random.sample(problems, num_problem_per_ds)

            for problem in subset_problems:
                if f'{ds_name}/{problem}' not in failures:
                    t_moving, t_fixed = problem.split('to')
                    t_moving_4 = t_moving.zfill(4)
                    t_fixed_4 = t_fixed.zfill(4)
                    ds_path = locate_directory(ds_name)
                    fixed_image_path = glob.glob(
                            f'{ds_path}/NRRD_filtered/*_t{t_fixed_4}_ch2.nrrd'
                    )[0]
                    moving_image_path = glob.glob(
                            f'{ds_path}/NRRD_filtered/*_t{t_moving_4}_ch2.nrrd'
                    )[0]

                    fixed_image_T = get_image_T(fixed_image_path)
                    moving_image_T = get_image_T(moving_image_path)
                    fixed_image_CM = get_image_CM(fixed_image_T)
                    moving_image_CM = get_image_CM(moving_image_T)
                    resized_fixed_image = get_cropped_image(
                                fixed_image_T,
                                fixed_image_CM, 2)
                    resized_moving_image = get_cropped_image(
                                moving_image_T,
                                moving_image_CM, 2)

                    grid_search(resized_fixed_image, resized_moving_image,
                            f'{SAVE_PATH}/{ds_name}_{problem}.npy')


def euler_search_single(resized_fixed_image,
                        resized_moving_image,
                        problem,
                        save_dir):

    '''This function takes a pair of resized fixed and resized moving images,
    performs a grid search over the specified parameter space, reports the
    parameters that give rise to the highest GNCC, as well as reporting the
    GNCC.
    '''
    grid = grid_search(resized_fixed_image,
                resized_moving_image,
                f'{save_dir}/{problem}.npy')
    dx, dy, dtheta = np.unravel_index(np.nanargmax(grid), grid.shape)

    rotate_translate = build_transform_matrix(
            resized_moving_image.shape,
            dx, dy, dtheta, order='r')
    translate_rotate = build_transform_matrix(
            resized_moving_image.shape,
            dx, dy, dtheta, order='t')

    image_shape = (resized_moving_image.shape[1], resized_moving_image.shape[0])
    transform_image_rt = cv2.warpAffine(resized_moving_image,
                    rotate_translate,
                    image_shape)
    transform_image_tr = cv2.warpAffine(resized_moving_image,
                    translate_rotate,
                    image_shape)
    rt_gncc = calculate_gncc(resized_fixed_image, transform_image_rt)
    tr_gncc = calculate_gncc(resized_fixed_image, transform_image_tr)

    if rt_gncc > tr_gncc:
        return rotate_translate, rt_gncc
    else:
        return translate_rotate, tr_gncc


def extract_high_grid_value_range(results_path,
                                  grid_threshold,
                                  range_threshold=2):

    '''This function extracts the grid regions where the GNCC score surpasses
    the given threshold across all registration problems. The step-by-step
    method is detailed as the following:
        * Create a unified binary array: for each 3D grid, create a binary mask
        with the same shape such that all elements with GNCC > threshold is set
        to 1; otherwise set to 0;
        * Sum all binary arrays: add together all these binary arrays. The
        resulting array will have higher values in positions where multiple
        original arrays had high values;
        * Determine overlap: we can choose to threshold the resulting sum by
        specifying the minimum number of times we expect that GNCC >
        `grid_threshold` at each grid position; this minimum is
        `range_threshold`, which has a default value set to 2, meaning that we
        only include a grid's position in the final search range if at least
        two grids have GNCC > `grid_threshold` at this position.
    '''

    all_files = [f for f in os.listdir(results_path) if
            os.path.isfile(os.path.join(results_path, f)) and 'json' not in f]
    all_grids = []
    for file in tqdm(all_files):
        all_grids.append(np.load(f'{results_path}/{file}'))

    binary_grids = [(grid > grid_threshold).astype(int) for grid in all_grids]
    sum_grids = np.sum(np.stack(binary_grids), axis=0)
    overlap_grids = (sum_grids > range_threshold).astype(int)
    indices = np.argwhere(overlap_grids > 0)
    print(f'indices: {indices}')
    x_values = indices[:, 0]
    y_values = indices[:, 1]
    theta_values = indices[:, 2]
    x_range, y_range, theta_range = (x_values.min(), x_values.max()),\
                                (y_values.min(), y_values.max()),\
                                (theta_values.min(), theta_values.max())
    return x_range, y_range, theta_range


def evaluate_euler_search(results_path):

    '''This function reads the sampled registration problems solved through
    Euler transformation via brute-force parameter space search and compare
    the searched results with the Elastix-Euler solutions in the following
    two aspects
    * GNCC score
    * distance of the worm's head position between the fixed and shifted moving
    image
    '''

    head_position_distances = dict()
    elastix_euler_gncc = dict()
    euler_search_gncc = dict()

    # locate the dataset and registration problem in Elastix solution set
    all_files = [f for f in os.listdir(results_path) if
            os.path.isfile(os.path.join(results_path, f))]
    all_files = [file for file in all_files if 'json' not in file]

    for file in tqdm(all_files):
        ds_name, problem = file.split('.')[0].split('_')
        t_fixed = problem.split('to')[1].zfill(4)
        t_moving = problem.split('to')[0].zfill(4)
        ds_path = locate_directory(ds_name)
        fixed_image_path = glob.glob(
                f'{ds_path}/NRRD_filtered/*_t{t_fixed}_ch2.nrrd'
        )[0]
        moving_image_path = glob.glob(
                f'{ds_path}/NRRD_filtered/*_t{t_moving}_ch2.nrrd'
        )[0]

        fixed_image_T = get_image_T(fixed_image_path)
        moving_image_T = get_image_T(moving_image_path)

        fixed_image_CM = get_image_CM(fixed_image_T)
        moving_image_CM = get_image_CM(moving_image_T)

        fixed_image_cropped = get_cropped_image(fixed_image_T, fixed_image_CM)
        moving_image_cropped = get_cropped_image(moving_image_T,
                    moving_image_CM)

        # transform head positions to the target dimension space
        head_positions = get_head_positions(ds_path)
        fixed_head_x, fixed_head_y = head_positions[t_fixed]
        moving_head_x, moving_head_y = head_positions[t_moving]

        shifted_fixed_head_xy = julia_adjust_point(
                fixed_image_T.shape,
                (fixed_head_x, fixed_head_y),
                fixed_image_CM,
                TARGET_DIM)
        shifted_moving_head_xy = julia_adjust_point(
                moving_image_T.shape,
                (moving_head_x, moving_head_y),
                moving_image_CM,
                TARGET_DIM)

        shifted_fixed_head_arr = np.array([
                shifted_fixed_head_xy[0],
                shifted_fixed_head_xy[1]]
                )
        shifted_moving_head_arr = np.array([
                shifted_moving_head_xy[0],
                shifted_moving_head_xy[1],
                1])

        # get the optimal transformation parameters
        grid = np.load(f'{results_path}/{file}')
        dx, dy, dtheta = np.unravel_index(np.nanargmax(grid), grid.shape)

        rotate_translate = build_transform_matrix(
                moving_image_cropped.shape,
                dx, dy, dtheta, order='r')
        translate_rotate = build_transform_matrix(
                moving_image_cropped.shape,
                dx, dy, dtheta, order='t')

        rt_head_x, rt_head_y = rotate_translate @ shifted_moving_head_arr
        rt_head_arr = np.array([round(rt_head_x), round(rt_head_y)])

        tr_head_x, tr_head_y = translate_rotate @ shifted_moving_head_arr
        tr_head_arr = np.array([round(tr_head_x), round(tr_head_y)])

        rt_head_distance = np.linalg.norm(rt_head_arr - shifted_fixed_head_arr)
        tr_head_distance = np.linalg.norm(tr_head_arr - shifted_fixed_head_arr)
        head_position_distances[f'{ds_name}_{problem}'] = \
                min(rt_head_distance, tr_head_distance)
        print(f"head dist: {min(rt_head_distance, tr_head_distance)}")

        # calculate gncc score of the pair of fixed and moving image
        # transformed via Elastix Euler
        euler_path = \
        f'/home/alicia/data_personal/registered/euler_transformed_{ds_name}/{problem}/result.nrrd'
        elastix_moving_image_T = get_image_T(euler_path)
        elastix_moving_image_CM = get_image_CM(elastix_moving_image_T)
        elastix_moving_image_cropped = get_cropped_image(
                    elastix_moving_image_T,
                    elastix_moving_image_CM
                )

        elastix_soln_gncc = calculate_gncc(
                    fixed_image_cropped,
                    elastix_moving_image_cropped
                )
        elastix_euler_gncc[f'{ds_name}_{problem}'] = elastix_soln_gncc
        search_soln_gncc = np.nanmax(grid)
        euler_search_gncc[f'{ds_name}_{problem}'] = search_soln_gncc
        print(f'elastix_soln_gncc: {elastix_soln_gncc}')
        print(f'euler_search_gncc: {search_soln_gncc}')

    # write results to disk
    with open(f'{results_path}/head_position_distances.json', 'w') as f:
        json.dump(head_position_distances, f, indent=4)
    with open(f'{results_path}/euler_search_gncc.json', 'w') as f:
        json.dump(euler_search_gncc, f, indent=4)
    with open(f'{results_path}/elastix_euler_gncc.json', 'w') as f:
        json.dump(elastix_euler_gncc, f, indent=4)


def get_head_positions(ds_path):

    '''Given the dataset path, read in the head positions of the animal at each
    time point
    '''

    head_positions = dict()
    with open(f'{ds_path}/head_pos.txt', 'r') as f:
        for line in f:
            time, head_x, head_y = line.strip().split()
            head_positions[time.zfill(4)] = (int(head_x), int(head_y))
    return head_positions


def grid_search(fixed_image, moving_image, save_file=None):

    '''This function computes the gnccs for the entire grid. Each grid holds the
    evaluation of the transform from a unique combination of (dx, dy, dtheta).
    To fill the grid, we compute the gncc of (fixed_image, euler_tfm_image)
    for a single grid with the given transformation parameters (dx, dy, dtheta)
    and order of transformation. The grid only keeps the gncc from the order of
    transformation that results in a higher gncc.
    '''

    x_dim = (X_RANGE.stop - X_RANGE.start) // X_RANGE.step
    y_dim = (Y_RANGE.stop- Y_RANGE.start) // Y_RANGE.step
    theta_dim = (THETA_RANGE.stop - THETA_RANGE.start) // THETA_RANGE.step

    grid = np.zeros((x_dim, y_dim, theta_dim))
    for dx in tqdm(X_RANGE):
        if save_file:
            np.save(save_file, grid)
        for dy in Y_RANGE:
            for dtheta in THETA_RANGE:
                rt_gncc, tr_gncc = compute_gncc(dx, dy, dtheta, fixed_image,
                        moving_image)
                if rt_gncc > tr_gncc:
                    grid[dx, dy, dtheta] = rt_gncc
                else:
                    grid[dx, dy, dtheta] = tr_gncc

    max_gncc = np.amax(grid)
    print(f'Grid {save_file} (if any) saved! max gncc = {max_gncc}')
    return grid


def compute_gncc(dx, dy, dtheta, fixed_image, moving_image):

    rotate_translate = build_transform_matrix(
                        moving_image.shape, dx, dy, dtheta, 'r')
    translate_rotate = build_transform_matrix(
                        moving_image.shape, dx, dy, dtheta, 't')

    image_shape = (moving_image.shape[1], moving_image.shape[0])
    transform_image_rt = cv2.warpAffine(moving_image,
                    rotate_translate,
                    image_shape)
    transform_image_tr = cv2.warpAffine(moving_image,
                    translate_rotate,
                    image_shape)
    rt_gncc = calculate_gncc(fixed_image, transform_image_rt)
    tr_gncc = calculate_gncc(fixed_image, transform_image_tr)

    return rt_gncc, tr_gncc


def build_transform_matrix(image_shape, dx, dy, dtheta, order):

    # here the image is already transposed
    center = (image_shape[1] // 2, image_shape[0] // 2)
    rotation_matrix = np.vstack(
                    [cv2.getRotationMatrix2D(center, dtheta, scale=1),
                    [0, 0, 1]])
    translation_matrix = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
    ])
    if order == 'r':
        return np.dot(rotation_matrix, translation_matrix)[:2, :]
    elif order == 't':
        return np.dot(translation_matrix, rotation_matrix)[:2, :]
    else:
        raise CustomError("Order can only be 'rotation first' (r) \
                or 'translation first (t)'")


if __name__ == "__main__":
    #x_range, y_range, theta_range = extract_high_grid_value_range(SAVE_PATH, 0.5, 2)
    #print(f'x_range: {x_range}\n y_range: {y_range}\n theta_range: {theta_range}')
    #evaluate_euler_search(SAVE_PATH)
    num_problem_per_ds = 10
    euler_search_all(num_problem_per_ds)
    #ds_path = '/home/alicia/data_prj_kfc/data_processed/2022-01-09-01_output'
    #head_positions = get_head_positions(ds_path)
    #t_moving = str(208).zfill(4)
    #t_fixed = str(620).zfill(4)
    #print(f"head at t=208: {head_positions[t_moving]}")
    #print(f'head at t=620: {head_positions[t_fixed]}')

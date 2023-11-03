from preprocess import crop, filter_and_crop
from util import locate_directory, get_image_T
import deepreg.model.layer as layer
import glob
import h5py
import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import tensorflow as tf

ckpt = "20231031-120756_raw_crop-v1_size-v2"
base = "/home/alicia/data_personal/regnet_ckpt/logs_predict"
pair_num = 2

ddf_nii_path = f"{base}/{ckpt}/test/pair_{pair_num}/ddf.nii.gz"
ddf_array = nib.load(ddf_nii_path).get_fdata()

"""
moving_image_nii_path = f"{base}/{ckpt}/test/pair_{pair_num}/moving_image.nii.gz"
moving_image = nib.load(moving_image_nii_path).get_fdata()

fixed_image_nii_path = f"{base}/{ckpt}/test/pair_{pair_num}/fixed_image.nii.gz"
fixed_image = nib.load(fixed_image_nii_path).get_fdata()

pred_fixed_image_nii_path = f"{base}/{ckpt}/test/pair_{pair_num}/pred_fixed_image.nii.gz"
pred_fixed_image = nib.load(pred_fixed_image_nii_path).get_fdata()
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset_type_n_name = 'test/2022-04-14-04'
dataset_type, dataset_name = dataset_type_n_name.split('/')
problem = '1023to1229'
t_moving, t_fixed = problem.split('to')

data_path = locate_directory(dataset_name)
fixed_roi_image_path = f"{data_path}/img_roi_watershed/{t_fixed}.nrrd"
moving_roi_image_path = f"{data_path}/img_roi_watershed/{t_moving}.nrrd"

fixed_roi_image_T = get_image_T(fixed_roi_image_path)
fixed_roi_image_median = np.median(fixed_roi_image_T)
moving_roi_image_T = get_image_T(moving_roi_image_path)
moving_roi_image_median = np.median(moving_roi_image_T)

resized_fixed_roi_image = crop(fixed_roi_image_T, fixed_roi_image_median)
resized_moving_roi_image = crop(moving_roi_image_T, moving_roi_image_median)
print(resized_moving_roi_image.shape)
x = tf.expand_dims(resized_fixed_roi_image, axis=0)
print(x.shape)
resized_fixed_roi_image_tf = tf.cast(tf.expand_dims(resized_fixed_roi_image,
    axis=0), dtype=tf.float32)
resized_moving_roi_image_tf = tf.cast(tf.expand_dims(resized_moving_roi_image,
    axis=0), dtype=tf.float32)

print(resized_fixed_roi_image_tf.shape)
print(resized_moving_roi_image_tf.shape)

fixed_image_size = resized_fixed_roi_image_tf.shape
warping = layer.Warping(fixed_image_size=fixed_image_size[1:4])
warped_moving_roi_image = warping(inputs=[ddf_array, resized_moving_roi_image_tf])
print(warped_moving_roi_image)

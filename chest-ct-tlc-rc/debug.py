
import os
import sys
import ast
import json
import tempfile
import traceback
import argparse
from pathlib import Path
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import SimpleITK as sitk
import shutil
from skimage.measure import label, regionprops

from utils import (
    resample, 
    rescale_intensity, 
    hole_fill,
    remove_dots,
    elastix_register_and_transform,
)

# voxelmorph config
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(THIS_DIR,'shapes-dice-vel-3-res-8-16-32-256f.h5')
GPU_ID = None # None #0, None - for cpu
MULTI_CHANNEL = False
SM_SIZE = (128,128,128)
RESCALE_FACTOR = 4

output_folder = 'test/debug'
fixed_file = os.path.join(output_folder,'fixed-image.nii.gz')
fixed_mask_file = os.path.join(output_folder,'fixed-mask.nii.gz')
moving_file = os.path.join(output_folder,'moving-image.nii.gz')
moved_file = os.path.join(output_folder,'moved-image.nii.gz')
moving_mask_file = os.path.join(output_folder,'moving-mask.nii.gz')
sm_fixed_file = os.path.join(output_folder,'sm-fixed.nii.gz')
sm_moving_file = os.path.join(output_folder,'sm-moving.nii.gz')
warp_file = os.path.join(output_folder,'sm-wrap.nii.gz')
sm_moved_file = os.path.join(output_folder,'sm-moved.nii.gz')


# maybe we actually need to crop!!!

fixed_mask_obj = sitk.ReadImage(fixed_mask_file)
fixed_mask_obj = resample(fixed_mask_obj,SM_SIZE,out_val=0)

fixed_obj = sitk.ReadImage(fixed_file)
fixed_resampled_obj = resample(fixed_obj,SM_SIZE,out_val=-1000)
fixed_resampled_obj = rescale_intensity(fixed_resampled_obj,mask_obj=fixed_mask_obj,min_val=-1000,max_val=1000,out_min_val=0.0,out_max_val=1.0)

moving_mask_obj = sitk.ReadImage(moving_mask_file)
moving_mask_obj = resample(moving_mask_obj,SM_SIZE,out_val=0)

moving_obj = sitk.ReadImage(moving_file)
moving_resampled_obj = resample(moving_obj,SM_SIZE,out_val=-1000)
moving_resampled_obj = rescale_intensity(moving_resampled_obj,mask_obj=moving_mask_obj,min_val=-1000,max_val=1000,out_min_val=0.0,out_max_val=1.0)

sitk.WriteImage(fixed_resampled_obj,sm_fixed_file)
sitk.WriteImage(moving_resampled_obj,sm_moving_file)

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(GPU_ID)

# load moving and fixed images
add_feat_axis = not MULTI_CHANNEL
sm_fixed, fixed_affine = vxm.py.utils.load_volfile(
    sm_moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
sm_moving = vxm.py.utils.load_volfile(sm_moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)

inshape = sm_moving.shape[1:-1]
nb_feats = sm_moving.shape[-1]

with tf.device(device):
    # load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(MODEL_FILE,**config).register(sm_moving, sm_fixed)
    # just checking if Transform works with warp...
    sm_moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([sm_moving, warp])

    # save warp
    vxm.py.utils.save_volfile(warp.squeeze(), warp_file, fixed_affine)
    # save moved image
    vxm.py.utils.save_volfile(sm_moved.squeeze(), sm_moved_file, fixed_affine)

    lg_moving = vxm.py.utils.load_volfile(moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
    _, lg_fixed_affine = vxm.py.utils.load_volfile(fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
    lg_inshape = lg_moving.shape[1:-1]
    lg_moved = vxm.networks.Transform(lg_inshape, rescale=RESCALE_FACTOR, nb_feats=nb_feats).predict([lg_moving, warp])
    vxm.py.utils.save_volfile(lg_moved.squeeze(), moved_file, lg_fixed_affine)

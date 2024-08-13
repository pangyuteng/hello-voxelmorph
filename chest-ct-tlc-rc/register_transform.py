#!/usr/bin/env python
"""
ref  https://raw.githubusercontent.com/voxelmorph/voxelmorph/dev/scripts/tf/register.py

"""

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
GPU_ID = None #0, None - for cpu
MULTI_CHANNEL = False
SM_SIZE = (128,128,128)
RESCALE_FACTOR = 4

# 
#
#  some context:
#    + assumption: we are happy performing registration in a downsampled space (SM_SIZE)
#
#    + given the assumption, we leverage inference with synthmorph on cpu
#    thus operations will not be complaining why registration takes too much time via Elastix
#  
#    + with synthmorph, while you register in the d ownsampled space, you are able to 
#    transform the inputs with a larger dimension* - as long as size is a multiplier of the original image shape.
#  
#  # gist of logic:
#  
#    + perform initial affine registration with SimpleElatix
#    + not-implemented/todo: crop the images based on region-of-interest, this is needed if FOV is "vastly" different between moving and fix.
#    + scale up the image - this is to be used later for voxelmorph transform
#    + scale down the image for voxelmorph registration
#    + with the scaled-up image, perform transformation
#    + then scale down images back to the original image size.
#

def register_transform(fixed_nifti_file,moving_list,output_folder,fixed_mask_nifti_file=None):

    moving_item = [item for item in moving_list if item.get('moving_image',None) is True][0]
    moving_nifti_file = moving_item["moving_file"]
    affine_only_moved_nifti_file = moving_item["affine_only_moved_file"]

    lg_out_size = (np.array(SM_SIZE)*RESCALE_FACTOR).astype(int).tolist()
    lg_fixed_file = os.path.join(output_folder,"lg-fixed-image.nii.gz")
    lg_fixed_mask_file = os.path.join(output_folder,"lg-fixed-mask.nii.gz")

    try:
        # initial affine transform:
        elastix_register_and_transform(
            fixed_nifti_file,
            moving_nifti_file,
            moving_list=moving_list,
        )

        fixed_obj = sitk.ReadImage(fixed_nifti_file)
        lg_fixed_resampled_obj = resample(fixed_obj,lg_out_size)
        sitk.WriteImage(lg_fixed_resampled_obj,lg_fixed_file)
        if fixed_mask_nifti_file:
            fixed_mask_obj = sitk.ReadImage(fixed_mask_nifti_file)
            fixed_mask_obj = resample(fixed_mask_obj,lg_out_size)
            sitk.WriteImage(fixed_mask_obj,lg_fixed_mask_file)

        # rescale up 
        for item in moving_list:
            moved_file = item["moved_file"]
            affine_only_moved_file = item["affine_only_moved_file"]
            lg_affine_only_moved_file = item["lg_affine_only_moved_file"]

            moving_obj = sitk.ReadImage(affine_only_moved_file)
            lg_moving_resampled_obj = resample(moving_obj,lg_out_size,out_val=item["out_val"])
            lg_moving_resampled_obj = sitk.Cast(lg_moving_resampled_obj,moving_obj.GetPixelID())
            sitk.WriteImage(lg_moving_resampled_obj,lg_affine_only_moved_file)

    except:
        traceback.print_exc()

    if not all([os.path.exists(item['affine_only_moved_file']) for item in moving_list]):
        raise ValueError('elastix_register_and_transform failed!')

    if not all([os.path.exists(item['lg_affine_only_moved_file']) for item in moving_list]):
        raise ValueError('lg files failed to generate!')

    if os.path.exists(lg_fixed_file) is False:
        raise ValueError('lg_fixed_file failed to generate!')

    # downsize and resasmple to perform registration.

    fixed_obj = sitk.ReadImage(fixed_nifti_file)
    fixed_resampled_obj = resample(fixed_obj,SM_SIZE,out_val=-1000)
    fixed_resampled_obj = rescale_intensity(fixed_resampled_obj)
    
    moving_obj = sitk.ReadImage(affine_only_moved_nifti_file)
    moving_resampled_obj = resample(moving_obj,SM_SIZE,out_val=-1000)
    moving_resampled_obj = rescale_intensity(moving_resampled_obj)

    warp_file = os.path.join(output_folder,'sm-wrap.nii.gz')
    sm_fixed_file = os.path.join(output_folder,'sm-fixed.nii.gz')
    sm_moving_file = os.path.join(output_folder,'sm-moving.nii.gz')
    sm_moved_file = os.path.join(output_folder,'sm-moved.nii.gz')
    
    sitk.WriteImage(fixed_resampled_obj,sm_fixed_file)
    sitk.WriteImage(moving_resampled_obj,sm_moving_file)

    # tensorflow device handling
    device, nb_devices = vxm.tf.utils.setup_device(GPU_ID)

    # load moving and fixed images
    add_feat_axis = not MULTI_CHANNEL
    sm_fixed, fixed_affine = vxm.py.utils.load_volfile(
        sm_fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
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

        for item in moving_list:
            lg_affine_only_moved_file = item["lg_affine_only_moved_file"]
            lg_moved_file = item["lg_moved_file"]
            # transform with `rescale` specified
            lg_moving = vxm.py.utils.load_volfile(lg_affine_only_moved_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
            _, lg_fixed_affine = vxm.py.utils.load_volfile(lg_fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
            lg_inshape = lg_moving.shape[1:-1]
            lg_moved = vxm.networks.Transform(lg_inshape, rescale=RESCALE_FACTOR, nb_feats=nb_feats).predict([lg_moving, warp])
            vxm.py.utils.save_volfile(lg_moved.squeeze(), lg_moved_file, lg_fixed_affine)

    # rescale back
    for item in moving_list:
        moving_file = item["moving_file"]
        moved_file = item["moved_file"]
        lg_moved_file = item["lg_moved_file"]
        is_mask = item["is_mask"]

        moving_obj = sitk.ReadImage(moving_file)
        lg_moved_obj = sitk.ReadImage(lg_moved_file)
        lg_moved_obj = sitk.Cast(lg_moved_obj,moving_obj.GetPixelID())
        moved_obj = resample(lg_moved_obj,moving_obj.GetSize(),out_val=item["out_val"])
        #if is_mask:
        #    moved_obj = hole_fill(moved_obj)
        moved_obj = sitk.Cast(moved_obj, moving_obj.GetPixelID())
        sitk.WriteImage(moved_obj,moved_file)

from skimage.metrics import hausdorff_distance
from scipy.spatial import distance

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def quality_check(qc_mask_fixed_file,qc_mask_moved_file,qc_json_file):

    fixed = sitk.ReadImage(qc_mask_fixed_file)
    moved = sitk.ReadImage(qc_mask_moved_file)
    fixed_mask = sitk.GetArrayFromImage(fixed) > 0
    moved_mask = sitk.GetArrayFromImage(moved) > 0

    registration_assessment_dict = dict(
        hausdorff_distance=hausdorff_distance(fixed_mask,moved_mask),
        dice=dice_coef(fixed_mask,moved_mask),
    )

    with open(qc_json_file,'w') as f:
        f.write(json.dumps(registration_assessment_dict,default=str,sort_keys=True))


HELP_CONTENT = """

there are too many input args to pump in via cli
thuse, we use json as the input args.
sample json content is provided below:

{
    "fixed_file": "path to fixed image.nii.gz",
    "qc_mask_fixed_file": "path to fixed mask.nii.gz", # optional, used for QC.
    "moving_list": [
        {
            "moving_image": true, # *specify the moving image as first item in moving_list, and set `main` to true*
            "moving_file": "path to moving image.nii.gz",
            "moved_basename": "image.nii.gz", # basename to be used when saving files in output_folder
            "out_val": -1000, # default value outside image
            "is_mask": false # if content is a mask/int, then we disable spline interpolation during resizing
        },
        {
            "qc_mask": true,  # optional, used for QC.
            "moving_file": "path to additional moving images/masks/classifications",
            "moved_basename": "segmentations.nii.gz",
            "out_val": 0,
            "is_mask": true
        },
        {
            "moving_file": "path to additional moving images/masks/classifications",
            "moved_basename": "classifications.nii.gz",
            "out_val": 0,
            "is_mask": true
        }
    ],
    "output_folder":"moved"
}


"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='details',
        usage='use "%(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('json_file',default=str,help=HELP_CONTENT)
    args = parser.parse_args()

    with open(args.json_file,'r') as f:
        content = json.loads(f.read())

    # obtain and check json content
    fixed_nifti_file = content['fixed_file']
    moving_list = content['moving_list']
    output_folder = content['output_folder']
    moving_image_set = False
    qc_mask_set = False
    qc_mask_fixed_file = None
    qc_mask_moved_file = None
    mask_moving_file = None

    os.makedirs(output_folder,exist_ok=True)
    if len(os.listdir(output_folder)) > 0:
        raise ValueError("files found in output_folder, please delete items in folder first!")

    # for ease of debugging/visualization, copying fixed set to output_folder.
    shutil.copy(fixed_nifti_file,os.path.join(output_folder,'fixed-image.nii.gz'))

    for n,item in enumerate(moving_list):
        base_name = item["moved_basename"]
        moving_list[n]["affine_only_moved_file"] = os.path.join(output_folder,f"affine-only-moved-{base_name}")
        moving_list[n]["lg_affine_only_moved_file"] = os.path.join(output_folder,f"lg-affine-only-moved-{base_name}")
        moving_list[n]["lg_moved_file"] = os.path.join(output_folder,f"lg-moved-{base_name}")
        moving_list[n]["moved_file"] = os.path.join(output_folder,f"moved-{base_name}")

        if item.get("moving_image",None) is True:
            moving_image_set = True
            shutil.copy(moving_list[n]['moving_file'],os.path.join(output_folder,'moving-image.nii.gz'))

        if item.get("qc_mask",None) is True and content.get('qc_mask_fixed_file',None):
            qc_mask_set = True
            qc_mask_fixed_file = content['qc_mask_fixed_file']
            qc_mask_moved_file = moving_list[n]["moved_file"]
            shutil.copy(moving_list[n]['moving_file'],os.path.join(output_folder,'moving-mask.nii.gz'))

    if moving_image_set is False:
        raise ValueError("`moving_image` needs to be set for one item in moving_list")

    if qc_mask_set:
        register_transform(fixed_nifti_file,moving_list,output_folder,fixed_mask_nifti_file=qc_mask_fixed_file)
    else:
        register_transform(fixed_nifti_file,moving_list,output_folder,)
    
    print("qc...")
    if qc_mask_set:
        tgt_fixed_mask_file = os.path.join(output_folder,'fixed-mask.nii.gz')
        shutil.copy(qc_mask_fixed_file,tgt_fixed_mask_file)
        qc_json_file = os.path.join(output_folder,"qc.json")
        quality_check(tgt_fixed_mask_file,qc_mask_moved_file,qc_json_file)

    print('done')

"""

docker run -it -u $(id -u):$(id -g) -w $PWD pangyuteng/voxelmorph:latest bash

python register_transform.py input.json


"""


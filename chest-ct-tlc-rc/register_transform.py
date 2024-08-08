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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def core(args):
    # tensorflow device handling
    device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

    # load moving and fixed images
    add_feat_axis = not args.multichannel
    moving = vxm.py.utils.load_volfile(args.resampled_moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        args.resampled_fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    inshape = moving.shape[1:-1]
    nb_feats = moving.shape[-1]

    with tf.device(device):
        # load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

        for og_file,moved_file,lg_moving_file,lg_moved_file in args.moving_list:
            # transform again, with `rescale` specified
            lg_moving = vxm.py.utils.load_volfile(lg_moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
            _,lg_fixed_affine = vxm.py.utils.load_volfile(args.lg_fixed, add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
            lg_inshape = lg_moving.shape[1:-1]
            lg_moved = vxm.networks.Transform(lg_inshape, rescale=args.rescale, nb_feats=nb_feats).predict([lg_moving, warp])
            vxm.py.utils.save_volfile(lg_moved.squeeze(), lg_moved_file, lg_fixed_affine)

    # save warp
    if args.warp:
        vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

    # save moved image
    vxm.py.utils.save_volfile(moved.squeeze(), args.sm_moved, fixed_affine)

#
# TODO: if shit works, please refactor this is fugly af.
#
def register_transform(fixed_nifti_file,moving_nifti_file,affine_only_moved_nifti_file,moving_list,output_folder):
    
    os.makedirs(output_folder,exist_ok=True)
    if len(os.listdir(output_folder)) > 0:
       raise ValueError("files found in output_folder, please delete items in folder first!")
    try:
        # initial affine transform:
        elastix_register_and_transform(
            fixed_nifti_file,
            moving_nifti_file,
            moving_list=moving_list,
        )
    except:
        traceback.print_exc()

    if not all([os.path.exists(item['affine_only_moved_file']) for item in moving_list]):
        raise ValueError('elastix_register_and_transform failed!')

    # voxelmorph config
    model_file = os.path.join(THIS_DIR,'shapes-dice-vel-3-res-8-16-32-256f.h5')
    gpu_id = None #0
    multichannel = False
    sm_size = (128,128,128)
    rescale = 4
    warp_file = None
    sm_moved_file = None
    # downsize and resasmple to perform registration.

    # rescale up 
    for item in moving_list:
        raise NotImplementedError()
        moved_file = moving_list[n]["moved_file"]
        affine_only_moved_file = item["affine_only_moved_file"]
        lg_moved_affine_only_file = item["lg_moved_affine_only_file"]

        lg_out_size = (np.array(sm_size)*rescale).astype(int).tolist()
        moving_obj = sitk.ReadImage(moved_file)
        lg_moving_resampled_obj = resample(moving_obj,lg_out_size)
        lg_moving_resampled_obj = sitk.Cast(lg_moving_resampled_obj,moving_obj.GetPixelID())
        sitk.WriteImage(lg_moving_resampled_obj,lg_moving_file)

    fixed_obj = sitk.ReadImage(fixed_nifti_file)
    og_size = fixed_obj.GetSize()
    fixed_resampled_obj = resample(fixed_obj,sm_size)
    fixed_resampled_obj = rescale_intensity(fixed_resampled_obj)
    
    moving_obj = sitk.ReadImage(affine_only_moved_nifti_file)
    moving_resampled_obj = resample(moving_obj,sm_size)
    moving_resampled_obj = rescale_intensity(moving_resampled_obj)

    resampled_fixed_file = os.path.join(output_folder,'tmp_fixed.nii.gz')
    sitk.WriteImage(fixed_resampled_obj,resampled_fixed_file)
    resampled_moving_file = os.path.join(output_folder,'tmp_moving.nii.gz')
    sitk.WriteImage(moving_resampled_obj,resampled_moving_file)


    # tensorflow device handling
    device, nb_devices = vxm.tf.utils.setup_device(gpu_id)

    # load moving and fixed images
    add_feat_axis = not multichannel
    sm_moving = vxm.py.utils.load_volfile(resampled_moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
    sm_fixed, fixed_affine = vxm.py.utils.load_volfile(
        resampled_moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    inshape = sm_moving.shape[1:-1]
    nb_feats = sm_moving.shape[-1]

    with tf.device(device):
        # load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(model_file, **config).register(sm_moving, sm_fixed)
        # just checking if Transform works with warp...
        sm_moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([sm_moving, warp])
        for item in moving_list:
            lg_moved_affine_only_file = item["lg_moved_affine_only_file"]
            lg_moved_file = item["lg_moved_file"]
            # transform with `rescale` specified
            lg_moving = vxm.py.utils.load_volfile(lg_moved_affine_only_file, add_batch_axis=True, add_feat_axis=add_feat_axis)
            _, lg_fixed_affine = vxm.py.utils.load_volfile(fixed_nifti_file, add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
            lg_inshape = lg_moving.shape[1:-1]
            lg_moved = vxm.networks.Transform(lg_inshape, rescale=rescale, nb_feats=nb_feats).predict([lg_moving, warp])
            vxm.py.utils.save_volfile(lg_moved.squeeze(), lg_moved_file, lg_fixed_affine)

    # save warp
    if warp_file:
        vxm.py.utils.save_volfile(warp.squeeze(), warp_file, fixed_affine)

    # save moved image
    if sm_moved_file:
        vxm.py.utils.save_volfile(moved.squeeze(), sm_moved_file, fixed_affine)

    # rescale back
    for item in moving_list:
        raise NotImplementedError()
        moving_file = moving_list[n]["moving_file"]
        moved_file = moving_list[n]["moved_file"]
        affine_only_moved_file = item["affine_only_moved_file"]
        lg_moved_affine_only_file = item["lg_moved_affine_only_file"]
        lg_moved_file = item["lg_moved_file"]
        is_mask = item["is_mask"]

        moving_obj = sitk.ReadImage(moving_file)
        lg_moved_obj = sitk.ReadImage(lg_moved_file)
        lg_moved_obj = sitk.Cast(lg_moved_obj,moving_obj.GetPixelID())
        moved_obj = resample(lg_moved_obj,og_size)
        if is_mask:
            moved_obj = hole_fill(moved_obj)
        moved_obj = sitk.Cast(moved_obj, moving_obj.GetPixelID())
        sitk.WriteImage(moved_obj,moved_file)

    '''
        # resize
        for og_file,moved_file,lg_moving_file,lg_moved_file in args.moving_list:
            moving_obj = sitk.ReadImage(og_file)
            lg_moved_obj = sitk.ReadImage(lg_moved_file)
            lg_moved_obj = sitk.Cast(lg_moved_obj,moving_obj.GetPixelID())
            moved_obj = resample(lg_moved_obj,og_size)
            if moved_file.endswith('lung.nii.gz'):
                moved_obj = hole_fill(moved_obj)
            moved_obj = sitk.Cast(moved_obj, moving_obj.GetPixelID())
            sitk.WriteImage(moved_obj,moved_file)

    '''
    print('here')
    sys.exit(1)

    '''
        # fixed is hrct, moving is ctwb.
        args.moving = cropped_ctwb_img_file
        args.moving_mask = cropped_ctwb_lung_file
        args.fixed = cropped_hrct_img_file
        args.fixed_mask = cropped_hrct_lung_file
        args.moved_mask = moved_cropped_ctwb_lung_file
        args.moving_list = [
            [ cropped_ctwb_img_file, moved_cropped_ctwb_file ],
            [ cropped_ctwb_lung_file, moved_cropped_ctwb_lung_file ],
            [ cropped_suv_img_file, moved_cropped_suv_lung_file ],
        ]

        # # fixed is ctwb, moving is hrct.
        # args.moving = cropped_hrct_img_file
        # args.moving_mask = cropped_hrct_lung_file
        # args.fixed = cropped_ctwb_img_file
        # args.fixed_mask = cropped_ctwb_lung_file
        # args.moved_mask = moved_cropped_hrct_lung_file
        # args.moving_list = [
        #     [ cropped_hrct_img_file, moved_cropped_hrct_file ],
        #     [ cropped_hrct_lung_file, moved_cropped_hrct_lung_file ],
        # ]

        args.warp = os.path.join(voxelmorph_folder,'warp.nii.gz')
        args.model = 'shapes-dice-vel-3-res-8-16-32-256f.h5'
        args.gpu = None #0
        args.multichannel = False
        args.size = (128,128,128)
        args.rescale = 4
        args.done_file = done_file


















        for n,x in enumerate(args.moving_list):
            args.moving_list[n].append(os.path.join(tmpdir,f'lg_moving_{n}.nii.gz'))
            args.moving_list[n].append(os.path.join(tmpdir,f'lg_moved_{n}.nii.gz'))

        out_size = args.size

        moving_obj = sitk.ReadImage(args.moving)
        moving_resampled_obj = resample(moving_obj,out_size)

        moving_mask_obj = sitk.ReadImage(args.moving_mask)
        moving_mask_resampled_obj = resample(moving_mask_obj,out_size)

        moving_resampled_obj = rescale_intensity(moving_resampled_obj)#,mask_obj=moving_mask_resampled_obj)
        args.resampled_moving = os.path.join(tmpdir,'sm_moving.nii.gz')
        print('moving_resampled_obj',moving_resampled_obj.GetSize())
        sitk.WriteImage(moving_resampled_obj,args.resampled_moving)

        fixed_obj = sitk.ReadImage(args.fixed)
        og_size = fixed_obj.GetSize()
        fixed_resampled_obj = resample(fixed_obj,out_size)

        fixed_mask_obj = sitk.ReadImage(args.fixed_mask)
        fixed_mask_resampled_obj = resample(fixed_mask_obj,out_size)

        fixed_resampled_obj = rescale_intensity(fixed_resampled_obj)#,mask_obj=fixed_mask_resampled_obj)

        args.resampled_fixed = os.path.join(tmpdir,'sm_fixed.nii.gz')
        print('fixed_resampled_obj',fixed_resampled_obj.GetSize())
        sitk.WriteImage(fixed_resampled_obj,args.resampled_fixed)
        
        args.sm_moved = os.path.join(tmpdir,'sm_moved.nii.gz')
        args.lg_fixed = os.path.join(tmpdir,'lg_fixed.nii.gz')

        lg_out_size = (np.array(out_size)*args.rescale).astype(int).tolist()

        for og_file,moved_file,lg_moving_file,lg_moved_file in args.moving_list:
            moving_obj = sitk.ReadImage(og_file)
            lg_moving_resampled_obj = resample(moving_obj,lg_out_size)
            lg_moving_resampled_obj = sitk.Cast(lg_moving_resampled_obj,moving_obj.GetPixelID())
            sitk.WriteImage(lg_moving_resampled_obj,lg_moving_file)

        lg_fixed_resampled_obj = resample(fixed_obj,lg_out_size)
        sitk.WriteImage(lg_fixed_resampled_obj,args.lg_fixed)
        del moving_obj,fixed_obj,moving_resampled_obj,fixed_resampled_obj,lg_moving_resampled_obj

        core(args)

        # resize
        for og_file,moved_file,lg_moving_file,lg_moved_file in args.moving_list:
            moving_obj = sitk.ReadImage(og_file)
            lg_moved_obj = sitk.ReadImage(lg_moved_file)
            lg_moved_obj = sitk.Cast(lg_moved_obj,moving_obj.GetPixelID())
            moved_obj = resample(lg_moved_obj,og_size)
            if moved_file.endswith('lung.nii.gz'):
                moved_obj = hole_fill(moved_obj)
            moved_obj = sitk.Cast(moved_obj, moving_obj.GetPixelID())
            sitk.WriteImage(moved_obj,moved_file)

        for x in [args.resampled_moving,args.sm_moved,args.resampled_fixed]:
            target_dir = os.path.dirname(args.warp)
            shutil.copy(x,target_dir)

    '''
from skimage.metrics import hausdorff_distance
from scipy.spatial import distance

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def quality_check(args):
    fixed = sitk.ReadImage(args.fixed_mask)
    moved = sitk.ReadImage(args.moved_mask)
    fixed_mask = sitk.GetArrayFromImage(fixed)
    moved_mask = sitk.GetArrayFromImage(moved)

    registration_assessment_dict = dict(
        hausdorff_distance=hausdorff_distance(fixed_mask,moved_mask),
        dice=dice_coef(fixed_mask,moved_mask),
    )

    with open(args.done_file,'w') as f:
        f.write(json.dumps(registration_assessment_dict,default=str,sort_keys=True))


if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file',default=str)
    # parser.add_argument('fixed_nifti_file',default=str)
    # parser.add_argument('moving_nifti_file',default=str)
    # parser.add_argument('moving_list',default=str,help="[('/fullpath/image.nii.gz','moved-image.nii.gz',-1000,False),('/fullpath/mask.nii.gz','moved-mask.nii.gz',0,True),('/fullpath/qia.nii.gz','moved-qia.nii.gz',0,True)], each tuple contains path,out-pixe-value,is_mask_boolean")
    # parser.add_argument('output_folder',default=str)
    args = parser.parse_args()

    # fixed_nifti_file = args.fixed_nifti_file
    # moving_nifti_file = args.moving_nifti_file
    # moving_list = ast.literal_eval(args.moving_list)
    # output_folder = args.output_folder
    with open(args.json_file,'r') as f:
        content = json.loads(f.read())

    fixed_nifti_file = content['fixed_nifti_file']
    _moving_nifti_file = None
    _affine_only_moved_nifti_file = None
    moving_list = content['moving_list']
    output_folder = content['output_folder']
    for n,item in enumerate(moving_list):
        base_name = item["moved_file"]
        moving_list[n]["affine_only_moved_file"] = os.path.join(output_folder,"affine-only-"+base_name)
        moving_list[n]["lg_moved_affine_only_file"] = os.path.join(output_folder,"lg-affine-only-"+base_name)
        moving_list[n]["lg_moved_file"] = os.path.join(output_folder,"lg-"+base_name)
        moving_list[n]["moved_file"] = os.path.join(output_folder,base_name)
        if item.get("main",None) is True:
            _moving_nifti_file = moving_list[n]["moving_file"]
            _affine_only_moved_nifti_file = moving_list[n]["affine_only_moved_file"]
    if _moving_nifti_file is None:
        raise ValueError("main tag not found in any item moving list")
    register_transform(fixed_nifti_file,_moving_nifti_file,_affine_only_moved_nifti_file,moving_list,output_folder)

    #quality_check(args)
    print('done')

"""

docker run -it -u $(id -u):$(id -g) -w $PWD pangyuteng/voxelmorph:latest bash

python register_transform.py 

dev
bash hola.sh

"""


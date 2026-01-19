#!/usr/bin/env python

import os
import argparse
import tempfile
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import nibabel as nib
from skimage.measure import label, regionprops
from nibabel.processing import resample_to_output, resample_from_to
from voxelmorph.py.utils import jacobian_determinant

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('--jdet', help='output jdet filename')
parser.add_argument('--movingsm', help='output movingsm filename')
parser.add_argument('--fixedsm', help='output fixedsm filename')
parser.add_argument('--moving-mask')
parser.add_argument('--moved-mask')
parser.add_argument('--moving-mask2')
parser.add_argument('--moved-mask2')

parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')

args = parser.parse_args()

#assert(args.gpu is None) # ensure using CPU
# tensorflow device handling
device, nb_devices = vxm.utils.setup_device(args.gpu)

def myload(nifti_file,minval=-1000,maxval=1000,out_minval=0,out_maxval=1,target_sz=128,scale_intensity=True):
    order = 3 if scale_intensity else 0
    target_shape = [target_sz,target_sz,target_sz]
    img_obj = nib.load(nifti_file)
    moving_shape_np = np.array(img_obj.shape).astype(np.float32)
    moving_spacing_np = np.array(img_obj.header.get_zooms()).astype(np.float32)
    target_shape_np = np.array(target_shape).astype(np.float32)
    target_spacing_np = moving_shape_np*moving_spacing_np/target_shape_np
    moving_resize_factor = moving_spacing_np/target_spacing_np
    # interesting `+1` in vox2out_vox https://github.com/nipy/nibabel/issues/1366
    out_img_obj = resample_to_output(img_obj,voxel_sizes=target_spacing_np,cval=minval,order=order)
    # hack to get desired shape
    new_img = nib.Nifti1Image(np.zeros(target_shape), out_img_obj.affine, out_img_obj.header)
    out_img_obj = resample_from_to(img_obj,new_img,cval=minval,order=order)
    out_img = out_img_obj.get_fdata()
    #out_img = out_img_obj.get_fdata()[:target_sz,:target_sz,:target_sz].astype(np.float32)
    if scale_intensity:
        out_img = ( (out_img-minval)/(maxval-minval) ).clip(out_minval,out_maxval)
    out_img = out_img[np.newaxis, ... , np.newaxis]
    return out_img_obj, out_img, out_img_obj.affine

sm_moving_obj, moving, _ = myload(args.moving)
sm_fixed_obj, fixed, fixed_affine = myload(args.fixed)

if args.movingsm:
    nib.save(sm_moving_obj,args.movingsm)
if args.fixedsm:
    nib.save(sm_fixed_obj,args.fixedsm)

inshape = moving.shape[1:-1]
nb_feats = 1

with tf.device(device):
    # load model and predict
    config = dict(inshape=inshape, input_model=None)
    print(moving.shape)
    print(fixed.shape)
    warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
    moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

# save warp
if args.warp:
    myfolder = os.path.dirname(args.warp)
    os.makedirs(myfolder,exist_ok=True)
    print("warp",warp.squeeze().shape)
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

# save jacobian determinant
if args.jdet:
    myfolder = os.path.dirname(args.jdet)
    os.makedirs(myfolder,exist_ok=True)
    jdet = jacobian_determinant(warp.squeeze())
    print("jdet",jdet.squeeze().shape)
    vxm.py.utils.save_volfile(jdet.squeeze(), args.jdet, fixed_affine)

minval,maxval = -1000,1000
moved = (moved.clip(0,1)*(maxval-minval))+minval
moved = moved.astype(np.int32)
if False:
    vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)

rescale_factor = 4 # from 128 to 512.
fixed_obj = nib.load(args.fixed)
_, lg_fixed, lg_fixed_affine = myload(args.fixed,target_sz=512)
if args.moved:
    myfolder = os.path.dirname(args.moved)
    os.makedirs(myfolder,exist_ok=True)
    interp_method = 'linear'

    _, lg_moving, _ = myload(args.moving,target_sz=512,scale_intensity=False)
    lg_inshape = lg_fixed.shape[1:-1]
    with tf.device(device):
        lg_moved = vxm.networks.Transform(lg_inshape,
            rescale=rescale_factor,
            nb_feats=nb_feats,
            interp_method=interp_method).predict([lg_moving, warp])

    lg_moved = lg_moved.astype(np.int32)
    lg_moved_obj = nib.Nifti1Image(lg_moved.squeeze(), lg_fixed_affine)
    #reshape this back
    lg_moved_obj = resample_from_to(lg_moved_obj,fixed_obj)
    nib.save(lg_moved_obj, args.moved)

def cleanup_mask(org_mask):
    new_mask = np.zeros_like(org_mask)
    for idx in np.unique(org_mask):
        if idx == 0:
            continue
        label_image = label(org_mask==idx)
        region_list = regionprops(label_image)
        if len(region_list) > 1: # get largest
            region_list = sorted(region_list,key=lambda x: x.area,reverse=True)
            largest_blob = (label_image == region_list[0].label).astype(np.uint8)
            new_mask[largest_blob==1]=idx
        else:
            new_mask[org_mask==idx]=idx
    return new_mask

if args.moving_mask:
    myfolder = os.path.dirname(args.moving_mask)
    os.makedirs(myfolder,exist_ok=True)
    interp_method = 'nearest'

    _, lg_moving, _ = myload(args.moving_mask,target_sz=512,scale_intensity=False)
    with tf.device(device):
        lg_moved = vxm.networks.Transform(lg_inshape,
            rescale=rescale_factor,
            nb_feats=nb_feats,
            interp_method=interp_method).predict([lg_moving, warp])

    lg_moved = lg_moved.astype(np.int32)
    lg_moved = cleanup_mask(lg_moved.squeeze())
    lg_moved_obj = nib.Nifti1Image(lg_moved, lg_fixed_affine)
    #reshape this back
    lg_moved_obj = resample_from_to(lg_moved_obj,fixed_obj)
    nib.save(lg_moved_obj, args.moved_mask)

if args.moving_mask2:
    myfolder = os.path.dirname(args.moving_mask2)
    os.makedirs(myfolder,exist_ok=True)
    interp_method = 'nearest'

    _, lg_moving, _ = myload(args.moving_mask2,target_sz=512,scale_intensity=False)
    with tf.device(device):
        lg_moved = vxm.networks.Transform(lg_inshape,
            rescale=rescale_factor,
            nb_feats=nb_feats,
            interp_method=interp_method).predict([lg_moving, warp])

    lg_moved = lg_moved.astype(np.int32)
    lg_moved_obj = nib.Nifti1Image(lg_moved.squeeze(), lg_fixed_affine)
    #reshape this back
    lg_moved_obj = resample_from_to(lg_moved_obj,fixed_obj)
    nib.save(lg_moved_obj, args.moved_mask2)

"""

docker run --memory=40g -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid \
    pangyuteng/voxelmorph:0.1.2 bash

python register_full_res.py \

"""
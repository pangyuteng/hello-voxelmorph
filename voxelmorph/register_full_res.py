#!/usr/bin/env python

import os
import argparse
import tempfile
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import nibabel as nib
from nibabel.processing import resample_to_output
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
raise NotImplementedError()
# TODO: pending testing
parser.add_argument('--moving_mask')
parser.add_argument('--moved_mask')

parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')

args = parser.parse_args()

assert(args.gpu is None) # ensure using CPU
# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)


def myload(nifti_file,minval=-1000,maxval=1000,out_minval=0,out_maxval=1,target_sz=128,scale_intensity=True):
    target_shape = [target_sz,target_sz,target_sz]
    img_obj = nib.load(nifti_file)
    moving_shape_np = np.array(img_obj.shape).astype(np.float32)
    moving_spacing_np = np.array(img_obj.header.get_zooms()).astype(np.float32)
    target_shape_np = np.array(target_shape).astype(np.float32)
    target_spacing_np = moving_shape_np*moving_spacing_np/target_shape_np
    moving_resize_factor = moving_spacing_np/target_spacing_np
    # interesting `+1` in vox2out_vox https://github.com/nipy/nibabel/issues/1366
    out_img_obj = resample_to_output(img_obj,voxel_sizes=target_spacing_np,cval=minval)
    out_img = out_img_obj.get_fdata()[:target_sz,:target_sz,:target_sz].astype(np.float32)
    if scale_intensity:
        out_img = ( (out_img-minval)/(maxval-minval) ).clip(out_minval,out_maxval)
    out_img = out_img[np.newaxis, ... , np.newaxis]
    return out_img_obj, out_img, out_img_obj.affine

moving_obj, moving, _ = myload(args.moving)
fixed_obj, fixed, fixed_affine = myload(args.fixed)

if args.movingsm:
    nib.save(moving_obj,args.movingsm)
if args.fixedsm:
    nib.save(fixed_obj,args.fixedsm)

inshape = moving.shape[1:-1]
nb_feats = 1

with tf.device(device):
    # load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
    moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

# save warp
if args.warp:
    print("warp",warp.squeeze().shape)
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

# save jacobian determinant
if args.jdet:
    jdet = jacobian_determinant(warp.squeeze())
    print("jdet",jdet.squeeze().shape)
    vxm.py.utils.save_volfile(jdet.squeeze(), args.jdet, fixed_affine)

minval,maxval = -1000,1000
moved = (moved.clip(0,1)*(maxval-minval))+minval
moved = moved.astype(np.int32)
if False:
    vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)

rescale_factor = 4 # from 128 to 512.
if args.moved:
    interp_method = 'linear'

    _, lg_moving, _ = myload(args.moving,target_sz=512,scale_intensity=False)
    _, lg_fixed, lg_fixed_affine = myload(args.fixed,target_sz=512)
    lg_inshape = lg_fixed.shape[1:-1]

    lg_moved = vxm.networks.Transform(lg_inshape,
        rescale=rescale_factor,
        nb_feats=nb_feats,
        interp_method=interp_method).predict([lg_moving, warp])

    #lg_moved = (lg_moved.clip(0,1)*(maxval-minval))+minval
    lg_moved = lg_moved.astype(np.int32)
    # TODO: you need to reshape this back
    vxm.py.utils.save_volfile(lg_moved.squeeze(), args.moved, lg_fixed_affine)

if args.moving_mask:
    interp_method = 'nearest'

    _, lg_moving, _ = myload(args.moving_mask,target_sz=512,scale_intensity=False)
    _, lg_fixed, lg_fixed_affine = myload(args.fixed,target_sz=512)
    
    lg_moved = vxm.networks.Transform(lg_inshape,
        rescale=rescale_factor,
        nb_feats=nb_feats,
        interp_method=interp_method).predict([lg_moving, warp])

    lg_moved = lg_moved.astype(np.int32)
    vxm.py.utils.save_volfile(lg_moved.squeeze(), args.moved_mask, lg_fixed_affine)


"""

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid \
    pangyuteng/voxelmorph bash

python register_full_res.py \
--moving /radraid/pteng-public/tlc-rv-10123-downsampled/5aeb4c6ca234f5f929f72f194f02ed2e/3b62d55f785ff444070a919a26676e4c/img.nii.gz \
--fixed /radraid/pteng-public/tlc-rv-10123-downsampled/5aeb4c6ca234f5f929f72f194f02ed2e/a816e5f7c3835543cc02322bfee0e06d/img.nii.gz \
--moved /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/eval/post_training/moved_3b62d55f785ff444070a919a26676e4c.nii.gz
--warp /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/eval/post_training/warp.nii.gz

parser.add_argument('--moving_mask')
parser.add_argument('--moved_mask')


fixed 
moving 
moved 
weight /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir/2380.h5
"""
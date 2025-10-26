#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.h5 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import argparse
import tempfile
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import nibabel as nib
from nibabel.processing import resample_to_output

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')

args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

"""
multichannel = False
add_batch_axis = True
add_feat_axis = not multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
"""

def myload(nifti_file):
    # super custom rescale and resize
    minval,maxval = -1000,1000
    TARGET_SZ = 128
    TARGET_SHAPE = [TARGET_SZ,TARGET_SZ,TARGET_SZ]

    img_obj = nib.load(nifti_file)
    moving_shape_np = np.array(img_obj.shape).astype(np.float32)
    moving_spacing_np = np.array(img_obj.header.get_zooms()).astype(np.float32)
    target_shape_np = np.array(TARGET_SHAPE).astype(np.float32)
    target_spacing_np = moving_shape_np*moving_spacing_np/target_shape_np
    moving_resize_factor = moving_spacing_np/target_spacing_np
    # interesting `+1` in vox2out_vox https://github.com/nipy/nibabel/issues/1366
    sm_img_obj = resample_to_output(img_obj,voxel_sizes=target_spacing_np)
    sm_img = sm_img_obj.get_fdata()[:TARGET_SZ,:TARGET_SZ,:TARGET_SZ].astype(np.float32)
    sm_img = ( (sm_img-minval)/(maxval-minval) ).clip(0,1)
    sm_img = sm_img[np.newaxis, ... , np.newaxis]
    return sm_img, sm_img_obj.affine

moving, _ = myload(args.moving)
fixed, fixed_affine = myload(args.fixed)

inshape = moving.shape[1:-1]
nb_feats = 1

with tf.device(device):
    # load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
    moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

"""
    is_mask = False
    if is_mask:
        interp_method = 'nearest'
    else:
        interp_method = 'linear'

    lg_moved = vxm.networks.Transform(lg_inshape,
        rescale=RESCALE_FACTOR,
        nb_feats=nb_feats,
        interp_method=interp_method).predict([lg_moving, warp])

    vxm.py.utils.save_volfile(lg_moved.squeeze(), lg_moved_file, lg_fixed_affine)
"""

# save warp
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

minval,maxval = -1000,1000
print(np.min(moved),np.max(moved),'!!!')
moved = (moved.clip(0,1)*(maxval-minval))+minval
moved = moved.astype(np.int32)
# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)

"""

docker run -it -u $(id -u):$(id -g) -w $PWD \
-v /mnt:/mnt \
pangyuteng/voxelmorph bash



"""
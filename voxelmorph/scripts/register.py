#!/usr/bin/env python

"""

https://raw.githubusercontent.com/voxelmorph/voxelmorph/dev/scripts/tf/register.py

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
import ast
import tempfile
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import SimpleITK as sitk
import shutil

from utils import resample, rescale_intensity

def main(args):
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

        # transform again, with `rescale` specified
        lg_moving = vxm.py.utils.load_volfile(args.lg_moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
        _,lg_fixed_affine = vxm.py.utils.load_volfile(args.lg_fixed, add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
        lg_inshape = lg_moving.shape[1:-1]
        lg_moved = vxm.networks.Transform(lg_inshape, rescale=args.rescale, nb_feats=nb_feats).predict([lg_moving, warp])

    # save warp
    if args.warp:
        vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

    # save moved image
    vxm.py.utils.save_volfile(moved.squeeze(), args.sm_moved, fixed_affine)
    vxm.py.utils.save_volfile(lg_moved.squeeze(), args.lg_moved, lg_fixed_affine)

if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving', required=True, help='moving image (source) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    parser.add_argument('--moved', required=True, help='warped image output filename')
    parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
    parser.add_argument('--warp', help='output warp deformation filename')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    parser.add_argument('--size', default='(128,128,128)',help='shape of resampled moving and fixed images prior feeding to voxelmorph')
    parser.add_argument('--rescale',type=int,default=4,help='attempt to trasnform using an higher resolution moving')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_size = ast.literal_eval(args.size)

        moving_obj = sitk.ReadImage(args.moving)
        og_size = moving_obj.GetSize()
        moving_resampled_obj = resample(moving_obj,out_size)
        moving_resampled_obj = rescale_intensity(moving_resampled_obj)
        args.resampled_moving = os.path.join(tmpdir,'sm_moving.nii.gz')
        sitk.WriteImage(moving_resampled_obj,args.resampled_moving)

        fixed_obj = sitk.ReadImage(args.fixed)
        fixed_resampled_obj = resample(fixed_obj,out_size)
        fixed_resampled_obj = rescale_intensity(fixed_resampled_obj)
        args.resampled_fixed = os.path.join(tmpdir,'sm_fixed.nii.gz')
        sitk.WriteImage(fixed_resampled_obj,args.resampled_fixed)
        
        args.sm_moved = os.path.join(tmpdir,'sm_moved.nii.gz')
        args.lg_moved = os.path.join(tmpdir,'lg_moved.nii.gz')
        args.lg_moving = os.path.join(tmpdir,'lg_moving.nii.gz')
        args.lg_fixed = os.path.join(tmpdir,'lg_fixed.nii.gz')

        lg_out_size = (np.array(out_size)*args.rescale).astype(int).tolist()
        lg_moving_resampled_obj = resample(moving_obj,lg_out_size)
        sitk.WriteImage(lg_moving_resampled_obj,args.lg_moving)
        lg_fixed_resampled_obj = resample(fixed_obj,lg_out_size)
        sitk.WriteImage(lg_fixed_resampled_obj,args.lg_fixed)
        del moving_obj,fixed_obj,moving_resampled_obj,fixed_resampled_obj,lg_moving_resampled_obj

        main(args)

        # resize
        lg_moved_obj = sitk.ReadImage(args.lg_moved)
        moved_obj = resample(lg_moved_obj,og_size)
        moved_obj = sitk.Cast(moved_obj, sitk.sitkInt32)
        sitk.WriteImage(moved_obj,args.moved)

        # TODO del later after dev.
        shutil.copy(args.resampled_moving,os.path.dirname(args.moved))
        shutil.copy(args.resampled_fixed,os.path.dirname(args.moved))
        shutil.copy(args.sm_moved,os.path.dirname(args.moved))
        shutil.copy(args.lg_moved,os.path.dirname(args.moved))
        shutil.copy(args.moving,os.path.dirname(args.moved))
        shutil.copy(args.fixed,os.path.dirname(args.moved))

    print('done')

"""

python register.py \
    --fixed readonly/rv.nii.gz \
    --moving readonly/tlc.nii.gz \
    --moved workdir/moved.nii.gz \
    --model shapes-dice-vel-3-res-8-16-32-256f.h5 \
    --gpu 0

"""
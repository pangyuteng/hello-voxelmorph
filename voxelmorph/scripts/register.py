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
    moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    inshape = moving.shape[1:-1]
    nb_feats = moving.shape[-1]

    with tf.device(device):
        # load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

    # save warp
    if args.warp:
        vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

    # save moved image
    vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)

if __name__ == "__main__":
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--moving', required=True, help='moving image (source) filename')
    parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
    parser.add_argument('--moved', required=True, help='warped image output filename')
    parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
    parser.add_argument('--size', default='(128,128,128)')
    parser.add_argument('--warp', help='output warp deformation filename')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_size = ast.literal_eval(args.size)
        moving_obj = sitk.ReadImage(args.moving)
        moving_resampled_obj = resample(moving_obj,out_size)
        moving_resampled_obj = rescale_intensity(moving_resampled_obj)
        moving_file = os.path.join(tmpdir,'moving.nii.gz')
        sitk.WriteImage(moving_resampled_obj,moving_file)
        shutil.copy(moving_file,os.path.dirname(args.moved))
        fixed_obj = sitk.ReadImage(args.fixed)
        fixed_resampled_obj = resample(fixed_obj,out_size)
        fixed_resampled_obj = rescale_intensity(fixed_resampled_obj)
        fixed_file = os.path.join(tmpdir,'fixed.nii.gz')
        sitk.WriteImage(fixed_resampled_obj,fixed_file)
        shutil.copy(fixed_file,os.path.dirname(args.moved))
        
        args.moving = moving_file
        args.fixed = fixed_file
        moving_obj = sitk.ReadImage(args.moving)
        fixed_obj = sitk.ReadImage(args.fixed)
        print(moving_obj.GetSize())
        print(fixed_obj.GetSize())
        main(args)

    print('done')

"""

python register.py \
    --fixed /radraid/pteng-public/tmp/RESEARCH/10123/10123_001ABRRO/2004-07-13/rv.nii.gz \
    --moving /radraid/pteng-public/tmp/RESEARCH/10123/10123_001ABRRO/2004-07-13/tlc.nii.gz \
    --moved /cvibraid/cvib2/Temp/tmp/moved.nii.gz \
    --model shapes-dice-vel-3-res-8-16-32-256f.h5 \
    --gpu 0

"""
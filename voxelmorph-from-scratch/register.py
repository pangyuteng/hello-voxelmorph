#
# COPY ADAPTED FROM voxelmorph/scripts/register.py 
# 

# Core library imports
import os
import argparse

# Third-party imports
import torch

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

# Local imports
import voxelmorph as vxm   # nopep8

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel

moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

"""
transform = tio.Resize((128,128,128))
rescale = tio.RescaleIntensity(out_min_max=(-1,1),in_min_max=(-1000,1000))

source_nii = tio.ScalarImage(source_path)
source_nii = transform(rescale(source_nii))

target_nii = tio.ScalarImage(source_path)
target_nii = transform(rescale(target_nii))

source = source_nii.tensor
target = target_nii.tensor
"""

# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

# set up tensors and permute
input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)


# predict
moved, warp = model(input_moving, input_fixed, registration=True)

# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)

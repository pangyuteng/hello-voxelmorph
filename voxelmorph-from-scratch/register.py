#
# COPY ADAPTED FROM voxelmorph/scripts/register.py 
# 

import os
import sys
import argparse

import torch
import torchio as tio
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

# Local imports
sys.path.append("/mnt/hd1/code/github/hello-voxelmorph/voxelmorph/torch/voxelmorph")
#sys.path.append("/cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/torch/voxelmorph")
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

# load and set up model
model = vxm.nn.models.VxmPairwise(
    ndim=3,
    source_channels=1,
    target_channels=1,
    nb_features=[16, 16, 16, 16, 16],
    integration_steps=0,
).to(device)
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)
model.eval()


moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

# set up tensors and permute

input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
print(input_moving.shape)

transform = tio.Resize((128,128,128))
rescale = tio.RescaleIntensity(out_min_max=(-1,1),in_min_max=(-1000,1000))

moving_nii = tio.ScalarImage(args.moving)
moving_nii = transform(rescale(moving_nii))

fixed_nii = tio.ScalarImage(args.fixed)
fixed_nii = transform(rescale(fixed_nii))

# TODO: unsure about axis 
input_moving = moving_nii.tensor.to(device)
input_moving = input_moving[None,:,:,:,:]
input_fixed = fixed_nii.tensor.to(device)
input_fixed = input_fixed[None,:,:,:,:]

# predict
warp, moved = model(input_moving, input_fixed,
            return_warped_source=True,
            return_field_type='displacement')

print(moved.shape,warp.shape)

# save moved image
if args.moved:
    moved = moved.detach().cpu().numpy().squeeze()
    print(moved.shape,'!!!')
    vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

# save warp
if args.warp:
    warp = warp.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)

"""

docker run -it -u $(id -u):$(id -g) --gpus all \
    -w $PWD -v /mnt:/mnt pangyuteng/voxelmorph:0.1.2-torch bash

tlc /mnt/hd2/data/ct-tlc-rv/CT-PET-VI-01-TLC-1.3.6.1.4.1.14519.5.2.1.297577087050970310787702792940607009472-1.3.6.1.4.1.14519.5.2.1.271234985032646055249895234670775984301.nii.gz
rv /mnt/hd2/data/ct-tlc-rv/CT-PET-VI-01-RV-1.3.6.1.4.1.14519.5.2.1.297577087050970310787702792940607009472-1.3.6.1.4.1.14519.5.2.1.15251667716707171236234114297150998860.nii.gz

python register.py \
    --moving /mnt/hd2/data/ct-tlc-rv/CT-PET-VI-01-TLC-1.3.6.1.4.1.14519.5.2.1.297577087050970310787702792940607009472-1.3.6.1.4.1.14519.5.2.1.271234985032646055249895234670775984301.nii.gz \
    --fixed /mnt/hd2/data/ct-tlc-rv/CT-PET-VI-01-RV-1.3.6.1.4.1.14519.5.2.1.297577087050970310787702792940607009472-1.3.6.1.4.1.14519.5.2.1.15251667716707171236234114297150998860.nii.gz \
    --moved ok.nii.gz \
    --model output/best.pt \
    -g 0

"""
import sys
import os
import random
import argparse
import numpy as np
import tensorflow as tf
from nibabel.processing import resample_to_output
import nibabel as nib
import voxelmorph as vxm
from voxelmorph.py.utils import jacobian_determinant


tf.compat.v1.experimental.output_all_intermediates(True) # https://github.com/tensorflow/tensorflow/issues/54458

# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument("--fixed",required=True,type=str)
parser.add_argument("--moving",required=True,type=str)
parser.add_argument("--moved",required=True,type=str)
parser.add_argument("--warp",type=str)
parser.add_argument("--jdet",type=str)
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--disable-oneshot', action='store_true')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of training epochs (default: 2)')
parser.add_argument('--steps-per-epoch', type=int, default=10,
                    help='frequency of model saves (default: 10)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01,
                    help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--kl-lambda', type=float, default=10,
                    help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                    help='image noise parameter for miccai 2018 network (recommended value is 0.02 when --use-probs is enabled)')  # nopep8
args = parser.parse_args()

# load and prepare training data
train_files = [args.fixed,args.moving]

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel
# TODO: let add_feat_axis be false, then add vessel mask.
# TODO: 

# scan-to-scan generator
generator = vxm.generators.scan_to_scan(
    train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape and number of features from sampled input
sample_shape = next(generator)[0][0].shape
inshape = sample_shape[1:-1]
nfeats = sample_shape[-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# tensorflow device handling
device, nb_devices = vxm.utils.setup_device(args.gpu)
assert np.mod(args.batch_size, nb_devices) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.keras')

config = dict(inshape=inshape, input_model=None)
model=vxm.networks.VxmDense.load(args.load_weights, **config)

# prepare image loss
# parser.add_argument('--image-loss', default='mse',
#                     help='image reconstruction loss - can be mse or ncc (default: mse)')
# if args.image_loss == 'ncc':
#     image_loss_func = vxm.losses.NCC().loss
# elif args.image_loss == 'mse':
#     image_loss_func = vxm.losses.MSE(args.image_sigma).loss
# else:
#     raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

from mylosses import SSIDLoss, SSIDSSVMDLoss

image_loss_func = SSIDLoss().loss
# need two image loss functions if bidirectional
if args.bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func,]
    weights = [1]

# prepare deformation loss
if args.use_probs:
    flow_shape = model.outputs[-1].shape[1:-1]
    losses += [vxm.losses.KL(args.kl_lambda, flow_shape).loss]
else:
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]

weights += [args.lambda_weight]

# multi-gpu support
if nb_devices > 1:
    save_callback = vxm.networks.ModelCheckpointParallel(save_filename)
    model = tf.keras.utils.multi_gpu_model(model, gpus=nb_devices)
else:
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, 
                                                    save_freq=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=losses, loss_weights=weights)

# save starting weights
model.save(save_filename.format(epoch=args.initial_epoch))
if not args.disable_oneshot:

    model.fit(generator,
            initial_epoch=args.initial_epoch,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=[save_callback],
            verbose=1
            )

def myload(nifti_file,minval=-1000,maxval=1000,out_minval=0,out_maxval=1,target_sz=128):
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
    out_img = ( (out_img-minval)/(maxval-minval) ).clip(out_minval,out_maxval)
    out_img = out_img[np.newaxis, ... , np.newaxis]
    return out_img_obj, out_img, out_img_obj.affine

moving_obj, moving, _ = myload(args.moving,minval=0,maxval=1)
fixed_obj, fixed, fixed_affine = myload(args.fixed,minval=0,maxval=1)

inshape = moving.shape[1:-1]
nb_feats = 1

with tf.device(device):
    warp = model.register(moving, fixed)
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

vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)


"""
pangyuteng/voxelmorph:0.1.1 DOES NOT WORK on V100
pangyuteng/voxelmorph:0.1.2 inference work on V100
BUT V100 training model.fit erros out, lib ver tweak needed

docker run --gpus device=4 --memory=40g -it \
-u $(id -u):$(id -g) --gpus device=4 \
-w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid \
pangyuteng/voxelmorph:0.1.1 bash

USEING RTX8000 with 0.1.1

CUDA_VISIBLE_DEVICES=0 python register_one_shot.py --disable-oneshot \
--fixed /radraid/pteng-public/tlc-rv-10123-downsampled/c290eec88fd4213abe64b16a240e0c63/682dae4d51d5e71bde5f0ee998f9b79d/img.nii.gz \
--moving /radraid/pteng-public/tlc-rv-10123-downsampled/c290eec88fd4213abe64b16a240e0c63/78db5d85b47261bced5650064f3e61d5/img.nii.gz \
--moved moved-0.nii.gz \
--model tmp --load-weights scripts/shapes-dice-vel-3-res-8-16-32-256f.h5

CUDA_VISIBLE_DEVICES=0 python register_one_shot.py \
--fixed /radraid/pteng-public/tlc-rv-10123-downsampled/c290eec88fd4213abe64b16a240e0c63/682dae4d51d5e71bde5f0ee998f9b79d/img.nii.gz \
--moving /radraid/pteng-public/tlc-rv-10123-downsampled/c290eec88fd4213abe64b16a240e0c63/78db5d85b47261bced5650064f3e61d5/img.nii.gz \
--moved moved-one-shot.nii.gz \
--model tmp --load-weights scripts/shapes-dice-vel-3-res-8-16-32-256f.h5

"""
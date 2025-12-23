
raise NotImplementedError("""

docker run --memory=100g -it -u $(id -u):$(id -g) --gpus '"device=4,5,6,7"' -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid pangyuteng/voxelmorph:0.1.2-tf-1.11 bash

docker run --memory=100g -it -u $(id -u):$(id -g) --gpus '"device=4,5,6,7"' -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid pangyuteng/voxelmorph:0.1.2-tf-1.11 bash

pangyuteng/voxelmorph:0.1.2-tf-1.11
AttributeError: module 'tensorflow.keras.utils' has no attribute 'multi_gpu_model'

TODO:

+ multigpu not working?!?!

+ pretrain with synthmorph data?
[ ] download and resample to 256^3
[ ] for image channel remove non-lung area
[ ] add channels so you have lobes and vessels as additional channels.


""")

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


# disable eager execution
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True) # https://github.com/tensorflow/tensorflow/issues/54458

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='optional atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
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
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01,
                    help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--kl-lambda', type=float, default=10,
                    help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                    help='image noise parameter for miccai 2018 network (recommended value is 0.02 when --use-probs is enabled)')  # nopep8
args = parser.parse_args()

prob_same = 0.05

# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size,
                                             bidir=args.bidir,
                                             add_feat_axis=add_feat_axis,prob_same=prob_same)
else:
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
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

if True:
    # build the model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        use_probs=args.use_probs,
        int_steps=args.int_steps,
        int_resolution=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats
    )

    # load initial weights (if provided)
    if args.load_weights:
        model.load_weights(args.load_weights)
else:
    config = dict(inshape=inshape, input_model=None)
    model=vxm.networks.VxmDense.load(args.load_weights, **config)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE(args.image_sigma).loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if args.bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
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
                                                       save_freq=20 * args.steps_per_epoch)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=losses, loss_weights=weights)

# save starting weights
model.save(save_filename.format(epoch=args.initial_epoch))

model.fit(generator,
         initial_epoch=args.initial_epoch,
         epochs=args.epochs,
         steps_per_epoch=args.steps_per_epoch,
         callbacks=[save_callback],
         verbose=1
         )

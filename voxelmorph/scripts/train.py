"""
reference

https://github.com/voxelmorph/voxelmorph/issues/271
http://tutorial.voxelmorph.net

"""

import os, sys
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne


isplot = False

#from tensorflow.keras.datasets import mnist
#(x_train_load, y_train_load), (x_test_load, y_test_load) = mnist.load_data()
#digit_sel = 5
# extract only instances of the digit 5
# x_train = x_train_load[y_train_load==digit_sel, ...]
# y_train = y_train_load[y_train_load==digit_sel]
# x_test = x_test_load[y_test_load==digit_sel, ...]
# y_test = y_test_load[y_test_load==digit_sel]

# # let's get some shapes to understand what we loaded.
# print('shape of x_train: {}, y_train: {}'.format(x_train.shape, y_train.shape))

# nb_val = 1000  # keep 1,000 subjects for validation
# x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
# y_val = y_train[-nb_val:]
# x_train = x_train[:-nb_val, ...]
# y_train = y_train[:-nb_val]

#nb_vis = 5
# choose nb_vis sample indexes
#idx = np.random.choice(x_train.shape[0], nb_vis, replace=False)
#example_digits = [f for f in x_train[idx, ...]]

# # plot
# if isplot:
#     ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True)

# fix data
# x_train = x_train.astype('float')/255
# x_val = x_val.astype('float')/255
# x_test = x_test.astype('float')/255

# # verify
# print('training maximum value', x_train.max())

# example_digits = [f for f in x_train[idx, ...]]
# if isplot:
#     ne.plot.slices(example_digits, cmaps=['gray'], do_colorbars=True)

# pad_amount = ((0, 0), (2,2), (2,2))

# # fix data
# x_train = np.pad(x_train, pad_amount, 'constant')
# x_val = np.pad(x_val, pad_amount, 'constant')
# x_test = np.pad(x_test, pad_amount, 'constant')

# # verify
# print('shape of training data', x_train.shape)

# configure unet input shape (concatenation of moving and fixed images)
weight_file = 'shapes-dice-vel-3-res-8-16-32-256f.h5'
ndim = 2
unet_input_features = 2
#inshape = (*x_train.shape[1:], unet_input_features)
x_train_shape = [128,128,128]
inshape = (*x_train_shape, unet_input_features)

# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

# build model
unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)

print('input shape: ', unet.input.shape)
print('output shape:', unet.output.shape)

# transform the results into a flow field.
disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)

# check tensor shape
print('displacement tensor:', disp_tensor.shape)

# using keras, we can easily form new models via tensor pointers
def_model = tf.keras.models.Model(unet.inputs, disp_tensor)

# build transformer layer
spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

# extract the first frame (i.e. the "moving" image) from unet input tensor
moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)

# warp the moving image with the transformer
moved_image_tensor = spatial_transformer([moving_image, disp_tensor])

outputs = [moved_image_tensor, disp_tensor]
vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)

# build model using VxmDense
inshape = x_train.shape[1:]
vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
#config = dict(inshape=inshape, input_model=None)
#unet = vxm.networks.VxmDense.load(weight_file, **config)
vxm_model.load_weights(weight_file)
sys.exit(1)
# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

# let's test it
train_generator = vxm_data_generator(x_train)
in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 0] for img in in_sample + out_sample] 
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
if isplot:
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)

nb_epochs = 10
steps_per_epoch = 100
hist = vxm_model.fit_generator(
    train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2)

import matplotlib.pyplot as plt

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

if isplot:
    plot_history(hist)

# let's get some data
val_generator = vxm_data_generator(x_val, batch_size = 1)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
images = [img[0, :, :, 0] for img in val_input + val_pred] 
titles = ['moving', 'fixed', 'moved', 'flow']

if isplot:
    ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
    ne.plot.flow([val_pred[1].squeeze()], width=5)


"""

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid \
    pangyuteng/voxelmorph:latest bash



"""
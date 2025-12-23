

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
import neurite as ne
import voxelmorph as vxm

im_scales = [16, 32, 64]
im_max_std = 1
in_shape = [160, 160, 192]
num_labels = 26
im = ne.utils.augment.draw_perlin(
    out_shape=(*in_shape, num_labels),
    scales=im_scales, 
    max_std=im_max_std)

print(im.shape,np.max(im),np.min(im))

"""

docker run --gpus all -it -u $(id -u):$(id -g) \
-w $PWD -v /mnt:/mnt \
pangyuteng/voxelmorph:0.1.2-tf-synth bash

  "in_shape": [160, 160, 192],
  "num_labels": 26,
  "num_maps": 100,
  "im_scales": [16, 32, 64],
  "def_scales": [8, 16, 32],
  "im_max_std": 1,

"""
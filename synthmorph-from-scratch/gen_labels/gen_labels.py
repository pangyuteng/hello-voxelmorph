
import sys
import os
import argparse
import tqdm
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib

import neurite as ne
import voxelmorph as vxm



def generate_label_maps(idx,in_shape, num_labels, num_maps, im_scales, def_scales, im_max_std, def_max_std,
                        save_label, label_dir, add_str=''):

    label_maps = []

    num_dim = len(in_shape)

    for _ in tqdm.tqdm(range(num_maps)):
        if _ != idx:
            continue
        # Draw image and warp.
        im = ne.utils.augment.draw_perlin(
            out_shape=(*in_shape, num_labels),
            scales=im_scales, max_std=im_max_std,
        )
        warp = ne.utils.augment.draw_perlin(
            out_shape=(*in_shape, num_labels, num_dim),
            scales=def_scales, max_std=def_max_std,
        )

        # Transform and create label map.
        im = vxm.utils.transform(im, warp)
        lab = tf.argmax(im, axis=-1)
        #label_maps.append(np.uint8(lab))
        os.makedirs(label_dir, exist_ok=True)
        i = _
        lab_map = np.uint8(lab)
        ni_img = nib.Nifti1Image(lab_map, affine=np.eye(4))
        nib.save(ni_img, os.path.join(label_dir, f'label_map_{add_str}{i + 1}.nii.gz'))

if __name__ == "__main__":

    # parse command line
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=f'Train a SynthMorph model on images synthesized from label maps.')

    # data organization parameters
    p.add_argument('idx',type=int)
    p.add_argument('--config-path', default='config/config.json',
                   help='config file with the training parameters specified')
    
    arg = p.parse_args()

    with open(arg.config_path) as config_file:
        data = json.load(config_file)

    label_maps = generate_label_maps(arg.idx,data['in_shape'], data['num_labels'], data['num_maps'], data['im_scales'],
                                        data['def_scales'], data['im_max_std'], data['def_max_std'],
                                        data['save_label'], data['label_dir'], data['add_str'])

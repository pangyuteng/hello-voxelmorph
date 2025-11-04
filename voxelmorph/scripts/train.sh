#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python \
    /opt/train.py \
    --img-list tl.csv \
    --model-dir /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir \
    --gpu 0 --batch-size 2 \
    --initial-epoch 1500 --epochs 3000 --legacy-image-sigma 0.02 --use_probs \
    --load-weights /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir/1500.h5

#    --initial-epoch 1140 \
#    --load-weights /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir/1500.h5

#    --load-weights /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/shapes-dice-vel-3-res-8-16-32-256f.h5

# --enc 256 256 256 256 \
# --dec 256 256 256 256 256 256 \
# --int-downsize 2 \
# --lr 1e-4 \
# --lambda 0.05 \
# --kl-lambda 10 \
# --int-steps 10 \
# --gpu 0 --batch-size 6 \
#
#--load-weights /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir/0020.h5
# {'name': 'vxm_dense', 'reg_field': 'preintegrated', 
# 'fill_value': None, 'hyp_model': None, 'input_model': None,
# 'unet_half_res': True, 'trg_feats': 1, 'src_feats': 1,
# 'use_probs': False, 'bidir': False, 'int_downsize': 2, 
# 'int_resolution': 2, 'svf_resolution': 1, 'int_steps': 10,
# 'nb_unet_conv_per_level': 1, 'unet_feat_mult': 1, 
#'nb_unet_levels': None, 'nb_unet_features': 
# [[256, 256, 256, 256], [256, 256, 256, 256, 256, 256]], 
#'inshape': [128, 128, 128], 'metadata': {}}

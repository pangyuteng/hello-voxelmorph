#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python \
    /cvibraid/cvib2/apps/personal/pteng/github/voxelmorph/scripts/tf/train.py \
    --img-list tl.csv \
    --model-dir /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir \
    --gpu 0 --batch-size 2 \
    --int-downsize 0

#
# --load-weights /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/shapes-dice-vel-3-res-8-16-32-256f.h5 \
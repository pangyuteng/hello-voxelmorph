#!/bin/bash
cd /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/torch
#/opt/train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
    --img-list tl.csv \
    --model-dir /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/torch/scripts/workdir \
    --gpu 0 --batch-size 1
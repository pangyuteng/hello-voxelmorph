#!/bin/bash
cd /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/tf

CUDA_VISIBLE_DEVICES=0 python train_256.py \
    --img-list scripts/tl.csv \
    --model-dir scripts/workdir \
    --gpu 0 --batch-size 1 

# --gpu 0,1,2,3 --batch-size 4


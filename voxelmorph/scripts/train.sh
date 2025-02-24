#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /opt/voxelmorph/scripts/tf/train.py \
    --img-list tl.csv --gpu 0  --batch-size 2
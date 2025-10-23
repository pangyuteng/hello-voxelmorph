#!/bin/bash

echo "$@"
fixed_file=$1
moving_file=$2
moved_file=$3

CUDA_VISIBLE_DEVICES=0 python \
    /opt/register.py \
    --gpu 0 \
    --model  /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/1220.h5 \
    --fixed ${fixed_file} --moving ${moving_file} --moved ${moved_file}

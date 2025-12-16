#!/bin/bash

echo "$@"
fixed_file=$1
moving_file=$2
moved_file=$3
weight_file=$4

CUDA_VISIBLE_DEVICES=0 python \
    /opt/register.py \
    --gpu 0 \
    --fixed ${fixed_file} --moving ${moving_file} \
    --moved ${moved_file} --model ${weight_file}
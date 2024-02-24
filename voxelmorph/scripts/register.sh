#!/bin/bash

echo "$@"
fixed_file=$1
moving_file=$2
moved_file=$3

python register.py \
    --gpu 0 \
    --model shapes-dice-vel-3-res-8-16-32-256f.h5 \
    --size "(128,128,128)" --rescale 4 \
    --fixed ${fixed_file} --moving ${moving_file} --moved ${moved_file}

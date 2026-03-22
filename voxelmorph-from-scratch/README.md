
# start with 128^3

initial weights using synthmorph, training with tlc,rv

image, random distortion
image, blend with random, lung distortion
image -> modified to set non-lung to 0
image -> include lobar mask, vessel mask ( or vessel loss?)
have combined loss: image perceptual?, mse, sstvd, vessel, lobe

showerthough, pulmonary vessel label network

# start with 128^3


+ download data

/mnt/hd1/code/github/ct-tlc-rv-regis

saved nifti files as txt 

+ segment lobes and vessel

+ build environment

cd /mnt/hd1/code/github/hello-voxelmorph/voxelmorph/torch
bash build_and_push.sh

+ train

--runtime=nvidia --gpus all --gpus '"device=2,3"'

docker run -it -u $(id -u):$(id -g) --gpus all \
    -w $PWD -v /mnt:/mnt pangyuteng/voxelmorph:0.1.2-torch bash

python train.py --img-list mylist.txt
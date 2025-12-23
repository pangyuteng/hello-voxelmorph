

testing out https://github.com/ivadomed/multimodal-registration


```
Voxelmorph commit: 52dd120f3ae9b0ab0fde5d0efe50627a4528bc9f
Neurite commit: c7bb05d5dae47d2a79e0fe5a8284f30b2304d335
Pystrum commit: 8cd5c483195971c0c51e9809f33aa04777aa35c8

bash build_and_push.sh

docker run --gpus all -it -u $(id -u):$(id -g) \
-w $PWD -v /mnt:/mnt \
pangyuteng/voxelmorph:0.1.2-tf-synth bash

python train_synthmorph.py --config-path config/config.json

```

--- 
# Python 3.9, Tensorflow 2.7.0 and Keras 2.7.0.

git submodule add git@github.com:adalca/pystrum.git
git submodule add git@github.com:adalca/neurite.git
git submodule add git@github.com:voxelmorph/voxelmorph.git
git submodule add git@github.com:ivadomed/multimodal-registration.git

Voxelmorph commit: 52dd120f3ae9b0ab0fde5d0efe50627a4528bc9f
Neurite commit: c7bb05d5dae47d2a79e0fe5a8284f30b2304d335
Pystrum commit: 8cd5c483195971c0c51e9809f33aa04777aa35c8



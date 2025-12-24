

testing out https://github.com/ivadomed/multimodal-registration


+ [x] generated synthmorph data shape of 256^3 using scripts in `gen_labels`

```
cd gen_labels
python gen_args.py > my.args
condnor_submit condor.sub
```

+ [ ] see if training script works  at 256^3

    + [ ] with multigpu
        + [ ] if multigpu fails, use single gpu

```

# using 4 Quadro RTX 8000 vram 49GB

cd synthmorph-from-scratch

docker run --memory=200g --cpus=32 --cpuset-cpus=0-32 -it -u $(id -u):$(id -g) --gpus '"device=4,5"' -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid pangyuteng/voxelmorph:0.1.2-tf-synth bash

python /opt/multimodal-registration/train_synthmorph.py --config-path config.json

```

---

```
Voxelmorph commit: 52dd120f3ae9b0ab0fde5d0efe50627a4528bc9f
Neurite commit: c7bb05d5dae47d2a79e0fe5a8284f30b2304d335
Pystrum commit: 8cd5c483195971c0c51e9809f33aa04777aa35c8
multimodal-registration commit: 2302e6418d5d73af3b2d9c417362d673b3a5e90e

bash build_and_push.sh

--gpus "device=4,5,6,7"'
--gpus all 
--shm-size=2g


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


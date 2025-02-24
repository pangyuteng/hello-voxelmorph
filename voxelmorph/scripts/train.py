"""
reference

https://github.com/voxelmorph/voxelmorph/issues/271
http://tutorial.voxelmorph.net

"""

"""

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid \
    pangyuteng/voxelmorph:latest bash

python /opt/voxelmorph/scripts/tf/train.py



"""
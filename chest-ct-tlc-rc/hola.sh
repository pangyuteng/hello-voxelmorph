#!/bin/bash

docker run -it -u $(id -u):$(id -g) -w $PWD -v /cvibraid:/cvibraid pangyuteng/voxelmorph:latest \
bash -c "bash test/test_register_transform.sh"

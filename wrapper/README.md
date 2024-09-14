

```

build_and_push.sh

docker run -it -u $(id -u):$(id -g) pangyuteng/voxelmorph-wrapper:0.1.0 pip -c "list"

docker run -it -u $(id -u):$(id -g) -w $PWD -v /cvibraid:/cvibraid pangyuteng/voxelmorph-wrapper:0.1.0 bash

```
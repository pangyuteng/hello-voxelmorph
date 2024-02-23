
```

docker run -it -u $(id -u):$(id -g) \
    -v /cvibraid:/cvibraid -v /radraid:/radraid \
    pangyuteng/voxelmorph bash


./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0

```
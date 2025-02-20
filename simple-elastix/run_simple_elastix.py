

"""

https://github.com/pangyuteng/public-misc/tree/master/docker/registration/simple-elastix

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid \
    pangyuteng/simple-elastix bash

# run resample.py first. then below

python run_itk_elastix.py workdir/fixed.nii.gz workdir/moving.nii.gz workdir


'''
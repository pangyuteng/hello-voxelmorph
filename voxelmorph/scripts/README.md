
### blah

```

https://github.com/voxelmorph/voxelmorph

VoxelMorph on neurite-OASIS demo
https://colab.research.google.com/drive/1ZefmWXBupRNsnIbBbGquhVDsk-7R7L1S?usp=sharing

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid \
    pangyuteng/voxelmorph bash

```

### sample registration with sample weights from voxelmorph

```

wget https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5

condor_submit register.sub

```

### sample training


+ source file - downsampled tlc,rv nifti copied from pteng/misc-cvib-ops/rsch-10123/tlc-rv-query/gen_condor.py

tl_masked.csv




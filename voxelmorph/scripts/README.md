
### dev

```

bash build_and_push.sh

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


images-RESEARCH-10123.csv source: https://gitlab.cvib.ucla.edu/qia/chest-ct-prm/-/blob/main/condor/images-RESEARCH-10123.csv


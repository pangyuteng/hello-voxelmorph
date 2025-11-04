
docker run -it -u $(id -u):$(id -g) -w $PWD \
-v /cvibraid:/cvibraid -v /radraid:/radraid \
pangyuteng/voxelmorph:latest bash

export fixed_file=/radraid/pteng-public/tmp/RESEARCH/10123/10123_524VERDO/2005-06-10/rv.nii.gz
export moving_file=/radraid/pteng-public/tmp/RESEARCH/10123/10123_524VERDO/2005-06-10/tlc.nii.gz
export fixedsm_file=/cvibraid/cvib2/Temp/tmp/fixed.nii.gz
export movingsm_file=/cvibraid/cvib2/Temp/tmp/moving.nii.gz
export moved_file=/cvibraid/cvib2/Temp/tmp/moved-1640.nii.gz
export weight_file=/cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir/1640.h5
export warp_file=/cvibraid/cvib2/Temp/tmp/warp-1640.nii.gz
export jdet_file=/cvibraid/cvib2/Temp/tmp/jdet-1640.nii.gz
export warp_file=/cvibraid/cvib2/Temp/tmp/warp-1640.nii.gz

CUDA_VISIBLE_DEVICES=0 python \
    /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/register.py \
    --gpu 0 \
    --fixed ${fixed_file} \
    --moving ${moving_file} \
    --fixedsm ${fixedsm_file} \
    --movingsm ${movingsm_file} \
    --moved ${moved_file} \
    --warp ${warp_file} \
    --jdet ${jdet_file} \
    --model ${weight_file}


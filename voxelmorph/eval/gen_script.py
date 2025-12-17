import os
import pandas as pd

master_csv_file = "/cvibraid/cvib2/apps/personal/pteng/misc-cvib-ops/rsch-10123/tlc-rv-query/tl_master.csv"
full_dataset_csv_file = "tl_masked.csv"
test_csv_file = "tl-for-testing.csv"

full_df = pd.read_csv(master_csv_file)
#full_df = pd.read_csv(full_dataset_csv_file)

df = pd.read_csv(test_csv_file,header=None)
timepoint_hash_list = df[0].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))) ).to_list()
timepoint_hash_list = list(set(timepoint_hash_list))

# timepoint_hash,series_hash,breath_hold,nifti_file

mydict = {
    "pre_training": {
        "weight_file":"/cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/shapes-dice-vel-3-res-8-16-32-256f.h5",
        "out_dir":"/cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/eval/pre_training",
    },
    "post_training": {
        "weight_file":"/cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir/2380.h5",
        "out_dir":"/cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/eval/post_training",
    },
}



for timepoint_hash in timepoint_hash_list:

    tlc_row = full_df[(full_df.timepoint_hash == timepoint_hash)&(full_df.breath_hold=='tlc')]    
    rv_row = full_df[(full_df.timepoint_hash == timepoint_hash)&(full_df.breath_hold=='rv')]

    tlc_file = tlc_row.nifti_file.to_list()[0]
    rv_file = rv_row.nifti_file.to_list()[0]
    tlc_series_hash = tlc_row.series_hash.to_list()[0]

    raise NotImplementedError()
    # TODO: need to download images and mask
    # move to misc-cvi
    # environment,experiment_id,patient_id,study_date,breath_hold,series_instance_uid,series_sub_id,timepoint_hash,series_hash,seri_file,nifti_file
    #dict(tlc_row)[""]

    """fixed_file moving_file moved_file weight_file"""
    for k,vdict in mydict.items():
        weight_file = vdict["weight_file"]
        out_dir = vdict["out_dir"]
        output_folder = os.path.join(out_dir,timepoint_hash)
        moved_tlc_file = os.path.join(output_folder,f"moved_{tlc_series_hash}.nii.gz")
        warp_file = os.path.join(output_folder,f"warp.nii.gz")
        cmd = f"{rv_file} {tlc_file} {moved_tlc_file} {weight_file}"
        #print(cmd)

"""
docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid -v /radraid:/radraid \
    pangyuteng/voxelmorph bash


fixed /radraid/pteng-public/tlc-rv-10123-downsampled/5aeb4c6ca234f5f929f72f194f02ed2e/a816e5f7c3835543cc02322bfee0e06d/img.nii.gz
moving /radraid/pteng-public/tlc-rv-10123-downsampled/5aeb4c6ca234f5f929f72f194f02ed2e/3b62d55f785ff444070a919a26676e4c/img.nii.gz 
moved /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/eval/post_training/moved_3b62d55f785ff444070a919a26676e4c.nii.gz
weight /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/scripts/workdir/2380.h5
"""
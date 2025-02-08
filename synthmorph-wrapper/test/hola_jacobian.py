import os
import sys
import time
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import voxelmorph as vxm
from voxelmorph.py.utils import jacobian_determinant
from synthmorph_wrapper import register_transform as rt

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    print(out_size,'ideal')
    out_size = [512,512,512]
    print(out_size)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def main(fixed_file,fixed_mask_file,moving_file,moving_mask_file,output_folder,gpu_id):
    
    fixed_img =  sitk.ReadImage(fixed_file)
    moving_img =  sitk.ReadImage(moving_file)
    print(fixed_img.GetSize())
    print(moving_img.GetSize())
    print('---')
    spacing = [2.0, 2.0, 2.0]
    spacing = [1.0,1.0,1.0]
    fixed_img = resample_img(fixed_img, out_spacing=spacing, is_label=False)
    moving_img = resample_img(moving_img, out_spacing=spacing, is_label=False)

    print(fixed_img.GetSize())
    print(moving_img.GetSize())
    fixed_file = os.path.join(output_folder,"fixed.nii.gz")
    moving_file = os.path.join(output_folder,"moving.nii.gz")
    wrap_file = os.path.join(output_folder,"wrap.nii.gz")
    jdet_file = os.path.join(output_folder,"jdet.nii.gz")

    sitk.WriteImage(fixed_img,fixed_file)
    sitk.WriteImage(moving_img,moving_file)

    os.makedirs(output_folder,exist_ok=True)

    fixed_file,fixed_mask_file,moving_file,moving_mask_file,output_folder,gpu_id
 
    device, nb_devices = vxm.tf.utils.setup_device(gpu_id)
    add_feat_axis = not rt.MULTI_CHANNEL

    sm_fixed, fixed_affine = vxm.py.utils.load_volfile(
        fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    sm_moving = vxm.py.utils.load_volfile(moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)


    inshape = sm_moving.shape[1:-1]
    nb_feats = sm_moving.shape[-1]

    with tf.device(device):
        # load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(rt.MODEL_FILE,**config).register(sm_moving, sm_fixed)
        print(warp.shape)
        warp = warp.squeeze()
        print(warp.shape)
        jdet = jacobian_determinant(warp)



    vxm.py.utils.save_volfile(warp.squeeze(), wrap_file, fixed_affine)
    vxm.py.utils.save_volfile(jdet.squeeze(), jdet_file, fixed_affine)


if __name__ == "__main__":

    fixed_file = sys.argv[1]
    fixed_mask_file = sys.argv[2]
    moving_file = sys.argv[3]
    moving_mask_file = sys.argv[4]
    output_folder = sys.argv[5]
    gpu_id = sys.argv[6]

    # resample and crop
    start_time = time.time()
    main(fixed_file,fixed_mask_file,moving_file,moving_mask_file,output_folder,gpu_id)
    end_time = time.time()
    print(f"time {end_time-start_time}(s)")
    print("done")

"""

CUDA_VISIBLE_DEVICES=2 python hola_jacobian.py \
    moved/sm-fixed.nii.gz moved/sm-moving.nii.gz moved/jdet.nii.gz 2

CUDA_VISIBLE_DEVICES=2 python hola_jacobian.py workdir/tlc.nii.gz None workdir/rv.nii.gz None workdir 2

"""
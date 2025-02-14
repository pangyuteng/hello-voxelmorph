import os
import sys
import time
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import voxelmorph as vxm
from voxelmorph.py.utils import jacobian_determinant
#from synthmorph_wrapper import register_transform as rt
MODEL_FILE = '/opt/synthmorph_wrapper/shapes-dice-vel-3-res-8-16-32-256f.h5'
from synthmorph_wrapper.utils import rescale_intensity

def resample_img(itk_image, out_spacing, out_size, out_value, is_label):

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetDefaultPixelValue(out_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def main(fixed_file,fixed_mask_file,moving_file,moving_mask_file,output_folder,gpu_id):
    
    fixed_obj =  sitk.ReadImage(fixed_file)
    moving_obj =  sitk.ReadImage(moving_file)
    print(fixed_obj.GetSize())
    print(moving_obj.GetSize())

    if os.path.exists(fixed_mask_file):
        fixed_mask_obj =  sitk.ReadImage(fixed_mask_file)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()    
        label_shape_filter.Execute(fixed_mask_obj)
        bounding_box = label_shape_filter.GetBoundingBox(1)
        cropped_size = bounding_box[int(len(bounding_box)/2):]
        cropped_start = bounding_box[0:int(len(bounding_box)/2)]
        fixed_obj = sitk.RegionOfInterest(fixed_obj,cropped_size,cropped_start)

    if os.path.exists(moving_mask_file):
        moving_mask_obj = sitk.LabelShapeStatisticsImageFilter()    
        label_shape_filter.Execute(moving_mask_obj)
        bounding_box = label_shape_filter.GetBoundingBox(1)
        cropped_size = bounding_box[int(len(bounding_box)/2):]
        cropped_start = bounding_box[0:int(len(bounding_box)/2)]
        moving_obj = sitk.RegionOfInterest(moving_obj,cropped_size,cropped_start)
    
    print('---')
    print(fixed_obj.GetSize())
    print(moving_obj.GetSize())
    print('---')
    is_label = False
    out_value = -2048

    original_spacing = fixed_obj.GetSpacing()
    # assume spacing is same
    assert(fixed_obj.GetSpacing()==moving_obj.GetSpacing())

    # option one, fix spacing
    out_spacing = [1.0, 1.0, 1.0]
    original_size = [max([x,y]) for x,y in zip(fixed_obj.GetSize(),moving_obj.GetSize())]
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # option two, fix size, needs to be factor of 64 (check voxelmorph code)
    out_size = [128,128,128]
    out_spacing = [
        int(np.round(original_size[0] * (original_spacing[0] / out_size[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_size[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_size[2])))
    ]

    print(out_size,'!!!!!!!!!!!!')
    print(out_spacing,'!!!!!!!!!!!!')

    fixed_file = os.path.join(output_folder,"fixed.nii.gz")
    moving_file = os.path.join(output_folder,"moving.nii.gz")
    moved_file = os.path.join(output_folder,"moved.nii.gz")
    wrap_file = os.path.join(output_folder,"wrap.nii.gz")
    jdet_file = os.path.join(output_folder,"jdet.nii.gz")
    
    fixed_obj = resample_img(fixed_obj,out_spacing,out_size,out_value,is_label)
    moving_obj = resample_img(moving_obj,out_spacing,out_size,out_value,is_label)

    fixed_obj = rescale_intensity(fixed_obj)
    moving_obj = rescale_intensity(moving_obj)
    print(fixed_obj.GetSize())
    print(moving_obj.GetSize())
    sitk.WriteImage(fixed_obj,fixed_file)
    sitk.WriteImage(moving_obj,moving_file)
    print("saved...")
    os.makedirs(output_folder,exist_ok=True)
 
    device, nb_devices = vxm.tf.utils.setup_device(gpu_id)
    add_feat_axis = True

    sm_fixed, fixed_affine = vxm.py.utils.load_volfile(
        fixed_file, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    sm_moving = vxm.py.utils.load_volfile(moving_file, add_batch_axis=True, add_feat_axis=add_feat_axis)

    inshape = sm_moving.shape[1:-1]
    nb_feats = sm_moving.shape[-1]

    with tf.device(device):
        # load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(MODEL_FILE,**config).register(sm_moving, sm_fixed)
        sm_moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([sm_moving, warp])
        print(warp.shape)
        warp = warp.squeeze()
        print(warp.shape)
        jdet = jacobian_determinant(warp)

    # TODO: alternatively upsample warp to 512,512,512
    # then compute d(J) and also generate moved image.
    vxm.py.utils.save_volfile(sm_moved.squeeze(), moved_file, fixed_affine)
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

"The Jacobian is a vector field related to the gradient of the warp. 
Most people use the scalar-valued determinant of the Jacobian det(J) 
(or, even more specifically, log(det(J))) to measure the properties of the warp
The idea is that the magnitude of det(J) tells you about 
expansions (>1) or contractions (<1)."
https://discuss.afni.nimh.nih.gov/t/jacobian-meaning/2921

docker run -it -u $(id -u):$(id -g) --gpus 1 nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi

cd /cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/synthmorph-wrapper/test

docker run -it --gpus device=1 -u $(id -u):$(id -g) \
    -v /cvibraid:/cvibraid pangyuteng/synthmorph-wrapper:0.1.0 bash

CUDA_VISIBLE_DEVICES=0 python hola_jacobian.py workdir/tlc.nii.gz None workdir/rv.nii.gz None workdir 0

to compare with conventional registration, we let TLC be the moving.
if implying for TLC to RV, jacobian will be less than 1 for lung that shrinked (?)
CUDA_VISIBLE_DEVICES=0 python hola_jacobian.py workdir/rv.nii.gz None workdir/tlc.nii.gz None workdir 0



"""
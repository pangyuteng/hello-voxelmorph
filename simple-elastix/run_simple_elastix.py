

import os
import sys
import ast
import shutil
import numpy as np
import SimpleITK as sitk

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from resample import resample_img

def register(fixed_obj,moving_obj,work_dir,save_deformation=False):
    logdir = os.path.join(work_dir,"elastix")

    parameter_map_0 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/affine.txt"))
    parameter_map_1 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/bspline1.txt"))
    parameter_map_2 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/bspline2.txt"))

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(parameter_map_0)
    parameterMapVector.append(parameter_map_1)
    parameterMapVector.append(parameter_map_2)

    elastixImageFilter = sitk.ElastixImageFilter()

    elastixImageFilter.LogToFileOn()        
    os.makedirs(logdir,exist_ok=True)
    elastixImageFilter.SetOutputDirectory(logdir)

    elastixImageFilter.SetFixedImage(fixed_obj)
    elastixImageFilter.SetMovingImage(moving_obj)

    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    moved_obj = elastixImageFilter.GetResultImage()

    if save_deformation:
        # parameterMap0 = sitk.ReadParameterFile("TransformParameters.0.R0.txt`)
        # transformixImageFilter.AddParameterMap(parameterMap0)

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(moving_obj)
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.ComputeDeterminantOfSpatialJacobianOn()
        transformixImageFilter.SetOutputDirectory(logdir)
        transformixImageFilter.Execute()

        source_det_jacobian_file = os.path.join(logdir,'spatialJacobian.nii.gz')
        det_jacobian_file = os.path.join(work_dir,'det_jacobian.nii.gz')
        shutil.copy(source_det_jacobian_file,det_jacobian_file)
        deformation_file = os.path.join(work_dir,'deformation.nii.gz')
        sitk.WriteImage(transformixImageFilter.GetDeformationField(),deformation_file)

    moved_file = os.path.join(work_dir,"moved.nii.gz")
    sitk.WriteImage(moved_obj,moved_file)


def resample_foo(fixed_obj,moving_obj):
    original_size = [max([x,y]) for x,y in zip(fixed_obj.GetSize(),moving_obj.GetSize())]
    original_spacing = fixed_obj.GetSpacing()

    # using 128**3 as out_size, as this excercise is to compare jacobian from voxlmorph with those from ../hola_jacobina.py
    out_size = [128,128,128] 
    out_spacing = [
        int(np.round(original_size[0] * (original_spacing[0] / out_size[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_size[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_size[2])))
    ]

    fixed_obj = resample_img(fixed_obj, out_spacing, out_size, out_value, is_label)
    moving_obj = resample_img(moving_obj, out_spacing, out_size, out_value, is_label)
    return fixed_obj, moving_obj

if __name__ == "__main__":

    fixed_path = sys.argv[1]
    moving_path = sys.argv[2]
    is_resample = ast.literal_eval(sys.argv[3])
    work_dir = sys.argv[4]
    os.makedirs(work_dir,exist_ok=True)

    out_value = -1000
    is_label = False
    fixed_obj = sitk.ReadImage(fixed_path)

    moving_obj = sitk.ReadImage(moving_path)

    if is_resample:
        fixed_obj,moving_obj = resample_foo(fixed_obj,moving_obj)
        sitk.WriteImage(fixed_obj, os.path.join(work_dir,'fixed.nii.gz'))
        sitk.WriteImage(moving_obj,os.path.join(work_dir,'moving.nii.gz'))

    register(fixed_obj,moving_obj,work_dir,save_deformation=True)

"""

https://github.com/pangyuteng/public-misc/tree/master/docker/registration/simple-elastix

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid \
    pangyuteng/simple-elastix bash

python run_simple_elastix.py workdir/rv.nii.gz workdir/tlc.nii.gz True workdir

"""
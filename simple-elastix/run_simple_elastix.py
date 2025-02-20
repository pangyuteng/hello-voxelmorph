

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import SimpleITK as sitk

def register(fixed_obj,moving_obj,work_dir,save_deformation=False):

    parameter_map_0 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/affine.txt"))
    parameter_map_1 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/bspline1.txt"))
    parameter_map_2 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/bspline2.txt"))

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(parameter_map_0)
    parameterMapVector.append(parameter_map_1)
    parameterMapVector.append(parameter_map_2)

    elastixImageFilter = sitk.ElastixImageFilter()
    logdir = os.path.join(work_dir,"elastix")
        
    elastixImageFilter.LogToFileOn()        
    os.makedirs(logdir,exist_ok=True)
    elastixImageFilter.SetOutputDirectory(logdir)

    elastixImageFilter.SetFixedImage(fixed_obj)
    elastixImageFilter.SetMovingImage(moving_obj)

    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    moved_obj = elastixImageFilter.GetResultImage()

    if save_deformation:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(moving_obj)
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.LogToConsoleOn()
        transformixImageFilter.SetOutputDirectory(logdir)
        transformixImageFilter.Execute()
        deformation_file = os.path.join(work_dir,'deformation.nii.gz')
        sitk.WriteImage(transformixImageFilter.GetDeformationField(),deformation_file)
    
    moved_file = os.path.join(work_dir,"moved.nii.gz")
    sitk.WriteImage(moved_obj,moved_file)

if __name__ == "__main__":

    fixed_path = sys.argv[1]
    moving_path = sys.argv[2]
    work_dir = sys.argv[3]

    os.makedirs(work_dir,exist_ok=True)

    fixed_obj = sitk.ReadImage(fixed_path)
    moving_obj = sitk.ReadImage(moving_path)
    register(fixed_obj,moving_obj,work_dir)

"""

https://github.com/pangyuteng/public-misc/tree/master/docker/registration/simple-elastix

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid \
    pangyuteng/simple-elastix bash

# run resample.py first. then below

python run_simple_elastix.py workdir/fixed.nii.gz workdir/moving.nii.gz workdir

"""
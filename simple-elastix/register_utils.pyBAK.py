

raise NotImplementedError()

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import itk
# fixed_image = itk.imread('data/CT_2D_head_fixed.mha', itk.F)

# cmoving_image = itk.imread('data/CT_2D_head_moving.mha', itk.F)

# # Import Default Parameter Map
# parameter_object = itk.ParameterObject.New()
# resolutions = 3
# parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid',3)
# parameter_object.AddParameterMap(parameter_map_rigid)

# # For the bspline default parameter map, an extra argument can be specified that define the final bspline grid spacing in physical space. 
# parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", resolutions, 20.0)
# parameter_object.AddParameterMap(parameter_map_bspline)


# # .. and/or load custom parameter maps from .txt file
# parameter_object.AddParameterFile('data/parameters_BSpline.txt')

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
    registered_obj = elastixImageFilter.GetResultImage()

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

    return registered_obj

if __name__ == "__main__":

    fixed_path = sys.argv[1]
    moving_path = sys.argv[2]
    work_dir = sys.argv[3]

    os.makedirs(work_dir,exist_ok=True)

    fixed_obj = sitk.ReadImage(fixed_path)
    moving_obj = sitk.ReadImage(moving_path)
    register(fixed_obj,moving_obj,work_dir)

"""

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid \
    pangyuteng/simple-elastix-new bash

cp ../synthmorph-wrapper/test/workdir/fixed.nii.gz .
cp ../synthmorph-wrapper/test/workdir/moving.nii.gz .

python register_utils.py workdir/fixed.nii.gz workdir/moving.nii.gz workdir

"""
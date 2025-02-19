import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import itk
# fixed_image = itk.imread('data/CT_2D_head_fixed.mha', itk.F)

# cmoving_image = itk.imread('data/CT_2D_head_moving.mha', itk.F)

# # Import Default Parameter Map
# parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid',3)
# parameter_object.AddParameterMap(parameter_map_rigid)

# # For the bspline default parameter map, an extra argument can be specified that define the final bspline grid spacing in physical space. 
# parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", resolutions, 20.0)
# parameter_object.AddParameterMap(parameter_map_bspline)


# # .. and/or load custom parameter maps from .txt file
# parameter_object.AddParameterFile('data/parameters_BSpline.txt')

def register(fixed_obj,moving_obj,work_dir,save_deformation=False):

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(os.path.join(THIS_DIR,"param/affine.txt"))
    parameter_object.AddParameterFile(os.path.join(THIS_DIR,"param/bspline1.txt"))
    parameter_object.AddParameterFile(os.path.join(THIS_DIR,"param/bspline2.txt"))
    
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_obj,moving_obj)
    elastix_object.SetParameterObject(parameter_object)

    elastix_object.SetLogToConsole(False)

    # # You can set this to 1, but setting any number > 1 will result in a segmentation fault
    # elastix_object.SetNumberOfThreads(2)

    elastix_object.UpdateLargestPossibleRegion()
    registered_obj = elastix_object.GetOutput()


    # if save_deformation:
    #     transformixImageFilter = sitk.TransformixImageFilter()
    #     transformixImageFilter.SetMovingImage(moving_obj)
    #     transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    #     transformixImageFilter.ComputeDeformationFieldOn()
    #     transformixImageFilter.LogToConsoleOn()
    #     transformixImageFilter.SetOutputDirectory(logdir)
    #     transformixImageFilter.Execute()
    #     deformation_file = os.path.join(work_dir,'deformation.nii.gz')
    #     sitk.WriteImage(transformixImageFilter.GetDeformationField(),deformation_file)

    # return registered_obj

if __name__ == "__main__":

    fixed_path = sys.argv[1]
    moving_path = sys.argv[2]
    work_dir = sys.argv[3]

    os.makedirs(work_dir,exist_ok=True)

    fixed_obj = itk.imread(fixed_path, itk.F)
    moving_obj = itk.imread(moving_path, itk.F)
    register(fixed_obj,moving_obj,work_dir)

"""

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid \
    pangyuteng/simple-elastix-new bash

cp ../synthmorph-wrapper/test/workdir/fixed.nii.gz .
cp ../synthmorph-wrapper/test/workdir/moving.nii.gz .

python run_itk_elastix.py workdir/fixed.nii.gz workdir/moving.nii.gz workdir

"""
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import itk

    
def register(fixed_obj,moving_obj,work_dir,save_deformation=False):

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(os.path.join(THIS_DIR,"param/affine.txt"))
    #parameter_object.AddParameterFile(os.path.join(THIS_DIR,"param/bspline1.txt"))
    #parameter_object.AddParameterFile(os.path.join(THIS_DIR,"param/bspline2.txt"))

    # https://github.com/InsightSoftwareConsortium/ITKElastix/blob/2f5756fc3970248da2565b4b87ca3df0a592d133/examples/ITK_Example10_Transformix_Jacobian.ipynb#L7
    output_directory = os.path.join(work_dir,'itk-out')
    os.makedirs(output_directory,exist_ok=True)
    moved_obj, result_transform_parameters = itk.elastix_registration_method(
        fixed_obj,moving_obj,
        parameter_object=parameter_object,
        log_to_console=True,
        output_directory=output_directory)

    # Calculate Jacobian matrix and it's determinant in a tuple
    jacobians = itk.transformix_jacobian(
        moving_obj, result_transform_parameters,
        log_to_console=True,
        output_directory=output_directory
    )

    # Casting tuple to two numpy matrices for further calculations.
    spatial_jacobian = np.asarray(jacobians[0]).astype(np.float32)
    det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float32)
    print(det_spatial_jacobian.shape)

    """
    Inspect the deformation field by looking at the determinant of the Jacobian of Tµ(x). 
    Values smaller than 1 indicate local compression, values larger than 1 indicate local 
    expansion, and 1 means volume preservation. The measure is quantitative: a value of
     1.1 means a 10% increase in volume. If this value deviates substantially from 1,
      you may be worried (but maybe not if this is what you expect for your application). 
      In case it is negative you have “foldings” in your transformation, and you definitely
       should be worried. For more information see elastix manual.
    """
    moved_file = os.path.join(work_dir,'moved.nii.gz')
    itk.imwrite(moved_obj,moved_file)

    # detj_obj = itk.GetImageFromArray(det_spatial_jacobian)
    # detj_obj.CopyInformation(moved_file)
    # detj_file = os.path.join(work_dir,'detj.nii.gz')
    # itk.imwrite(detj_obj,detj_file)

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
    raise ValueError("ABANDONED, resorted to SimpleITK binding, whats the point of itk-elastix, if at the end we are using Elastix")
    fixed_path = sys.argv[1]
    moving_path = sys.argv[2]
    work_dir = sys.argv[3]

    os.makedirs(work_dir,exist_ok=True)

    fixed_obj = itk.imread(fixed_path, itk.F)
    moving_obj = itk.imread(moving_path, itk.F)
    register(fixed_obj,moving_obj,work_dir)

"""

https://github.com/pangyuteng/public-misc/tree/master/docker/registration/itk-elastix

docker run -it -u $(id -u):$(id -g) \
    -w $PWD -v /cvibraid:/cvibraid \
    pangyuteng/itk-elastix bash

# run resample.py first. then below

python run_itk_elastix.py workdir/fixed.nii.gz workdir/moving.nii.gz workdir

"""
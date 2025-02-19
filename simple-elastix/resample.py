import SimpleITK as sitk
import numpy as np

def resample_img(img_obj, out_spacing, out_size, out_value, is_label):

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(img_obj.GetDirection())
    resample.SetOutputOrigin(img_obj.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(img_obj.GetPixelIDValue())
    resample.SetDefaultPixelValue(out_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(img_obj)

if __name__ == "__main__":
    out_value = -1000
    is_label = False
    fixed_obj = sitk.ReadImage("workdir/tlc.nii.gz")
    moving_obj = sitk.ReadImage("workdir/rv.nii.gz")

    original_size = [max([x,y]) for x,y in zip(fixed_obj.GetSize(),moving_obj.GetSize())]
    original_spacing = fixed_obj.GetSpacing()
    out_size = [128,128,128]
    out_spacing = [
        int(np.round(original_size[0] * (original_spacing[0] / out_size[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_size[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_size[2])))
    ]

    print(out_size,'!!!!!!!!!!!!')
    print(out_spacing,'!!!!!!!!!!!!')

    out_obj = resample_img(fixed_obj, out_spacing, out_size, out_value, is_label)
    sitk.WriteImage(out_obj,'workdir/fixed.nii.gz')
    out_obj = resample_img(moving_obj, out_spacing, out_size, out_value, is_label)
    sitk.WriteImage(out_obj,'workdir/moving.nii.gz')
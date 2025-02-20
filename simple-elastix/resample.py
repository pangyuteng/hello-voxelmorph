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
    pass
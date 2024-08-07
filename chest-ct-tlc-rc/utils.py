import os
import sys
import ast
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from skimage.measure import label, regionprops

# removes dots (human error) outside the lung.
#
def remove_dots(mask_obj):
    mask = sitk.GetArrayFromImage(mask_obj)
    label_img = label(mask)
    regions = sorted(regionprops(label_img),key=lambda r: r.area,reverse=True)
    print('len(regions)',len(regions))
    if len(regions) > 2:
        print("dots found, removing...")
        regions = [x for x in regions if x.area > 5] # arbitrary threshold
    arr = np.zeros_like(mask).astype(np.int16)
    for x in regions:
        arr[label_img==x.label] = 1
    tgt_obj = sitk.GetImageFromArray(arr)
    tgt_obj.SetSpacing(mask_obj.GetSpacing())
    tgt_obj.SetOrigin(mask_obj.GetOrigin())
    tgt_obj.SetDirection(mask_obj.GetDirection())
    return tgt_obj

#
# observation, lung contours may have holes especially in hrct due to CAD wont segment vessels.
# to compute dice and view masked suv
# to be consistent, we fill up the holes for the lung masks for both modalities
#
def hole_fill(mask_obj):
    mask = sitk.GetArrayFromImage(mask_obj)
    label_img = label(mask==0)
    regions = sorted(regionprops(label_img),key=lambda r: r.area,reverse=True)
    bkgd_val = regions[0].label
    mask[label_img!=bkgd_val] = 1
    mask = morphology.binary_closing(mask).astype(np.int16)
    tgt_obj = sitk.GetImageFromArray(mask)
    tgt_obj.SetSpacing(mask_obj.GetSpacing())
    tgt_obj.SetOrigin(mask_obj.GetOrigin())
    tgt_obj.SetDirection(mask_obj.GetDirection())
    return tgt_obj

def rescale_intensity(src_obj,mask_obj=None):
    
    min_val, max_val = -1000, 1000
    arr = sitk.GetArrayFromImage(src_obj)
    min_val, max_val = np.percentile(arr,[10,90])
    clampFilt = sitk.ClampImageFilter()
    clampFilt.SetLowerBound(min_val)
    clampFilt.SetUpperBound(max_val)
    src_obj = clampFilt.Execute(src_obj)
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(1.0)
    rescalFilt.SetOutputMinimum(0.0)
    # Reads the image using SimpleITK
    tmp_obj = rescalFilt.Execute(sitk.Cast(src_obj, sitk.sitkFloat32))
    if mask_obj:
        mask = sitk.GetArrayFromImage(mask_obj)
        img = sitk.GetArrayFromImage(tmp_obj)

        dilated = morphology.binary_dilation(
            mask, morphology.ball(radius=3)
        )
        border = np.logical_and(dilated==1,mask==0)
        img[border==1]=1.0

        tgt_obj = sitk.GetImageFromArray(img)
        tgt_obj.SetSpacing(tmp_obj.GetSpacing())
        tgt_obj.SetOrigin(tmp_obj.GetOrigin())
        tgt_obj.SetDirection(tmp_obj.GetDirection())
    else:
        tgt_obj = tmp_obj
    return tgt_obj

def resample(src_obj,out_size,method=sitk.sitkNearestNeighbor):
    src_size = np.array(src_obj.GetSize())
    src_spacing = np.array(src_obj.GetSpacing())
    tgt_size = np.array(out_size)
    tgt_spacing = src_size*src_spacing/tgt_size
    ref_obj = sitk.Image(out_size, sitk.sitkInt16)
    ref_obj.SetDirection(src_obj.GetDirection())
    ref_obj.SetOrigin(src_obj.GetOrigin())
    ref_obj.SetSpacing(tgt_spacing)
    tgt_obj = sitk.Resample(
        src_obj, ref_obj, sitk.Transform(),
        method, 0, src_obj.GetPixelID(),
    )
    return tgt_obj

def elastix_register_and_transform(fixed_file,moving_file,moving_list=[]):

    fixed = sitk.ReadImage(fixed_file)
    moving = sitk.ReadImage(moving_file)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetOutputDirectory('/tmp')
    
    defaultTranslationParameterMap = sitk.GetDefaultParameterMap("translation")
    defaultTranslationParameterMap['DefaultPixelValue'] = ['-1000']
    defaultTranslationParameterMap['MaximumNumberOfIterations'] = ['512'] 
    defaultAffineParameterMap = sitk.GetDefaultParameterMap("affine")
    defaultAffineParameterMap['DefaultPixelValue'] = ['-1000']
    defaultAffineParameterMap['MaximumNumberOfIterations'] = ['512'] 
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(defaultTranslationParameterMap)
    parameterMapVector.append(defaultAffineParameterMap)
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.Execute()

    for moving_file,moved_file,out_pixel_value,is_mask in moving_list:

        transform_tuple = elastixImageFilter.GetTransformParameterMap()
        transform = list(transform_tuple)
        transform[-1]['DefaultPixelValue']=[str(out_pixel_value)]
        if is_mask:
            transform[-1]['FinalBSplineInterpolationOrder']=["0"]
            transform[-1]["ResultImagePixelType"] = ["int"]    

        #transform_tuple = (transform,)
        og_obj = sitk.ReadImage(moving_file)
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(og_obj)
        transformixImageFilter.SetTransformParameterMap(transform_tuple)
        transformixImageFilter.SetOutputDirectory("/tmp")
        transformixImageFilter.LogToConsoleOn()
        elastixImageFilter.LogToFileOn()
        transformixImageFilter.Execute()
        moved = transformixImageFilter.GetResultImage()
        moved = sitk.Cast(moved,og_obj.GetPixelID())
        sitk.WriteImage(moved,moved_file)


if __name__ == "__main__":
    pass
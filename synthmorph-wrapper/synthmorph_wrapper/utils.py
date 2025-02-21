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
from skimage.segmentation import watershed

def int_hole_fill(image_obj):
    image = sitk.GetArrayFromImage(image_obj)
    image = image.astype(np.int)
    
    # locate background
    label_img = label(image==0)
    regions = sorted(regionprops(label_img),key=lambda r: r.area,reverse=True)
    bkgd_val = regions[0].label
    # generate foreground
    mask = np.zeros_like(image)
    mask[label_img!=bkgd_val] = 1
    mask = morphology.binary_closing(mask).astype(np.int16)
    # watershed with existing foreground values
    distance = mask.astype(np.int)
    image_watershed = watershed(-distance, image, mask=mask)

    tgt_obj = sitk.GetImageFromArray(image_watershed)
    tgt_obj.SetSpacing(image_obj.GetSpacing())
    tgt_obj.SetOrigin(image_obj.GetOrigin())
    tgt_obj.SetDirection(image_obj.GetDirection())
    return tgt_obj

def rescale_intensity(src_obj,mask_obj=None,min_val=-1000,max_val=1000,out_min_val=0.0,out_max_val=1.0):

    arr = sitk.GetArrayFromImage(src_obj)
    if min_val is None or max_val is None:
        min_val, max_val = np.percentile(arr,[10,90])
    clampFilt = sitk.ClampImageFilter()
    clampFilt.SetLowerBound(min_val)
    clampFilt.SetUpperBound(max_val)
    src_obj = clampFilt.Execute(src_obj)
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(out_max_val)
    rescalFilt.SetOutputMinimum(out_min_val)
    # Reads the image using SimpleITK
    tmp_obj = rescalFilt.Execute(sitk.Cast(src_obj, sitk.sitkFloat32))
    if mask_obj:
        mask = sitk.GetArrayFromImage(mask_obj)
        img = sitk.GetArrayFromImage(tmp_obj)

        dilated = morphology.binary_dilation(
            mask>0, morphology.ball(radius=3)
        )
        #border = np.logical_and(dilated==1,mask==0)
        #img[border==1]=out_max_val
        #img[border==0]=min_val
        img[dilated==0]=0

        tgt_obj = sitk.GetImageFromArray(img)
        tgt_obj.SetSpacing(tmp_obj.GetSpacing())
        tgt_obj.SetOrigin(tmp_obj.GetOrigin())
        tgt_obj.SetDirection(tmp_obj.GetDirection())
    else:
        tgt_obj = tmp_obj
    return tgt_obj

def resample(src_obj,out_size,method=sitk.sitkNearestNeighbor,out_val=0):
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
        method, out_val, src_obj.GetPixelID(),
    )
    return tgt_obj

def elastix_register_and_transform(fixed_image_file,moving_image_file,moving_list=[],default_pixel_value_str='-1000'):

    fixed = sitk.ReadImage(fixed_image_file)
    moving = sitk.ReadImage(moving_image_file)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetOutputDirectory('/tmp')

    defaultTranslationParameterMap = sitk.GetDefaultParameterMap("translation")
    defaultTranslationParameterMap['DefaultPixelValue'] = [default_pixel_value_str]
    defaultTranslationParameterMap['MaximumNumberOfIterations'] = ['512'] 
    defaultTranslationParameterMap['UseDirectionCosines'] = ['false']

    defaultAffineParameterMap = sitk.GetDefaultParameterMap("affine")
    defaultAffineParameterMap['DefaultPixelValue'] = [default_pixel_value_str]
    defaultAffineParameterMap['MaximumNumberOfIterations'] = ['512'] 
    defaultAffineParameterMap['UseDirectionCosines'] = ['false']
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(defaultTranslationParameterMap)
    parameterMapVector.append(defaultAffineParameterMap)
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.Execute()

    for item in moving_list:
        moving_file = item["moving_file"]
        moved_file = item["affine_only_moved_file"]
        out_pixel_value = item["out_val"]
        is_mask = item["is_mask"]

        transform_tuple = elastixImageFilter.GetTransformParameterMap()
        transform = list(transform_tuple)
        transform[-1]['DefaultPixelValue']=[str(out_pixel_value)]
        if is_mask:
            transform[-1]['FinalBSplineInterpolationOrder']=["0"]
            transform[-1]["ResultImagePixelType"] = ["int"]    
        # 
        # TODO: maybe something funky here? with int transformration
        # 
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

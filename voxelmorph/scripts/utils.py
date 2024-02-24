import sys
import numpy as np
import SimpleITK as sitk

def rescale_intensity(src_obj,min_val=-1000,max_val=1000):
    clampFilt = sitk.ClampImageFilter()
    clampFilt.SetLowerBound(min_val)
    clampFilt.SetUpperBound(max_val)
    src_obj = clampFilt.Execute(src_obj)
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(1.0)
    rescalFilt.SetOutputMinimum(0.0)
    # Reads the image using SimpleITK
    tgt_obj = rescalFilt.Execute(sitk.Cast(src_obj, sitk.sitkFloat32))
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
        method, 0, src_obj.GetPixelID()
    )
    return tgt_obj

if __name__ == "__main__":
    img_file = sys.argv[1]
    out_file = sys.argv[2]
    out_size = (128,128,128)
    src_obj = sitk.ReadImage(img_file)
    out_obj = resample(src_obj,out_size)
    sitk.WriteImage(out_obj,out_file)


'''
python utils.py \
    /radraid/pteng-public/tmp/RESEARCH/10123/MP200_AS_64_134/2018-08-29/tlc.nii.gz \
    /cvibraid/cvib2/Temp/tmp/down.nii.gz
'''
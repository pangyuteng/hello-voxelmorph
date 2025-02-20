
import os
import sys
import logging
logger = logging.getLogger('qc')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
import argparse
import hashlib
import traceback
import numpy as np
from scipy import ndimage
from skimage import measure
import SimpleITK as sitk
import json
from utils import imread, resample_img, lung_seg

from PIL import Image
import imageio
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.ndimage.morphology import distance_transform_edt
import skfmm

def vessel_seg(img_obj,lung_obj):

    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()
    
    lung_mask = sitk.GetArrayFromImage(lung_obj)

    arr_list = []
    for x in np.arange(2,4,1.0):
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(float(x))
        smoothed = gaussian.Execute(img_obj)
        myfilter = sitk.ObjectnessMeasureImageFilter()
        myfilter.SetBrightObject(True)
        myfilter.SetObjectDimension(1) # 1: lines (vessels),
        myfilter.SetAlpha(0.5) 
        myfilter.SetBeta(0.5)
        myfilter.SetGamma(5.0)
        tmp_obj = myfilter.Execute(smoothed)
        arr_list.append(sitk.GetArrayFromImage(tmp_obj))
    
    arr = np.max(np.array(arr_list),axis=0)
    arr[lung_mask==0]=0

    vessel_mask = np.zeros_like(arr)
    label_image, num = ndimage.measurements.label(arr>0)
    region = measure.regionprops(label_image)
    region = sorted(region,key=lambda x:x.area,reverse=True)
    for r in region[:2]: # pick largest 10
        mask = label_image == r.label
        vessel_mask[mask==1] = 1
        

    tmp = (skeletonize(vessel_mask) > 0).astype(np.int32)
    skeleton_obj = sitk.GetImageFromArray(tmp)
    skeleton_obj.SetSpacing(spacing)
    skeleton_obj.SetOrigin(origin)
    skeleton_obj.SetDirection(direction)

    vessel_obj = sitk.GetImageFromArray(vessel_mask)
    vessel_obj.SetSpacing(spacing)
    vessel_obj.SetOrigin(origin)
    vessel_obj.SetDirection(direction)

    return vessel_obj,skeleton_obj

def get_dist_map(mask_obj,seed_obj):

    iso_mask_obj = resample_img(mask_obj, out_spacing=[1.0, 1.0, 1.0], is_label=True)
    iso_seed_obj = resample_img(seed_obj, out_spacing=[1.0, 1.0, 1.0], is_label=True)
        
    img = sitk.GetArrayFromImage(iso_mask_obj)>0
    seed_map = sitk.GetArrayFromImage(iso_seed_obj)>0

    mask = ~img.astype(bool)
    img = img.astype(float)
    m = np.ones_like(img)
    m[seed_map==1] = 0
    m = np.ma.masked_array(m, mask)
    dist_map = skfmm.distance(m)

    spacing = iso_mask_obj.GetSpacing()
    origin = iso_mask_obj.GetOrigin()
    direction = iso_mask_obj.GetDirection()
    iso_dist_map_obj = sitk.GetImageFromArray(dist_map)
    iso_dist_map_obj.SetSpacing(spacing)
    iso_dist_map_obj.SetOrigin(origin)
    iso_dist_map_obj.SetDirection(direction)    
    
    dist_map_obj = resample_img(iso_dist_map_obj, out_spacing=mask_obj.GetSpacing(), is_label=True)

    return dist_map_obj

def main(tlc_file,rv_file,work_dir,is_plot=True):    
    
    tlc_obj = imread(tlc_file)
    rv_obj = imread(rv_file)
    
    rv_lung_mask_file = os.path.join(work_dir,"lung_mask.nii.gz")
    rv_lung_obj = imread(rv_lung_mask_file)

    tlc_lung_file = os.path.join(work_dir,"tlc_lung_mask.nii.gz")
    tlc_skeleton_file = os.path.join(work_dir,"tlc_vessel_skeleton.nii.gz")
    rv_skeleton_file = os.path.join(work_dir,"rv_vessel_skeleton.nii.gz")

    compute_vessel_skeleton = any([not os.path.exists(x) for x in [
        tlc_lung_file,tlc_skeleton_file,rv_skeleton_file
    ]])
    if compute_vessel_skeleton:

        tlc_lung_obj = lung_seg(tlc_obj)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(tlc_lung_file)
        writer.SetUseCompression(True)
        writer.Execute(tlc_lung_obj)
        
        tlc_vsl_obj, tlc_skl_obj = vessel_seg(tlc_obj,tlc_lung_obj)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(tlc_skeleton_file)
        writer.SetUseCompression(True)
        writer.Execute(tlc_skl_obj)

        rv_vsl_obj, rv_skl_obj = vessel_seg(rv_obj,rv_lung_obj)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(rv_skeleton_file)
        writer.SetUseCompression(True)
        writer.Execute(rv_skl_obj)
        
        for k,v in [('rv',rv_skl_obj),('tlc',tlc_skl_obj)]:
            tmp = sitk.GetArrayFromImage(v)
            target_shape = (tmp.shape[1],tmp.shape[2])
            mip_list = []
            for x in range(3):
                img = np.sum(tmp,axis=x).squeeze()
                img = (255*(img-np.min(img))/(np.max(img)-np.min(img))).clip(0,255).astype(np.uint8)
                img = np.array(Image.fromarray(img).resize(size=target_shape))        
                mip_list.append(img)

            png_file = os.path.join(work_dir,f"{k}_vessel_skeleton.png")
            tmp = np.concatenate(mip_list,axis=1)
            imageio.imwrite(png_file,tmp)


    registered_tlc_skeleton_file = os.path.join(work_dir,"registered_tlc_skeleton.nii.gz")
    deformation_file = os.path.join(work_dir,"registered_deformation.nii.gz")
    if not os.path.exists(registered_tlc_skeleton_file):
        
        tlc_skl_obj = imread(tlc_skeleton_file)

        parameter_map_0 = sitk.ReadParameterFile(os.path.join(work_dir,"elastix/TransformParameters.0.txt"))
        parameter_map_1 = sitk.ReadParameterFile(os.path.join(work_dir,"elastix/TransformParameters.1.txt"))
        parameter_map_2 = sitk.ReadParameterFile(os.path.join(work_dir,"elastix/TransformParameters.2.txt"))

        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(parameter_map_0)
        parameterMapVector.append(parameter_map_1)
        parameterMapVector.append(parameter_map_2)

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.SetTransformParameterMap(parameterMapVector)
        transformixImageFilter.ComputeDeterminantOfSpatialJacobianOn()
        #transformixImageFilter.ComputeSpatialJacobianOn()

        transformixImageFilter.SetMovingImage(tlc_skl_obj)
        transformixImageFilter.LogToConsoleOn()
        logdir = os.path.join(work_dir,'transformix')
        os.makedirs(logdir,exist_ok=True)
        transformixImageFilter.SetOutputDirectory(logdir)
        transformixImageFilter.Execute()

        transformed = transformixImageFilter.GetResultImage()
        deformation = transformixImageFilter.GetDeformationField()

        sitk.WriteImage(deformation,deformation_file)

        arr = sitk.GetArrayFromImage(transformed)
        arr = (arr > 0.1).astype(np.int32)

        try:
            print(np.unique(arr))
            assert([0,1]==list(np.unique(arr)))
        except:
            raise ValueError("transformixImageFilter likely failed - found no skeleton in transformed image!")

        spacing = transformed.GetSpacing()
        origin = transformed.GetOrigin()
        direction = transformed.GetDirection()
        myobj = sitk.GetImageFromArray(arr)
        myobj.SetSpacing(spacing)
        myobj.SetOrigin(origin)
        myobj.SetDirection(direction)

        sitk.WriteImage(myobj, registered_tlc_skeleton_file)

    registered_dist_map_file = os.path.join(work_dir,"registered_dist_map.nii.gz")
    qc_json_file = os.path.join(work_dir,"qc.json")
    if not os.path.exists(qc_json_file):
        registered_tlc_skl_obj = imread(registered_tlc_skeleton_file)    
        rv_skl_obj = imread(rv_skeleton_file)
        rv_lung_obj = imread(rv_lung_mask_file)        

        # generate tlc-skeleton seeded map
        dist_map_obj = get_dist_map(rv_lung_obj,registered_tlc_skl_obj)
        sitk.WriteImage(dist_map_obj, registered_dist_map_file)
                
        rv_skeleton = sitk.GetArrayFromImage(rv_skl_obj)
        dist_map = sitk.GetArrayFromImage(dist_map_obj)
        # get dist_map value at each rv skeleton
        dist_map[rv_skeleton==0]=np.nan
        my_mean = float(np.nanmean(dist_map))
        my_std = float(np.nanstd(dist_map))

        print(f"mean(sd) distance of rv and tlc skeletons: {my_mean}({my_std}) mm")

        qc_dict = dict(
            mean_mm=my_mean,
            std_mm=my_std,
        )
        
        with open(qc_json_file,'w') as f:
            f.write(json.dumps(qc_dict))

        return qc_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('tlc_file', type=str)
    parser.add_argument('rv_file', type=str)
    parser.add_argument('-w','--work_dir', type=str,default=None)

    args = parser.parse_args()
    tlc_file = os.path.abspath(args.tlc_file)
    rv_file = os.path.abspath(args.rv_file)
    work_dir = args.work_dir

    # create cache dir
    if work_dir is None:
        h = hashlib.md5()        
        h.update(tlc_file.encode('utf-8'))
        h.update(rv_file.encode('utf-8'))
        hash = h.hexdigest()
        work_dir = os.path.abspath(f'work_dir/{hash}')
        os.makedirs(work_dir,exist_ok=True)

    main(tlc_file,rv_file,work_dir)

'''
python qc.py sample_images/tlc.nii.gz sample_images/rv.nii.gz
'''